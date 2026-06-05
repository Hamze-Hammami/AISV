#!/usr/bin/env python3
"""
ROS2 node for Depth-Anything V2 with TensorRT.

Subscribes to RGB images, processes them through Depth-Anything V2 TensorRT engine,
and publishes depth images in a format compatible with vision.

Requires: tensorrt, pycuda, opencv-python, numpy, rclpy
"""

import os
import time
import sys
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import tensorrt as trt
import pycuda.driver as cuda
cuda.init()
import pycuda.autoinit  # GPU context

class DepthAnythingNode(Node):
    """
    ROS2 node for Depth-Anything V2 TensorRT inference.
    
    Subscribes to:
        - RGB image topic (configurable)
    
    Publishes:
        - 16UC1 depth image (cm) compatible with vision
        - Optional visualization output
    """
    
    def __init__(self):
        super().__init__('depth_anything_node')
        
        # Declare parameters
        self.declare_parameter('rgb_topic', '/cam')
        self.declare_parameter('depth_topic', '/dpt/depth')
        self.declare_parameter('viz_topic', '/depth_viz')
        self.declare_parameter('engine_path', 'models/depth_anything_v2_vitb_vkitti_fp16.engine')
        self.declare_parameter('model_path', '')
        self.declare_parameter('max_depth', 65.0)
        self.declare_parameter('input_hw', [518, 518])
        self.declare_parameter('enable_viz', True)
        self.declare_parameter('viz_colormap', 'INFERNO')
        self.declare_parameter('publish_rate', 30.0)
        
        # Added visualization parameters
        self.declare_parameter('viz_min_depth', 0.5)  # Min depth for visualization (meters)
        self.declare_parameter('viz_max_depth', 30.0)  # Max depth for visualization (meters)
        
        # Get parameters
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.viz_topic = self.get_parameter('viz_topic').get_parameter_value().string_value
        self.engine_path_param = self.get_parameter('engine_path').get_parameter_value().string_value
        self.model_path_param = self.get_parameter('model_path').get_parameter_value().string_value
        self.max_depth = self.get_parameter('max_depth').get_parameter_value().double_value
        self.input_hw = self.get_parameter('input_hw').get_parameter_value().integer_array_value
        self.enable_viz = self.get_parameter('enable_viz').get_parameter_value().bool_value
        self.viz_colormap_str = self.get_parameter('viz_colormap').get_parameter_value().string_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        
        # Get visualization depth range parameters
        self.viz_min_depth = self.get_parameter('viz_min_depth').get_parameter_value().double_value
        self.viz_max_depth = self.get_parameter('viz_max_depth').get_parameter_value().double_value
        
        # If input_hw is empty, use default values
        if not self.input_hw:
            self.input_hw = [518, 518]
        
        # Warn if max_depth exceeds 16-bit capacity (for cm, max is 655.35m)
        if self.max_depth * 100.0 > 65535:
            self.get_logger().warn(f"max_depth of {self.max_depth}m exceeds 16-bit cm capacity (655.35m). Values will be clipped.")
        
        # Normalization parameters
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Set the colormap for visualization - use INFERNO as default
        colormap_name = f'COLORMAP_{self.viz_colormap_str}'
        if hasattr(cv2, colormap_name):
            self.colormap = getattr(cv2, colormap_name)
        else:
            self.get_logger().warn(f"Colormap {self.viz_colormap_str} not found, using INFERNO")
            self.colormap = cv2.COLORMAP_INFERNO
        
        # Set up CV Bridge
        self.bridge = CvBridge()
        
        # Initialize variables for TensorRT (avoid naming conflicts)
        self.trt_engine = None
        self.trt_ctx = None
        self.input_buffers = None
        self.output_buffers = None
        self.cuda_stream = None
        
        # Create publishers
        self.depth_publisher = self.create_publisher(Image, self.depth_topic, 10)
        if self.enable_viz:
            self.viz_publisher = self.create_publisher(Image, self.viz_topic, 10)
        
        # Store latest frame for timer-based publishing
        self.latest_frame = None
        self.latest_header = None
        
        # Set up timer for regular processing
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.process_frame)
        
        # Load the TensorRT model after everything else is set up
        self.load_model()
        
        # Create subscription last to ensure model is loaded first
        self.rgb_subscription = self.create_subscription(
            Image,
            self.rgb_topic,
            self.rgb_callback,
            10)
        
        self.get_logger().info(f'Depth-Anything node initialized - publishing to {self.depth_topic}')
        self.get_logger().info(f'Using depth scaling: 1 uint16 value = 1 centimeter')
        self.get_logger().info(f'Visualization depth range: {self.viz_min_depth}m to {self.viz_max_depth}m')
    
    def load_model(self):
        """Load the TensorRT engine."""
        try:
            # Look for engine in multiple possible locations
            engine_paths = [
                self.engine_path_param,
                os.path.join(self.model_path_param, os.path.basename(self.engine_path_param)),
                os.path.join('models', os.path.basename(self.engine_path_param)),
                os.path.join('src/vision/models', os.path.basename(self.engine_path_param)),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models', 
                           os.path.basename(self.engine_path_param))
            ]
            
            engine_loaded = False
            for path in engine_paths:
                if os.path.exists(path):
                    self.get_logger().info(f"Loading TensorRT engine from: {path}")
                    
                    # Initialize TensorRT
                    logger = trt.Logger(trt.Logger.ERROR)
                    with open(path, "rb") as f, trt.Runtime(logger) as rt:
                        self.trt_engine = rt.deserialize_cuda_engine(f.read())
                    
                    # Create execution context
                    self.trt_ctx = self.trt_engine.create_execution_context()
                    
                    # Allocate buffers
                    self.input_buffers, self.output_buffers, self.cuda_stream = self.alloc_buffers()
                    
                    self.get_logger().info(f"TensorRT engine loaded successfully from {path}")
                    engine_loaded = True
                    break
            
            if not engine_loaded:
                self.get_logger().error(f"TensorRT engine not found in any of these locations: {engine_paths}")
                raise FileNotFoundError(f"Engine not found in any of: {engine_paths}")
                
        except Exception as e:
            self.get_logger().error(f"Failed to load TensorRT engine: {e}")
            raise
    
    def alloc_buffers(self):
        """Allocate device and host buffers."""
        inputs, outputs = {}, {}
        stream = cuda.Stream()

        # Determine number of tensors based on available API
        if hasattr(self.trt_engine, "num_io_tensors"):
            # TensorRT 10+ API
            n_tensors = self.trt_engine.num_io_tensors
            for i in range(n_tensors):
                name = self.trt_engine.get_tensor_name(i)
                shape = self.trt_engine.get_tensor_shape(name)
                dtype = trt.nptype(self.trt_engine.get_tensor_dtype(name))
                mode = self.trt_engine.get_tensor_mode(name)
                is_input = (mode == trt.TensorIOMode.INPUT)
                
                # Allocate host/device buffers
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                device = cuda.mem_alloc(host.nbytes)
                
                # Register device pointer
                self.trt_ctx.set_tensor_address(name, int(device))
                
                entry = {"host": host, "device": device, "shape": shape}
                (inputs if is_input else outputs)[name] = entry
        
        elif hasattr(self.trt_engine, "num_bindings"):
            # TensorRT 8.x legacy binding API
            n_tensors = self.trt_engine.num_bindings
            for i in range(n_tensors):
                name = self.trt_engine.get_binding_name(i)
                shape = self.trt_engine.get_binding_shape(i)
                dtype = trt.nptype(self.trt_engine.get_binding_dtype(i))
                is_input = self.trt_engine.binding_is_input(i)
                
                # Allocate host/device buffers
                host = cuda.pagelocked_empty(int(np.prod(shape)), dtype)
                device = cuda.mem_alloc(host.nbytes)
                
                entry = {"host": host, "device": device, "shape": shape}
                (inputs if is_input else outputs)[name] = entry
        else:
            raise RuntimeError("Cannot determine number of tensors from engine API")

        return inputs, outputs, stream
    
    def preprocess(self, bgr):
        """Preprocess BGR image for the model."""
        # Convert input_hw to a shape that resize can use
        input_size = (int(self.input_hw[1]), int(self.input_hw[0]))  # w, h
        
        rgb = cv2.cvtColor(cv2.resize(bgr, input_size), cv2.COLOR_BGR2RGB) / 255.0
        norm = (rgb - self.mean) / self.std
        return norm.transpose(2, 0, 1)[None].astype(np.float32)
    
    def colourise(self, depth_cm):
        """
        Map uint16 depth in centimeters to BGR heatmap with improved visualization.
        
        Args:
            depth_cm: Depth image in centimeters as uint16
        
        Returns:
            BGR visualization with enhanced color mapping
        """
        # Convert from uint16 centimeters to float meters
        depth_m = depth_cm.astype(np.float32) / 100.0
        
        # Clip to configured visualization range
        depth_m = np.clip(depth_m, self.viz_min_depth, self.viz_max_depth)
        
        # Normalize to 0-255 range for visualization
        depth_norm = ((depth_m - self.viz_min_depth) / (self.viz_max_depth - self.viz_min_depth) * 255).astype(np.uint8)
        
        # Apply colormap
        colorized = cv2.applyColorMap(depth_norm, self.colormap)
        
        # Add improved visualization: mark areas with no depth (0) in black
        mask_no_depth = (depth_cm == 0)
        colorized[mask_no_depth] = [0, 0, 0]
        
        # Add depth scale indicators
        h, w = depth_m.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Draw depth scale bar on the right side
        bar_width = 20
        bar_padding = 10
        bar_x = w - bar_width - bar_padding
        bar_height = h - 2 * bar_padding
        
        # Draw colorbar
        for y in range(bar_padding, bar_padding + bar_height):
            depth_value = self.viz_max_depth - (y - bar_padding) / bar_height * (self.viz_max_depth - self.viz_min_depth)
            depth_idx = int(((depth_value - self.viz_min_depth) / (self.viz_max_depth - self.viz_min_depth)) * 255)
            depth_idx = np.clip(depth_idx, 0, 255)
            color = cv2.applyColorMap(np.array([[depth_idx]], dtype=np.uint8), self.colormap)[0][0]
            cv2.line(colorized, (bar_x, y), (bar_x + bar_width, y), color.tolist(), 1)
        
        # Draw colorbar frame
        cv2.rectangle(colorized, (bar_x, bar_padding), (bar_x + bar_width, bar_padding + bar_height), (255, 255, 255), 1)
        
        # Add labels
        # Top (max depth)
        label = f"{self.viz_max_depth:.1f}m"
        cv2.putText(colorized, label, (bar_x - 50, bar_padding + 10), font, font_scale, (255, 255, 255), thickness)
        
        # Middle
        mid_depth = (self.viz_min_depth + self.viz_max_depth) / 2
        label = f"{mid_depth:.1f}m"
        cv2.putText(colorized, label, (bar_x - 50, bar_padding + bar_height // 2), font, font_scale, (255, 255, 255), thickness)
        
        # Bottom (min depth)
        label = f"{self.viz_min_depth:.1f}m"
        cv2.putText(colorized, label, (bar_x - 50, bar_padding + bar_height - 10), font, font_scale, (255, 255, 255), thickness)
        
        return colorized
    
    def rgb_callback(self, msg):
        """Store the latest RGB frame for processing."""
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.latest_header = msg.header
        except Exception as e:
            self.get_logger().error(f"Error in RGB callback: {e}")
    
    def process_frame(self):
        """Process the latest frame and publish depth."""
        if self.latest_frame is None or self.trt_engine is None:
            return
        
        try:
            # Create a local copy to avoid race conditions
            frame = self.latest_frame.copy()
            header = self.latest_header
            
            t0 = time.time()
            
            # Get input/output tensor names
            in_name = next(iter(self.input_buffers))
            out_name = next(iter(self.output_buffers))
            
            # Preprocess and copy to device
            img = self.preprocess(frame)
            np.copyto(self.input_buffers[in_name]["host"], img.ravel())
            cuda.memcpy_htod_async(self.input_buffers[in_name]["device"], self.input_buffers[in_name]["host"], self.cuda_stream)
            
            # Inference: use appropriate execute method based on TensorRT version
            if hasattr(self.trt_ctx, "execute_async_v3"):
                # TensorRT 10+
                self.trt_ctx.execute_async_v3(self.cuda_stream.handle)
            else:
                # Legacy binding-based v2
                bindings = [int(self.input_buffers[in_name]["device"]), int(self.output_buffers[out_name]["device"])]
                self.trt_ctx.execute_async_v2(bindings, self.cuda_stream.handle)
            
            # Copy output back
            cuda.memcpy_dtoh_async(self.output_buffers[out_name]["host"], self.output_buffers[out_name]["device"], self.cuda_stream)
            self.cuda_stream.synchronize()
            
            # Post-process depth
            depth = self.output_buffers[out_name]["host"].reshape(self.output_buffers[out_name]["shape"])[0]
            
            # Convert to uint16 (cm) - Keep the 100.0 scaling for accuracy
            depth_cm = np.clip(depth * 100.0, 0, 65535).astype(np.uint16)
            
            # Resize to match input frame resolution if needed
            h, w = frame.shape[:2]
            if depth_cm.shape[0] != h or depth_cm.shape[1] != w:
                depth_cm = cv2.resize(depth_cm, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Create and publish depth message (16UC1 format)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_cm, encoding="16UC1")
            depth_msg.header = header
            self.depth_publisher.publish(depth_msg)
            
            # Create and publish visualization if enabled
            if self.enable_viz:
                # Use our improved colorization method
                depth_viz = self.colourise(depth_cm)
                depth_viz_msg = self.bridge.cv2_to_imgmsg(depth_viz, encoding="bgr8")
                depth_viz_msg.header = header
                self.viz_publisher.publish(depth_viz_msg)
            
            # Calculate and log FPS
            inference_time = time.time() - t0
            fps = 1.0 / inference_time
            
            # Log FPS periodically to avoid log spam
            if int(time.time()) % 5 == 0:
                self.get_logger().info(f'Depth inference: {fps:.1f} FPS')
            
        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = DepthAnythingNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"Error initializing DepthAnythingNode: {e}", file=sys.stderr)
        rclpy.logging.get_logger("depth_anything_node").error(f"Initialization error: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()