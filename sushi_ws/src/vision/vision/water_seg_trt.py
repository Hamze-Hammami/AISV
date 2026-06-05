#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda_driver
import time
import traceback
import os
from PIL import Image as PILImage
from torchvision import transforms

class WaterSegmentationModel:
    def __init__(self, model_path, device, cuda_context):
        self.device = device
        self.cuda_context = cuda_context
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self.load_engine(model_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self.allocate_buffers(self.engine)

    def load_engine(self, engine_file_path):
        with open(engine_file_path, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise ValueError("Failed to load TensorRT engine.")
            return engine

    def allocate_buffers(self, engine):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda_driver.Stream()
        
        profile_index = 0
        num_bindings = engine.num_io_tensors
        
        for binding_index in range(num_bindings):
            name = engine.get_tensor_name(binding_index)
            dtype = engine.get_tensor_dtype(name)
            shape = engine.get_tensor_shape(name)
            size = trt.volume(shape)
            
            host_mem = np.zeros(size, dtype=trt.nptype(dtype))
            device_mem = cuda_driver.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            
            if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs.append({
                    'host': host_mem, 
                    'device': device_mem,
                    'name': name,
                    'shape': shape
                })
            else:
                outputs.append({
                    'host': host_mem, 
                    'device': device_mem,
                    'name': name,
                    'shape': shape
                })
                
        return inputs, outputs, bindings, stream

    def predict(self, image):
        try:
            # Convert to RGB if image is in BGR format (OpenCV default)
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
                
            image_pil = PILImage.fromarray(rgb_image)
            image_tensor = self.transform(image_pil).unsqueeze(0).numpy()
            
            self.cuda_context.push()
            
            cuda_driver.memcpy_htod_async(self.inputs[0]['device'], image_tensor.ravel(), self.stream)
            
            self.context.set_tensor_address(self.inputs[0]['name'], int(self.inputs[0]['device']))
            self.context.set_tensor_address(self.outputs[0]['name'], int(self.outputs[0]['device']))
            
            status = self.context.execute_async_v3(stream_handle=int(self.stream.handle))
            if not status:
                raise RuntimeError("Inference failed")
                
            self.stream.synchronize()
            
            cuda_driver.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
            self.stream.synchronize()
            
            self.cuda_context.pop()

            # Handle possible different output shapes
            output_data = self.outputs[0]['host']
            if output_data.size == 256*256:
                # Keep as probability map initially (don't threshold yet)
                predicted_mask = output_data.reshape(256, 256)
            else:
                # Alternative shape handling if needed
                predicted_mask = output_data.reshape(1, 256, 256)[0]
            
            # Convert to float32 for processing
            predicted_mask = predicted_mask.astype(np.float32)
            
            # Apply bilateral filtering to smooth while preserving edges
            # First convert to 0-255 range for OpenCV functions
            mask_255 = (predicted_mask * 255).astype(np.uint8)
            # Apply bilateral filter to smooth while preserving edges
            smoothed_mask = cv2.bilateralFilter(mask_255, 9, 75, 75)
            # Convert back to probability map
            predicted_mask = smoothed_mask.astype(np.float32) / 255.0
            
            # Apply Gaussian blur to further smooth boundaries
            predicted_mask = cv2.GaussianBlur(predicted_mask, (5, 5), 0)
            
            # Now threshold to get binary mask
            binary_mask = (predicted_mask > 0.5).astype(np.uint8)
            
            # Apply lighter morphological operations
            kernel = np.ones((5, 5), np.uint8)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            
            # Find contours and smooth them
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create an empty mask for the smoothed contours
            smooth_mask = np.zeros_like(binary_mask)
            
            # Process each contour to smooth it
            for contour in contours:
                # Skip very small contours
                if cv2.contourArea(contour) < 500:
                    continue
                    
                # Apply contour approximation for smoother edges
                epsilon = 0.002 * cv2.arcLength(contour, True)
                approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                
                # Further smooth with spline interpolation for larger contours
                if len(approx_contour) > 10 and cv2.contourArea(contour) > 2000:
                    # Convert to array of points
                    points = approx_contour.reshape(-1, 2)
                    
                    # Append the first point to close the loop
                    points = np.vstack([points, points[0]])
                    
                    # Create a smoothed version with more points
                    t = np.linspace(0, 1, len(points))
                    t_new = np.linspace(0, 1, len(points) * 3)
                    
                    # Create smooth x and y coordinates
                    x_smooth = np.interp(t_new, t, points[:, 0])
                    y_smooth = np.interp(t_new, t, points[:, 1])
                    
                    # Apply Gaussian smoothing to the interpolated points
                    x_smooth = cv2.GaussianBlur(x_smooth.reshape(-1, 1), (5, 1), 0).flatten()
                    y_smooth = cv2.GaussianBlur(y_smooth.reshape(-1, 1), (5, 1), 0).flatten()
                    
                    # Combine back into a contour format
                    smooth_contour = np.column_stack([x_smooth, y_smooth]).astype(np.int32)
                    
                    # Draw the smoothed contour
                    cv2.drawContours(smooth_mask, [smooth_contour], 0, 1, -1)
                else:
                    # For smaller contours, just use the approximation
                    cv2.drawContours(smooth_mask, [approx_contour], 0, 1, -1)
            
            # If no contours were drawn, revert to the original binary mask
            if np.sum(smooth_mask) == 0:
                smooth_mask = binary_mask
                
            # Fill holes in the mask
            # First invert the mask
            inv_mask = 1 - smooth_mask
            # Label connected components in the inverted mask
            num_labels, labels_im = cv2.connectedComponents(inv_mask)
            # Create a mask for holes (connected components in inverted mask that don't touch the border)
            holes_mask = np.zeros_like(inv_mask)
            h, w = inv_mask.shape
            border_labels = set()
            
            # Find labels that touch the border
            border_pixels = np.concatenate([
                labels_im[0, :],        # top row
                labels_im[-1, :],       # bottom row
                labels_im[:, 0],        # leftmost column
                labels_im[:, -1]        # rightmost column
            ])
            border_labels = set(np.unique(border_pixels))
            
            # Fill in holes (components that don't touch the border)
            for label in range(1, num_labels):
                if label not in border_labels:
                    holes_mask[labels_im == label] = 1
            
            # Add filled holes to the smooth mask
            final_mask = smooth_mask.copy()
            final_mask[holes_mask == 1] = 1
            
            # Clean up small isolated regions
            num_labels, labels_im = cv2.connectedComponents(final_mask)
            for label in range(1, num_labels):
                if np.sum(labels_im == label) < 500:
                    final_mask[labels_im == label] = 0

            return final_mask
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            traceback.print_exc()
            return None

    def create_mask_visualization(self, frame, water_mask, draw_contours=True):
        try:
            if water_mask is None:
                return frame.copy()

            if water_mask.shape[:2] != frame.shape[:2]:
                water_mask = cv2.resize(water_mask, (frame.shape[1], frame.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            
            # Convert frame to proper format
            # If frame is BGR (from OpenCV), convert to RGB for processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Create a blue overlay for water
            water_overlay = np.zeros_like(frame_rgb)
            water_overlay[:, :, 0] = 255  # Blue channel in RGB
            
            # Resize mask to match original image dimensions
            mask_resized = water_mask
            binary_mask = (mask_resized > 0.5).astype(np.uint8)
            
            # Ensure binary mask is broadcastable to 3 channels
            binary_mask_3d = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
            
            # Apply alpha blending
            alpha = 0.5
            
            # Create blended version of the original with blue overlay
            blended = cv2.addWeighted(frame_rgb, 1-alpha, water_overlay, alpha, 0)
            
            # Apply mask to create final visualization
            vis = frame_rgb.copy()
            vis = np.where(binary_mask_3d, blended, vis)
            
            # Convert back to BGR for OpenCV
            vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
            
            # Draw contour around water if requested
            if draw_contours:
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis_bgr, contours, -1, (0, 255, 255), 2)  # Yellow contour
            
            return vis_bgr
            
        except Exception as e:
            print(f"Error in visualization: {e}")
            traceback.print_exc()
            return frame.copy()

    def overlay_water_segmentation(self, frame, water_mask, draw_contours=True):
        return self.create_mask_visualization(frame, water_mask, draw_contours)

    def __del__(self):
        try:
            if hasattr(self, 'cuda_context'):
                self.cuda_context.pop()
        except Exception as e:
            print(f"Error during cleanup: {e}")

class WaterSegNode(Node):
    def __init__(self):
        super().__init__('water_seg_node')
        
        self.declare_parameter('source_topic', '/cam')
        self.declare_parameter('water_seg_size', 256)
        self.declare_parameter('min_water_area', 500)
        self.declare_parameter('morphology_kernel', 7)
        self.declare_parameter('morphology_iterations', 15)
        self.declare_parameter('water_threshold', 0.5)
        self.declare_parameter('water_overlay_alpha', 0.7)
        self.declare_parameter('model_path', '')
        self.declare_parameter('prediction_interval', 0.0)  # Default to 0, which means predict every frame
        self.declare_parameter('force_prediction', True)    # Force prediction on every frame
        self.declare_parameter('raw_output', False)         # Skip visualization processing
        
        self.source_topic = self.get_parameter('source_topic').get_parameter_value().string_value
        self.water_seg_size = self.get_parameter('water_seg_size').get_parameter_value().integer_value
        self.min_water_area = self.get_parameter('min_water_area').get_parameter_value().integer_value
        self.morphology_kernel = self.get_parameter('morphology_kernel').get_parameter_value().integer_value
        self.morphology_iterations = self.get_parameter('morphology_iterations').get_parameter_value().integer_value
        self.water_threshold = self.get_parameter('water_threshold').get_parameter_value().double_value
        self.water_overlay_alpha = self.get_parameter('water_overlay_alpha').get_parameter_value().double_value
        self.model_path_param = self.get_parameter('model_path').get_parameter_value().string_value
        self.prediction_interval = self.get_parameter('prediction_interval').get_parameter_value().double_value
        self.force_prediction = self.get_parameter('force_prediction').get_parameter_value().bool_value
        self.raw_output = self.get_parameter('raw_output').get_parameter_value().bool_value
        
        self.bridge = CvBridge()
        
        self.mask_pub = self.create_publisher(Image, 'water_mask', 10)
        self.visualization_pub = self.create_publisher(Image, 'water_visualization', 10)
        
        self.image_sub = self.create_subscription(Image, self.source_topic, self.image_callback, 10)
        
        self.initialize_cuda()
        self.load_model()
        
        # Initialize variables for tracking prediction timing
        self.last_prediction_time = 0.0
        self.last_water_mask = None
        self.last_frame = None
        self.last_header = None
        
        # Set up message counters for debugging
        self.frame_count = 0
        self.prediction_count = 0
        
        # Print configuration
        self.get_logger().info(f'Water segmentation node initialized with:')
        self.get_logger().info(f' - Source topic: {self.source_topic}')
        self.get_logger().info(f' - Prediction interval: {self.prediction_interval}s')
        self.get_logger().info(f' - Force prediction on every frame: {self.force_prediction}')
        self.get_logger().info(f' - Raw output mode (simple visualization): {self.raw_output}')
    
    def initialize_cuda(self):
        try:
            cuda_driver.init()
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.cuda_context = cuda_driver.Device(0).retain_primary_context()
            self.cuda_context.push()
            self.get_logger().info(f"CUDA initialized successfully on device: {self.device}")
        except Exception as e:
            self.get_logger().error(f"CUDA initialization failed: {e}")
            traceback.print_exc()
            self.cuda_context = None
    
    def load_model(self):
        if self.cuda_context is None:
            self.get_logger().error("CUDA context not initialized, cannot load model")
            self.model = None
            return
            
        try:
            model_paths = [
                'models/water_student_model2.engine',
                'src/vision/models/water_student_model2.engine',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/water_student_model2.engine'),
            ]
            
            if self.model_path_param and os.path.exists(os.path.join(self.model_path_param, 'water_student_model2.engine')):
                model_paths.insert(0, os.path.join(self.model_path_param, 'water_student_model2.engine'))
            
            for path in model_paths:
                if os.path.exists(path):
                    self.model = WaterSegmentationModel(path, self.device, self.cuda_context)
                    self.get_logger().info(f"Water segmentation model loaded successfully from {path}")
                    return
                    
            self.get_logger().error(f"Model not found in any of these locations: {model_paths}")
            self.model = None
                
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            traceback.print_exc()
            self.model = None
    
    def image_callback(self, msg):
        try:
            if self.model is None:
                self.get_logger().error("No water segmentation model loaded, skipping frame")
                return
                
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            current_time = time.time()
            self.frame_count += 1
            
            # Store the latest frame and header for future visualizations
            self.last_frame = frame
            self.last_header = msg.header
            
            # Determine whether to predict on this frame
            should_predict = False
            
            # If force_prediction is True, predict on every frame
            if self.force_prediction:
                should_predict = True
                
            # If prediction_interval is set, check if enough time has passed
            elif self.prediction_interval <= 0.0:
                # Interval of 0 or negative means predict every frame
                should_predict = True
            elif self.last_water_mask is None or (current_time - self.last_prediction_time) >= self.prediction_interval:
                should_predict = True
            
            if should_predict:
                # Perform water mask prediction
                self.prediction_count += 1
                water_mask = self.model.predict(frame)
                if water_mask is None:
                    self.get_logger().error("Water mask prediction failed")
                    return
                
                # The model.predict now returns a fully processed smooth mask
                # No need for additional morphological operations here
                
                h, w = frame.shape[:2]
                if water_mask.shape != (h, w):
                    water_mask = cv2.resize(water_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                
                # Update timestamp and store the latest mask
                self.last_prediction_time = current_time
                self.last_water_mask = water_mask
                
                # Print prediction stats every 100 frames
                if self.prediction_count % 100 == 0:
                    self.get_logger().info(f"Predictions: {self.prediction_count}/{self.frame_count} frames ({self.prediction_count/self.frame_count*100:.1f}%)")
            else:
                # Use cached water mask
                water_mask = self.last_water_mask
                
                # If we have a mask but its size doesn't match the current frame, resize it
                if water_mask is not None:
                    h, w = frame.shape[:2]
                    if water_mask.shape != (h, w):
                        water_mask = cv2.resize(water_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            
            # Only publish if we have a valid water mask
            if water_mask is not None:
                # Create visualization and publish results
                try:
                    mask_msg = self.bridge.cv2_to_imgmsg(water_mask * 255, encoding="mono8")
                    mask_msg.header = msg.header
                    self.mask_pub.publish(mask_msg)
                    
                    # Use the visualization function that matches the original code
                    # Pass the raw_output parameter to control contour drawing
                    water_overlay = self.model.create_mask_visualization(frame, water_mask, draw_contours=not self.raw_output)
                    
                    vis_msg = self.bridge.cv2_to_imgmsg(water_overlay, "bgr8")
                    vis_msg.header = msg.header
                    self.visualization_pub.publish(vis_msg)
                except Exception as e:
                    self.get_logger().error(f"Error publishing results: {e}")
                    traceback.print_exc()
            
        except Exception as e:
            self.get_logger().error(f"Error processing frame: {e}")
            traceback.print_exc()
    
    def __del__(self):
        try:
            if hasattr(self, 'cuda_context') and self.cuda_context is not None:
                self.cuda_context.pop()
        except Exception as e:
            self.get_logger().error(f"Error during cleanup: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = WaterSegNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down water segmentation node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()