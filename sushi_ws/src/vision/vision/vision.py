#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, Range
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, Point32, Vector3
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String, Float32MultiArray
from sensor_msgs.msg import PointCloud
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
import threading
import time
from collections import deque
import tf2_ros
import os
import traceback
import json

# Import SORT tracker and TF functions
from vision.sort import Sort
from vision.tf import publish_static_transform, publish_dynamic_transform, publish_camera_marker

class FrameBuffer:
    def __init__(self, maxsize=5):  # Reduced default maxsize from 10 to 5
        self.rgb_frames = deque(maxlen=maxsize)
        self.depth_frames = deque(maxlen=maxsize)
        self.water_masks = deque(maxlen=maxsize)
        self.detections = deque(maxlen=maxsize)
        self.obstacles = deque(maxlen=maxsize)
        self.lock = threading.Lock()
        
        # Add a flag to ensure we process each frame only once
        self.processed_timestamps = set()
        # Add memory monitoring
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 10.0  # Run cleanup every 10 seconds
        
        # Add timestamp tracking for obstacle timeouts
        self.last_detection_time = time.time()
        self.last_obstacle_time = time.time()
        self.last_object_detection_time = time.time()  # Separate object detection time
        self.last_obstacle_detection_time = time.time()  # Separate obstacle detection time

    def add_rgb(self, frame, timestamp):
        with self.lock:
            # Store a reference to the timestamp for tracking
            ts_key = timestamp.sec * 1000000000 + timestamp.nanosec
            # Optionally resize large images before storing
            if frame.shape[0] > 480:  # If height > 480, resize to conserve memory
                scale_factor = 480.0 / frame.shape[0]
                new_width = int(frame.shape[1] * scale_factor)
                frame = cv2.resize(frame, (new_width, 480))
            self.rgb_frames.append((frame, timestamp))
            self._check_cleanup()

    def add_depth(self, frame, timestamp):
        with self.lock:
            # Store original depth values without conversion to preserve precision
            # Only resize if needed to conserve memory
            if frame.shape[0] > 480:
                scale_factor = 480.0 / frame.shape[0]
                new_width = int(frame.shape[1] * scale_factor)
                frame = cv2.resize(frame, (new_width, 480))
            self.depth_frames.append((frame, timestamp))
            self._check_cleanup()

    def add_water_mask(self, mask, timestamp):
        with self.lock:
            # Ensure mask is binary and uint8 to minimize memory usage
            if mask.dtype != np.uint8:
                mask = mask.astype(np.uint8)
            self.water_masks.append((mask, timestamp))
            self._check_cleanup()

    def add_detections(self, detections, timestamp):
        with self.lock:
            # Split detections by type and update respective timestamps
            object_detections = []
            obstacle_detections = []
            
            for det in detections:
                if det.get('type', 'object') == 'object':
                    object_detections.append(det)
                else:
                    obstacle_detections.append(det)
            
            if object_detections:
                self.last_object_detection_time = time.time()
            if obstacle_detections:
                self.last_obstacle_detection_time = time.time()
                
            # Store all detections together
            filtered_detections = [d for d in detections if d['score'] > 0.3]
            self.detections.append((filtered_detections, timestamp))
            self._check_cleanup()
            
    def add_obstacles(self, obstacles, timestamp):
        with self.lock:
            self.obstacles.append((obstacles, timestamp))
            self.last_obstacle_time = time.time()  # Update timestamp
            self._check_cleanup()
    
    def _check_cleanup(self):
        """Periodically check and perform memory cleanup"""
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self.last_cleanup_time = current_time
            # Clear processed timestamps that are older than recent frames
            if self.rgb_frames:
                oldest_timestamp = self.rgb_frames[0][1]
                oldest_ts = oldest_timestamp.sec * 1000000000 + oldest_timestamp.nanosec
                self.processed_timestamps = {ts for ts in self.processed_timestamps if ts >= oldest_ts}

    def get_latest_data(self):
        with self.lock:
            if not self.rgb_frames or not self.depth_frames:
                return None, None, None, None, None
            
            # Get the latest from each buffer
            rgb_frame, rgb_ts = self.rgb_frames[-1]
            depth_frame, depth_ts = self.depth_frames[-1]
            
            # Get the latest water mask, or None if not available
            water_mask = None
            if self.water_masks:
                water_mask, _ = self.water_masks[-1]
            
            # Get the latest detections, or empty list if not available
            detections = []
            if self.detections:
                detections, _ = self.detections[-1]
                
            # Get the latest obstacles, or empty list if not available
            obstacles = []
            if self.obstacles:
                obstacles, _ = self.obstacles[-1]
            
            return rgb_frame, depth_frame, water_mask, detections, obstacles

    def check_detection_timeout(self, timeout_seconds=2.0):
        """
        Check if detections have timed out.
        Returns True if no new detections have been received in the last timeout_seconds.
        """
        current_time = time.time()
        time_since_last_detection = current_time - self.last_detection_time
        
        # Log timeout status periodically
        if int(time_since_last_detection) % 5 == 0 and time_since_last_detection > 1.0:
            print(f"Time since last detection: {time_since_last_detection:.1f}s (timeout: {timeout_seconds}s)")
        
        return time_since_last_detection > timeout_seconds

    def check_object_timeout(self, timeout_seconds=2.0):
        """Check timeout specifically for object detections."""
        current_time = time.time()
        time_since_last = current_time - self.last_object_detection_time
        return time_since_last > timeout_seconds

    def check_obstacle_timeout(self, timeout_seconds=2.0):
        """
        Check if obstacles have timed out.
        Returns True if no new obstacles have been received in the last timeout_seconds.
        """
        current_time = time.time()
        time_since_last_obstacle = current_time - self.last_obstacle_detection_time
        
        # Log timeout status periodically
        if int(time_since_last_obstacle) % 5 == 0 and time_since_last_obstacle > 1.0:
            print(f"Time since last obstacle: {time_since_last_obstacle:.1f}s (timeout: {timeout_seconds}s)")
        
        return time_since_last_obstacle > timeout_seconds

    def clear_old_data(self):
        """Manually clear buffers to help with memory management"""
        with self.lock:
            # Keep only the latest frame in each buffer
            if len(self.rgb_frames) > 1:
                latest = self.rgb_frames[-1]
                self.rgb_frames.clear()
                self.rgb_frames.append(latest)
                
            if len(self.depth_frames) > 1:
                latest = self.depth_frames[-1]
                self.depth_frames.clear()
                self.depth_frames.append(latest)
                
            if len(self.water_masks) > 1:
                latest = self.water_masks[-1]
                self.water_masks.clear()
                self.water_masks.append(latest)
                
            if len(self.detections) > 1:
                latest = self.detections[-1]
                self.detections.clear()
                self.detections.append(latest)
                
            if len(self.obstacles) > 1:
                latest = self.obstacles[-1]
                self.obstacles.clear()
                self.obstacles.append(latest)
            
            # Force garbage collection
            self.processed_timestamps.clear()
            
    def clear_detections_and_obstacles(self):
        """
        Clear all detections and obstacles data (for timeout handling).
        Also resets the timestamp counters.
        """
        with self.lock:
            self.detections.clear()
            self.obstacles.clear()
            # Set timestamps to current time to avoid immediate timeout after clearing
            current_time = time.time()
            self.last_detection_time = current_time
            self.last_obstacle_time = current_time

class visionNode(Node):
    def __init__(self):
        super().__init__('vision')
        
        # Use individual declare_parameter calls
        self.declare_parameter('rgb_topic', '/cam')
        # Fixed to use raw depth topic directly
        self.declare_parameter('depth_topic', '/toast/cam/depth')
        self.declare_parameter('detection_topic', '/raw_detections')
        self.declare_parameter('water_mask_topic', '/water_mask')
        self.declare_parameter('obstacle_topic', '/obstacle_detection/raw_bboxes')
        self.declare_parameter('fov_degrees', 72.0)
        self.declare_parameter('enable_video', True)
        self.declare_parameter('enable_water_seg', True)
        self.declare_parameter('enable_mask', True)
        self.declare_parameter('max_age', 7)  # Reduced from 15 to 7 to clear tracks faster
        self.declare_parameter('min_hits', 3)
        self.declare_parameter('tracker_iou_threshold', 0.3)
        self.declare_parameter('use_tracker', True)
        self.declare_parameter('object_water_threshold', 0.5)  # 50% for objects
        self.declare_parameter('obstacle_water_threshold', 0.3)  # 30% for obstacles
        self.declare_parameter('robot_length', 1.0)  # Robot length in meters
        self.declare_parameter('robot_width', 0.8)  # Robot width in meters
        self.declare_parameter('camera_front_offset', 0.0)  # Camera is at the front
        # Add parameter for using YOLO for both objects and obstacles classification
        self.declare_parameter('use_yolo_classifier', True)
        # Add depth conversion parameters
        self.declare_parameter('min_depth_mm', 500)   # Min depth in mm
        self.declare_parameter('max_depth_mm', 12000) # Max depth in mm
        # Add water boundary parameters
        self.declare_parameter('water_boundary_roi_height', 0.3)
        self.declare_parameter('water_boundary_roi_width', 0.8)
        # Add robot mask parameter
        self.declare_parameter('enable_robot_mask', True)  # Enable robot mask by default
        self.declare_parameter('robot_mask_path', os.path.join(os.path.dirname(__file__), '..', 'masks', 'sim.png'))        # Add timeout parameters for detections and obstacles
        self.declare_parameter('detection_timeout', 2.0)  # Increased to 2.0 seconds
        self.declare_parameter('obstacle_timeout', 2.0)   # Increased to 2.0 seconds
        # Add water mask parameters
        self.declare_parameter('min_water_mask_area_ratio', 0.1)  # Minimum water area as fraction of image
        self.declare_parameter('max_water_contours', 2)  # Maximum number of water contours before considering invalid
        self.declare_parameter('front_depth_threshold', 1.8)  # Increased from 1.2m to 1.8m
        
        # Add parameters for simulation injection
        self.declare_parameter('sim_injection_image_path', '/home/hamze/Desktop/SeaClean/toast_ws/src/vision/trick/bg.png')  # Default path to injection image
        self.declare_parameter('sim_injection_alpha', 0.7)  # Blend alpha value (0-1)
        
        # Add servo control parameters and variables
        self.declare_parameter('collection_speed_factor', 0.7)  # Speed factor during collection
        self.collection_speed_factor = self.get_parameter('collection_speed_factor').get_parameter_value().double_value
        self.servo_state = 0  # 0: idle, 1: collecting, 2: retracting
        self.collection_start_time = None
        self.collection_duration = 4.0  # Duration between servo states in seconds
        
        # Add servo publisher
        self.servo_publisher = self.create_publisher(String, '/servo', 10)
        
        # Get parameters
        self.rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        self.depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        self.detection_topic = self.get_parameter('detection_topic').get_parameter_value().string_value
        self.water_mask_topic = self.get_parameter('water_mask_topic').get_parameter_value().string_value
        self.obstacle_topic = self.get_parameter('obstacle_topic').get_parameter_value().string_value
        self.fov_degrees = self.get_parameter('fov_degrees').get_parameter_value().double_value
        self.enable_video = self.get_parameter('enable_video').get_parameter_value().bool_value
        self.enable_water_seg = self.get_parameter('enable_water_seg').get_parameter_value().bool_value
        self.enable_mask = self.get_parameter('enable_mask').get_parameter_value().bool_value
        self.max_age = self.get_parameter('max_age').get_parameter_value().integer_value
        self.min_hits = self.get_parameter('min_hits').get_parameter_value().integer_value
        self.tracker_iou_threshold = self.get_parameter('tracker_iou_threshold').get_parameter_value().double_value
        self.use_tracker = self.get_parameter('use_tracker').get_parameter_value().bool_value
        self.object_water_threshold = self.get_parameter('object_water_threshold').get_parameter_value().double_value
        self.obstacle_water_threshold = self.get_parameter('obstacle_water_threshold').get_parameter_value().double_value
        self.robot_length = self.get_parameter('robot_length').get_parameter_value().double_value
        self.robot_width = self.get_parameter('robot_width').get_parameter_value().double_value
        self.camera_front_offset = self.get_parameter('camera_front_offset').get_parameter_value().double_value
        self.use_yolo_classifier = self.get_parameter('use_yolo_classifier').get_parameter_value().bool_value
        self.min_depth_mm = self.get_parameter('min_depth_mm').get_parameter_value().integer_value
        self.max_depth_mm = self.get_parameter('max_depth_mm').get_parameter_value().integer_value
        self.water_boundary_roi_height = self.get_parameter('water_boundary_roi_height').get_parameter_value().double_value
        self.water_boundary_roi_width = self.get_parameter('water_boundary_roi_width').get_parameter_value().double_value
        self.enable_robot_mask = self.get_parameter('enable_robot_mask').get_parameter_value().bool_value
        self.robot_mask_path = self.get_parameter('robot_mask_path').get_parameter_value().string_value
        self.detection_timeout = self.get_parameter('detection_timeout').get_parameter_value().double_value
        self.obstacle_timeout = self.get_parameter('obstacle_timeout').get_parameter_value().double_value
        self.min_water_mask_area_ratio = self.get_parameter('min_water_mask_area_ratio').get_parameter_value().double_value
        self.max_water_contours = self.get_parameter('max_water_contours').get_parameter_value().integer_value
        self.front_depth_threshold = self.get_parameter('front_depth_threshold').get_parameter_value().double_value
        self.sim_injection_image_path = self.get_parameter('sim_injection_image_path').get_parameter_value().string_value
        self.sim_injection_alpha = self.get_parameter('sim_injection_alpha').get_parameter_value().double_value
        
        self.bridge = CvBridge()
        self.frame_buffer = FrameBuffer()
        
        # Initialize SORT tracker if enabled
        if self.use_tracker:
            self.object_tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.tracker_iou_threshold)
            self.obstacle_tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, iou_threshold=self.tracker_iou_threshold)
            self.last_object_tracker_reset = 0
            self.last_obstacle_tracker_reset = 0
            self.get_logger().info("SORT trackers initialized separately for objects and obstacles")
        else:
            self.object_tracker = None
            self.obstacle_tracker = None
            self.get_logger().info("SORT trackers disabled")
            
        # Initialize TF broadcaster
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        publish_static_transform(self.tf_broadcaster, self)
        publish_dynamic_transform(self.tf_broadcaster, self)
        
        # Robot mask handling
        self.robot_mask = None
        self.robot_mask_original = None
        self.robot_mask_resized = False

        if self.enable_robot_mask:
            try:
                # Use the exact path provided
                robot_mask_path = self.robot_mask_path
                
                # Check if the file exists
                if os.path.exists(robot_mask_path):
                    self.get_logger().info(f"Loading robot mask from: {robot_mask_path}")
                    self.robot_mask = cv2.imread(robot_mask_path, cv2.IMREAD_GRAYSCALE)
                    
                    if self.robot_mask is not None:
                        # Threshold to binary
                        _, self.robot_mask = cv2.threshold(self.robot_mask, 128, 255, cv2.THRESH_BINARY)
                        self.robot_mask_original = self.robot_mask.copy()
                        self.get_logger().info(f"Robot mask loaded successfully, shape: {self.robot_mask.shape}")
                    else:
                        self.get_logger().error(f"Failed to load robot mask from {robot_mask_path} - imread returned None")
                else:
                    self.get_logger().error(f"Robot mask file not found at: {robot_mask_path}")
            except Exception as e:
                self.get_logger().error(f"Error loading robot mask: {e}")
                self.get_logger().error(traceback.format_exc())
        
        # Initialize simulation injection variables
        self.sim_injection_image = None
        self.sim_injection_resized = False
        
        # Load simulation injection image if path exists
        self.load_sim_injection_image()
        
        # We're no longer tracking water mask status in vision (moved to control system)
        
        # Setup publishers
        self.setup_publishers()

        # Subscribe to topics
        self.rgb_sub = self.create_subscription(Image, self.rgb_topic, self.rgb_callback, 10)
        self.depth_sub = self.create_subscription(Image, self.depth_topic, self.depth_callback, 10)
        self.detection_sub = self.create_subscription(PoseArray, self.detection_topic, self.detection_callback, 10)
        self.water_mask_sub = self.create_subscription(Image, self.water_mask_topic, self.water_mask_callback, 10)
        
        # Only subscribe to obstacle topic if not using YOLO classifier
        if not self.use_yolo_classifier:
            self.obstacle_sub = self.create_subscription(String, self.obstacle_topic, self.obstacle_callback, 10)
            self.get_logger().info("Using separate obstacle detector")
        else:
            self.obstacle_sub = None
            self.get_logger().info("Using YOLO for both objects and obstacles classification")
        
        # ADD: Subscribe to robot pose
        self.robot_pose = None
        self.robot_pose_sub = self.create_subscription(
            PoseStamped, 
            'robot_pose', 
            self.robot_pose_callback, 
            10
        )

        # Processing timer
        self.create_timer(0.05, self.process_data)  # 20Hz processing
        
        # Current water data for visualization
        self.current_water_mask = None
        self.current_water_overlay = None
        self.declare_parameter('memory_cleanup_interval', 5.0)  # seconds
        self.memory_cleanup_interval = self.get_parameter('memory_cleanup_interval').get_parameter_value().double_value
        self.last_memory_cleanup = time.time()
        
        # Water boundary detection variables
        self.last_water_boundary_depth = 5.0  # Default to 5m distance
        
        # Add flags for tracking valid frames
        self.last_valid_frame_time = time.time()
        self.frame_validity_timeout = 0.5  # 500ms timeout for frame validity
        
        # Add published object and obstacle tracking
        self.last_published_objects = []
        self.last_published_obstacles = []
        
        # Frame counter for periodic logging
        self.frame_counter = 0
        
        self.get_logger().info(f"vision fusion initialized - Using raw depth from {self.depth_topic}")
        self.get_logger().info(f"Depth range: {self.min_depth_mm}mm to {self.max_depth_mm}mm")
        if self.enable_robot_mask:
            self.get_logger().info("Robot mask enabled - objects in robot mask area will be ignored")
        self.get_logger().info(f"Simulation injection enabled - using image from {self.sim_injection_image_path}")

    # ADD: Robot pose callback
    def robot_pose_callback(self, msg):
        self.robot_pose = msg
        self.get_logger().debug(f"Received robot pose: x={msg.pose.position.x:.2f}, y={msg.pose.position.y:.2f}")

    def setup_publishers(self):
        self.object_poses_pub = self.create_publisher(PoseArray, 'object_poses', 10)
        self.obstacle_poses_pub = self.create_publisher(PoseArray, 'obstacle_poses', 10)
        self.obstacle_sizes_pub = self.create_publisher(Float32MultiArray, 'obstacle_sizes', 10)  # New publisher for obstacle sizes
        self.object_pose_stamped_pub = self.create_publisher(PoseStamped, 'object_pose_stamped', 10)
        self.markers_pub = self.create_publisher(MarkerArray, 'object_markers', 10)
        self.fused_image_pub = self.create_publisher(Image, 'fused_image', 10)
        self.camera_marker_publisher = self.create_publisher(Marker, 'camera_marker', 10)
        self.point_cloud_pub = self.create_publisher(PointCloud, 'points_3d', 10)
        self.front_depth_pub = self.create_publisher(Range, 'front_depth', 10)
        # Removed water_mask_status_publisher (now handled by control system)
        
        # Add simulation injection publisher
        self.sim_injection_publisher = self.create_publisher(Image, 'sim_injection', 10)
        
        # Add injection mask publisher for debugging
        self.injection_mask_publisher = self.create_publisher(Image, 'injection_mask', 10)
        
        # Initialize camera marker
        publish_camera_marker(self.camera_marker_publisher, self)
        
        # Additional publishers based on flags
        if self.enable_video:
            self.frame_publisher = self.create_publisher(Image, 'video_frame', 10)
        else:
            self.frame_publisher = None
            
        # Removed water_seg_publisher - don't publish water segmentation
            
        if self.enable_mask:
            self.mask_publisher = self.create_publisher(Image, 'mask', 10)
            
            # Add robot mask publisher if enabled
            if self.enable_robot_mask:
                self.robot_mask_publisher = self.create_publisher(Image, 'robot_mask', 10)
            else:
                self.robot_mask_publisher = None
        else:
            self.mask_publisher = None
            self.robot_mask_publisher = None

    def rgb_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.frame_buffer.add_rgb(frame, msg.header.stamp)
            self.last_valid_frame_time = time.time()  # Update valid frame time
        except Exception as e:
            self.get_logger().error(f"Error in RGB callback: {e}")

    def depth_callback(self, msg):
        try:
            # Only process raw depth formats, not colorized ones
            # Handle different incoming depth formats and store raw depth
            self.get_logger().debug(f"Received depth image with encoding: {msg.encoding}")
            
            if msg.encoding == "16UC1":
                # This is the format from oak_cam.py - already in mm
                depth_raw = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                self.get_logger().debug(f"Processing 16UC1 depth image, shape: {depth_raw.shape}")
            elif msg.encoding == "32FC1":
                # Convert meters to mm for consistency
                depth_float = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                depth_raw = (depth_float * 1000).astype(np.uint16)  # Convert m to mm
                self.get_logger().debug(f"Processing 32FC1 depth image, shape: {depth_raw.shape}")
            elif msg.encoding == "bgr8":
                # Ignore colorized depth images - we should be subscribed to raw depth
                self.get_logger().warn("Received colorized depth (bgr8) - ignoring! Please check topic subscription.")
                return
            else:
                self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
                return
                
            # Clip depth values to the valid range
            depth_clipped = np.clip(depth_raw, self.min_depth_mm, self.max_depth_mm)
            
            # Add to buffer
            self.frame_buffer.add_depth(depth_clipped, msg.header.stamp)
                
        except Exception as e:
            self.get_logger().error(f"Error in depth callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def detection_callback(self, msg):
        """
        SIMPLIFIED detection callback that properly handles empty detections.
        Always publishes arrays (empty or with data) so behavior system knows the current state.
        """
        try:
            detections = []
            obstacles = []

            # Parse message
            for pose in msg.poses:
                x1 = pose.position.x
                y1 = pose.position.y
                w = pose.position.z
                h = pose.orientation.x
                score = pose.orientation.y
                class_id = int(pose.orientation.z)

                # Skip degenerate boxes
                if w <= 0 or h <= 0:
                    continue

                det_dict = {
                    'bbox': (x1, y1, x1 + w, y1 + h),
                    'width': w,
                    'height': h,
                    'score': score,
                    'class_id': class_id
                }

                # Fixed class mapping
                if self.use_yolo_classifier:
                    if class_id == 1:  # Goal/object
                        det_dict['type'] = 'object'
                        detections.append(det_dict)
                    elif class_id == 0:  # Obstacle
                        det_dict.update({
                            'type': 'obstacle',
                            'vio_depth': 0.0,
                            'vio_features': 0,
                            'vio_lateral': 0.0,
                            'vio_angle_deg': 0.0
                        })
                        obstacles.append(det_dict)
                else:
                    # Legacy: treat everything as object
                    det_dict['type'] = 'object'
                    detections.append(det_dict)

            # ALWAYS update the buffer - this clears old data
            self.frame_buffer.add_detections(detections, msg.header.stamp)
            self.frame_buffer.add_obstacles(obstacles, msg.header.stamp)

            # IMMEDIATE publishing for empty detections
            # This ensures behavior system gets updates immediately
            current_time = self.get_clock().now().to_msg()
            
            # Publish objects
            if not detections:
                # NO OBJECTS: Publish empty array immediately
                empty_pose_array = PoseArray()
                empty_pose_array.header.stamp = current_time
                empty_pose_array.header.frame_id = "map"
                self.object_poses_pub.publish(empty_pose_array)
                
                # Also publish empty stamped pose
                empty_pose_stamped = PoseStamped()
                empty_pose_stamped.header.stamp = current_time
                empty_pose_stamped.header.frame_id = "map"
                self.object_pose_stamped_pub.publish(empty_pose_stamped)
                
                # Clear tracking
                self.last_published_objects = []
                
                self.get_logger().debug("Published empty object arrays - no objects detected")
            
            # Publish obstacles  
            if not obstacles:
                # NO OBSTACLES: Publish empty array immediately
                empty_pose_array = PoseArray()
                empty_pose_array.header.stamp = current_time
                empty_pose_array.header.frame_id = "map"
                self.obstacle_poses_pub.publish(empty_pose_array)
                
                # Also publish empty sizes
                empty_sizes = Float32MultiArray()
                self.obstacle_sizes_pub.publish(empty_sizes)
                
                # Clear tracking
                self.last_published_obstacles = []
                
                self.get_logger().debug("Published empty obstacle arrays - no obstacles detected")

            # Handle tracker updates if enabled
            if self.use_tracker:
                if detections:
                    obj_boxes = [[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['score']]
                                for d in detections]
                    _ = self.object_tracker.update(np.array(obj_boxes))
                else:
                    _ = self.object_tracker.update(np.empty((0, 5)))

                if obstacles:
                    obs_boxes = [[o['bbox'][0], o['bbox'][1], o['bbox'][2], o['bbox'][3], o['score']]
                                for o in obstacles]
                    _ = self.obstacle_tracker.update(np.array(obs_boxes))
                else:
                    _ = self.obstacle_tracker.update(np.empty((0, 5)))

        except Exception as e:
            self.get_logger().error(f"Error in detection_callback: {e}", throttle_duration_sec=2.0)



    def water_mask_callback(self, msg):
        try:
            water_mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
            # Normalize and threshold if needed
            water_mask = (water_mask > 128).astype(np.uint8)
            self.frame_buffer.add_water_mask(water_mask, msg.header.stamp)
            self.current_water_mask = water_mask
            
            # Get the latest RGB frame for injection - this provides the clean, unmodified frame
            rgb_frame, _, _, _, _ = self.frame_buffer.get_latest_data()
            
            # Process simulation injection if we have a frame
            # The frame from get_latest_data() should now be clean since we're no longer modifying
            # the original rgb_frame in process_data()
            if rgb_frame is not None:
                # Make a complete copy to ensure original is untouched
                clean_frame = rgb_frame.copy()
                self.process_sim_injection(clean_frame, water_mask, msg.header.stamp)
                
        except Exception as e:
            self.get_logger().error(f"Error in water mask callback: {e}")

    def load_sim_injection_image(self):
        try:
            if os.path.exists(self.sim_injection_image_path):
                self.get_logger().info(f"Loading sim injection image from: {self.sim_injection_image_path}")
                self.sim_injection_image = cv2.imread(self.sim_injection_image_path)
                
                if self.sim_injection_image is not None:
                    self.get_logger().info(f"Sim injection image loaded successfully, shape: {self.sim_injection_image.shape}")
                else:
                    self.get_logger().error(f"Failed to load sim injection image - imread returned None")
            else:
                self.get_logger().error(f"Sim injection image not found at: {self.sim_injection_image_path}")
        except Exception as e:
            self.get_logger().error(f"Error loading sim injection image: {e}")
            self.get_logger().error(traceback.format_exc())

    def process_sim_injection(self, rgb_frame, water_mask, timestamp):
        """Process simulation injection with complete replacement (no blending).
        This creates a view where non-water areas are completely replaced with the injection image."""
        
        # If no injection image loaded yet, try to load it
        if self.sim_injection_image is None:
            self.load_sim_injection_image()
            
        # If still no injection image, return early
        if self.sim_injection_image is None:
            return
            
        try:
            # Use the clean frame - now guaranteed to have no contours/text
            clean_frame = rgb_frame.copy()
            
            # Resize injection image if needed
            h, w = clean_frame.shape[:2]
            if not self.sim_injection_resized or self.sim_injection_image.shape[:2] != (h, w):
                self.sim_injection_image = cv2.resize(self.sim_injection_image, (w, h))
                self.sim_injection_resized = True
                self.get_logger().info(f"Resized injection image to {w}x{h}")
                
            # Create a mask for areas to inject (not water)
            # Ensure water mask is binary and handle None case
            if water_mask is None:
                # If no water mask, use the entire frame except robot
                non_water_mask = np.ones((h, w), dtype=np.uint8) * 255
                self.get_logger().warn("No water mask available - injecting everywhere")
            else:
                # 255 where not water (0 in water_mask)
                non_water_mask = (water_mask == 0).astype(np.uint8) * 255

            # If robot mask is available, exclude robot areas from injection
            if self.enable_robot_mask and self.robot_mask is not None:
                # Resize robot mask if needed
                if self.robot_mask.shape[:2] != (h, w):
                    self.resize_robot_mask_if_needed(clean_frame.shape)
                    
                # Ensure robot mask is binary
                if self.robot_mask.dtype != np.uint8:
                    self.robot_mask = (self.robot_mask > 0).astype(np.uint8) * 255
                    
                # Exclude robot areas from the injection mask
                non_robot_mask = cv2.bitwise_not(self.robot_mask)
                injection_mask = cv2.bitwise_and(non_water_mask, non_robot_mask)
            else:
                injection_mask = non_water_mask
            
            # Create mask for regions to keep original (water or robot)
            keep_original_mask = cv2.bitwise_not(injection_mask)
            
            # Apply masks to combine the images
            # NO BLENDING - COMPLETE REPLACEMENT
            # Where injection_mask is 255 (white), use sim_injection_image
            # Where keep_original_mask is 255 (white), use original clean_frame
            foreground = cv2.bitwise_and(self.sim_injection_image, self.sim_injection_image, mask=injection_mask)
            background = cv2.bitwise_and(clean_frame, clean_frame, mask=keep_original_mask)
            
            # Combine the two - result is clean with no contours, boxes or text
            injection_result = cv2.add(foreground, background)
            
            # Add debug info periodically to avoid log spam
            if self.frame_counter % 50 == 0:
                if water_mask is not None:
                    water_coverage = np.sum(water_mask)/(water_mask.shape[0]*water_mask.shape[1])
                    self.get_logger().info(f"Injection stats: complete replacement mode, " 
                                         f"water coverage={water_coverage:.2f}, "
                                         f"injection area={(np.sum(injection_mask)/255)/(h*w):.2f}")
            
            # Publish injection mask for debugging
            try:
                # Create a colorized version of the injection mask for better visualization
                mask_vis = np.zeros((h, w, 3), dtype=np.uint8)
                # Red channel: injection areas
                mask_vis[:,:,2] = injection_mask
                # Green channel: water mask
                if water_mask is not None:
                    mask_vis[:,:,1] = water_mask * 255
                # Blue channel: robot mask
                if self.enable_robot_mask and self.robot_mask is not None:
                    mask_vis[:,:,0] = self.robot_mask
                    
                # Publish the colorized mask
                mask_msg = self.bridge.cv2_to_imgmsg(mask_vis, "bgr8")
                mask_msg.header.stamp = timestamp
                mask_msg.header.frame_id = "camera"
                self.injection_mask_publisher.publish(mask_msg)
            except Exception as e:
                self.get_logger().error(f"Error publishing injection mask: {e}")
            
            # Publish the injection result
            try:
                injection_msg = self.bridge.cv2_to_imgmsg(injection_result, "bgr8")
                injection_msg.header.stamp = timestamp
                injection_msg.header.frame_id = "camera"
                self.sim_injection_publisher.publish(injection_msg)
                self.get_logger().debug("Published clean simulation injection image")
            except Exception as e:
                self.get_logger().error(f"Error publishing injection image: {e}")
                
        except Exception as e:
            self.get_logger().error(f"Error processing sim injection: {e}")
            self.get_logger().error(traceback.format_exc())

    def obstacle_callback(self, msg):
        try:
            # Parse JSON data from String message
            obstacle_data = json.loads(msg.data)
            
            # Convert to the same format as detections
            obstacles = []
            for i, data in enumerate(obstacle_data):
                # Check if we have the new format with depth info or old format with just bbox
                if isinstance(data, dict):
                    x1, y1, x2, y2 = data['bbox']
                    depth = data.get('depth', 0.0)
                    num_features = data.get('num_features', 0)
                    lateral = data.get('lateral', 0.0)
                    angle_deg = data.get('angle_deg', 0.0)
                else:
                    # Legacy format with just bbox
                    x1, y1, x2, y2 = data
                    depth = 0.0
                    num_features = 0
                    lateral = 0.0
                    angle_deg = 0.0
                
                # Skip obstacles with invalid bounding boxes
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                width = x2 - x1
                height = y2 - y1
                
                obstacles.append({
                    'bbox': (x1, y1, x2, y2),
                    'width': width,  # Store bbox width
                    'height': height,  # Store bbox height
                    'score': 1.0,  # Default score for obstacles
                    'class_id': 0,  # Default class ID for obstacles
                    'type': 'obstacle',  # Mark as obstacle
                    'vio_depth': depth,  # Store depth calculated from VIO features
                    'vio_features': num_features,  # Store number of VIO features
                    'vio_lateral': lateral,  # Store lateral offset calculated from features
                    'vio_angle_deg': angle_deg  # Store angle calculated from features
                })
            
            # String messages don't have headers, so use current clock time
            current_time = self.get_clock().now().to_msg()
            
            if obstacles:
                self.frame_buffer.add_obstacles(obstacles, current_time)
                self.frame_buffer.last_obstacle_detection_time = time.time()  # Update timestamp
            else:
                # If no obstacles are detected in the current frame but we were tracking some before,
                # publish empty obstacle messages to clear previous obstacles
                if not self.frame_buffer.check_obstacle_timeout(self.obstacle_timeout) and self.last_published_obstacles:
                    self.get_logger().info("No obstacles detected in current frame - publishing empty obstacles")
                    
                    # Publish empty pose array for obstacles
                    empty_pose_array = PoseArray()
                    empty_pose_array.header.stamp = self.get_clock().now().to_msg()
                    empty_pose_array.header.frame_id = "map"
                    self.obstacle_poses_pub.publish(empty_pose_array)
                    
                    # Publish empty size array for obstacles
                    empty_sizes = Float32MultiArray()
                    self.obstacle_sizes_pub.publish(empty_sizes)
                    
                    # Clear last published obstacles
                    self.last_published_obstacles = []
                    
        except Exception as e:
            self.get_logger().error(f"Error in obstacle callback: {e}")
            self.get_logger().error(traceback.format_exc())

    def create_default_robot_mask(self, shape):
        """Create an empty robot mask if none exists."""
        h, w = shape[:2]
        self.robot_mask = np.zeros((h, w), dtype=np.uint8)
        self.robot_mask_original = self.robot_mask.copy()
        self.robot_mask_resized = True
        self.get_logger().warn("Created empty robot mask as fallback")

    def resize_robot_mask_if_needed(self, frame_shape):
        """Resize robot mask to match frame dimensions if needed."""
        if self.robot_mask is None:
            return
            
        h, w = frame_shape[:2]
        mask_h, mask_w = self.robot_mask.shape[:2]
        
        if h != mask_h or w != mask_w:
            if not self.robot_mask_resized or self.robot_mask.shape != frame_shape[:2]:
                self.get_logger().info(f"Resizing robot mask from {mask_w}x{mask_h} to {w}x{h}")
                self.robot_mask = cv2.resize(self.robot_mask_original, (w, h), 
                                            interpolation=cv2.INTER_NEAREST)
                self.robot_mask_resized = True

    def check_robot_mask(self, bbox):
        """Check if a bounding box overlaps with the robot mask."""
        if self.robot_mask is None or not self.enable_robot_mask:
            return False
            
        x1, y1, x2, y2 = map(int, bbox)
        h, w = self.robot_mask.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        roi = self.robot_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return False
            
        # Calculate overlap ratio
        robot_ratio = np.sum(roi) / roi.size
        
        # If object is significantly in robot mask and we're not already collecting
        if robot_ratio > 0.6 and self.servo_state == 0:  # Increased threshold to 60%
            self.start_collection_sequence()
            return True
        
        return robot_ratio > 0.3  # Keep original threshold for general mask checking

    def start_collection_sequence(self):
        """Start the object collection sequence."""
        if self.servo_state == 0:  # Only start if we're idle
            self.get_logger().info("Starting collection sequence")
            self.servo_state = 1
            self.collection_start_time = self.get_clock().now()
            
            # Store current position for movement tracking - FIX: Check if robot_pose exists and is not None
            if hasattr(self, 'robot_pose') and self.robot_pose is not None:
                self.collection_start_pose = self.robot_pose
            else:
                # Initialize to None to avoid errors later
                self.collection_start_pose = None
                self.get_logger().warn("No robot pose available when starting collection")
            
            # Publish servo command 1
            msg = String()
            msg.data = "1"
            self.servo_publisher.publish(msg)
            
            self.get_logger().info("Published servo command 1")

    def update_collection_sequence(self):
        """Update the collection sequence state."""
        if self.servo_state == 1 and self.collection_start_time is not None:
            current_time = self.get_clock().now()
            elapsed_time = (current_time - self.collection_start_time).nanoseconds / 1e9
            
            # Check if robot is moving enough during collection
            if hasattr(self, 'robot_pose') and hasattr(self, 'collection_start_pose'):
                self.check_robot_motion_during_collection()
            
            # Log collection progress periodically
            if int(elapsed_time) != int(elapsed_time - 0.1):  # Log every second approximately
                self.get_logger().info(f"Collection in progress: {elapsed_time:.1f}/{self.collection_duration:.1f}s")
                
            if elapsed_time >= self.collection_duration:
                # Time to retract
                self.servo_state = 2
                msg = String()
                msg.data = "2"
                self.servo_publisher.publish(msg)
                self.get_logger().info("Published servo command 2")
                
                # Reset state after publishing
                self.servo_state = 0
                self.collection_start_time = None



    def check_robot_motion_during_collection(self):
        """
        Check if the robot is moving during collection.
        Returns True if motion is sufficient, False if we need to send additional motion commands.
        """
        if self.servo_state != 1 or self.collection_start_time is None:
            return True  # Not in collection mode, no checking needed
        
        # Calculate elapsed time in collection
        current_time = self.get_clock().now()
        elapsed_time = (current_time - self.collection_start_time).nanoseconds / 1e9
        
        # Check if we have valid pose data
        if self.collection_start_pose is None or self.robot_pose is None:
            self.get_logger().warn(f"Missing pose data for motion check: start_pose={self.collection_start_pose is not None}, current_pose={self.robot_pose is not None}")
            return True  # Skip check if we don't have valid pose data
        
        # If more than half of collection time has passed and robot isn't moving
        # enough, we might need to take additional action
        if elapsed_time > self.collection_duration / 2:
            # Check if robot has moved since collection started
            start_x = self.collection_start_pose.pose.position.x
            start_y = self.collection_start_pose.pose.position.y
            current_x = self.robot_pose.pose.position.x
            current_y = self.robot_pose.pose.position.y
            
            # Calculate distance moved
            distance_moved = math.sqrt((current_x - start_x)**2 + (current_y - start_y)**2)
            
            # If movement is insufficient, log a warning
            if distance_moved < 0.1:  # Less than 10cm movement
                self.get_logger().warn(f"Insufficient robot motion during collection: {distance_moved:.2f}m in {elapsed_time:.1f}s")
                return False
        
        return True


    def check_bbox_in_frame(self, bbox, frame_shape):
        """Check if bbox is still within the frame boundaries"""
        h, w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        
        # Check if bbox is completely outside the frame
        if x2 < 0 or x1 >= w or y2 < 0 or y1 >= h:
            return False
            
        # Check if bbox is mostly within the frame
        bbox_area = (x2 - x1) * (y2 - y1)
        if bbox_area <= 0:
            return False
            
        # Calculate visible area
        vis_x1 = max(0, x1)
        vis_y1 = max(0, y1)
        vis_x2 = min(w, x2)
        vis_y2 = min(h, y2)
        
        visible_area = (vis_x2 - vis_x1) * (vis_y2 - vis_y1)
        visible_ratio = visible_area / bbox_area
        
        # Consider in frame if at least 25% is visible
        return visible_ratio >= 0.25

    # Removed water mask status publishing - this will now be handled by the control system

    # Add these timestamp update methods to the FrameBuffer class
    def add_detections(self, detections, timestamp):
        with self.lock:
            # Split detections by type and update respective timestamps
            object_detections = []
            obstacle_detections = []
            
            for det in detections:
                if det.get('type', 'object') == 'object':
                    object_detections.append(det)
                else:
                    obstacle_detections.append(det)
            
            if object_detections:
                self.last_object_detection_time = time.time()
            if obstacle_detections:
                self.last_obstacle_detection_time = time.time()
                
            # Store all detections together
            filtered_detections = [d for d in detections if d['score'] > 0.3]
            self.detections.append((filtered_detections, timestamp))
            # Also update the general detection timestamp
            self.last_detection_time = time.time()
            self._check_cleanup()
            
    def add_obstacles(self, obstacles, timestamp):
        with self.lock:
            self.obstacles.append((obstacles, timestamp))
            self.last_obstacle_time = time.time()  # Update timestamp
            # Also update the specific obstacle detection timestamp
            self.last_obstacle_detection_time = time.time()
            self._check_cleanup()

    # Improved check_and_handle_tracker_reset method for visionNode class
    def check_and_handle_tracker_reset(self):
        """Check if SORT trackers need to be reset based on separate detection timeouts."""
        if not self.use_tracker:
            return False
            
        current_time = time.time()
        min_reset_interval = self.detection_timeout * 2  # Minimum time between resets
        reset_occurred = False
        
        # Check object tracker
        if (self.frame_buffer.check_object_timeout(self.detection_timeout) and 
            current_time - self.last_object_tracker_reset > min_reset_interval):
            self.get_logger().info("Object detections timed out - resetting object tracker")
            self.object_tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, 
                                    iou_threshold=self.tracker_iou_threshold)
            self.last_object_tracker_reset = current_time
            
            # Clear only object publications and publish empty object messages
            if self.last_published_objects:
                self.get_logger().info("Publishing empty object messages due to timeout")
                empty_pose_array = PoseArray()
                empty_pose_array.header.stamp = self.get_clock().now().to_msg()
                empty_pose_array.header.frame_id = "map"
                self.object_poses_pub.publish(empty_pose_array)
                
                empty_pose_stamped = PoseStamped()
                empty_pose_stamped.header.stamp = self.get_clock().now().to_msg()
                empty_pose_stamped.header.frame_id = "map"
                self.object_pose_stamped_pub.publish(empty_pose_stamped)
                
                self.last_published_objects = []  # Clear only object publications
                reset_occurred = True
            
        # Check obstacle tracker
        if (self.frame_buffer.check_obstacle_timeout(self.obstacle_timeout) and 
            current_time - self.last_obstacle_tracker_reset > min_reset_interval):
            self.get_logger().info("Obstacle detections timed out - resetting obstacle tracker")
            self.obstacle_tracker = Sort(max_age=self.max_age, min_hits=self.min_hits, 
                                    iou_threshold=self.tracker_iou_threshold)
            self.last_obstacle_tracker_reset = current_time
            
            # Clear only obstacle publications and publish empty obstacle messages
            if self.last_published_obstacles:
                self.get_logger().info("Publishing empty obstacle messages due to timeout")
                empty_pose_array = PoseArray()
                empty_pose_array.header.stamp = self.get_clock().now().to_msg()
                empty_pose_array.header.frame_id = "map"
                self.obstacle_poses_pub.publish(empty_pose_array)
                
                empty_sizes = Float32MultiArray()
                self.obstacle_sizes_pub.publish(empty_sizes)
                
                self.last_published_obstacles = []  # Clear only obstacle publications
                reset_occurred = True
            
        # Only clear markers if both trackers were reset or if both lists are empty
        if reset_occurred and (not self.last_published_objects and not self.last_published_obstacles):
            # Publish empty marker array to clear existing markers
            empty_marker_array = MarkerArray()
            delete_marker = Marker()
            delete_marker.action = Marker.DELETEALL
            delete_marker.header.stamp = self.get_clock().now().to_msg()
            delete_marker.header.frame_id = "map"
            empty_marker_array.markers.append(delete_marker)
            self.markers_pub.publish(empty_marker_array)
            
        return reset_occurred

    # Modified publish_empty_messages method
    def publish_empty_messages(self):
        """Publish empty messages when all objects/obstacles leave view"""
        # Publish empty marker array to clear existing markers
        empty_marker_array = MarkerArray()
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.header.stamp = self.get_clock().now().to_msg()
        delete_marker.header.frame_id = "map"
        empty_marker_array.markers.append(delete_marker)
        self.markers_pub.publish(empty_marker_array)
        
        # Publish empty pose arrays
        empty_pose_array = PoseArray()
        empty_pose_array.header.stamp = self.get_clock().now().to_msg()
        empty_pose_array.header.frame_id = "map"
        self.object_poses_pub.publish(empty_pose_array)
        self.obstacle_poses_pub.publish(empty_pose_array)
        
        # Publish empty size array
        empty_sizes = Float32MultiArray()
        self.obstacle_sizes_pub.publish(empty_sizes)
        
        # Publish empty pose stamped
        empty_pose_stamped = PoseStamped()
        empty_pose_stamped.header.stamp = self.get_clock().now().to_msg()
        empty_pose_stamped.header.frame_id = "map"
        self.object_pose_stamped_pub.publish(empty_pose_stamped)
        
        # Clear last published data
        self.last_published_objects = []
        self.last_published_obstacles = []
        
        self.get_logger().info("Published empty messages - all objects/obstacles cleared")

    def process_water_boundary(self, water_mask, depth_frame, vis_frame=None):
        """Calculate depth at the topmost water boundary (horizon) using the highest y-coordinate and publish as Range message."""
        # Create the Range message
        range_msg = Range()
        range_msg.header.stamp = self.get_clock().now().to_msg()
        range_msg.header.frame_id = "camera"
        range_msg.radiation_type = Range.INFRARED
        range_msg.field_of_view = math.radians(self.fov_degrees)
        range_msg.min_range = 0.5
        range_msg.max_range = 8.0
        
        # Default to 0 (no valid mask) - will be overridden if we find a valid horizon
        range_msg.range = 0.0
        
        # Check for missing water mask or depth frame
        if water_mask is None or depth_frame is None:
            self.get_logger().debug("No water mask or depth frame - publishing zero front depth")
            self.front_depth_pub.publish(range_msg)
            return
        
        try:
            # Get image dimensions
            h, w = water_mask.shape[:2]
            
            # Find all contours in the water mask
            contours, _ = cv2.findContours(water_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if we have no contours
            if not contours:
                self.get_logger().debug("No water contours found - publishing zero front depth")
                self.front_depth_pub.publish(range_msg)
                return
            
            # Filter contours to find the one with the highest y-coordinate (lowest y-value)
            min_significant_area = h * w * self.min_water_mask_area_ratio  # Minimum area threshold
            horizon_contour = None
            min_y = h  # Initialize with max height (bottom of image)
            contour_details = []  # For debugging
            
            for i, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if area < min_significant_area:
                    contour_details.append(f"Contour {i}: top_y=skipped, area={area:.0f} (too small)")
                    continue  # Skip small contours
                
                # Get the topmost y-coordinate of the contour
                contour_points = contour[:, 0, :]  # Shape: (N, 2) with (x, y)
                top_y = np.min(contour_points[:, 1])  # Minimum y-coordinate (highest point in image)
                
                # Log contour details for debugging
                contour_details.append(f"Contour {i}: top_y={top_y}, area={area:.0f}")
                
                # Select the contour with the highest top point (lowest y-value)
                if top_y < min_y:
                    min_y = top_y
                    horizon_contour = contour
            
            # Log all contour details
            self.get_logger().debug(f"Found {len(contours)} contours: {'; '.join(contour_details)}")
            
            # If no valid horizon contour is found
            if horizon_contour is None:
                self.get_logger().debug("No valid horizon contour found (all too small) - publishing zero front depth")
                self.front_depth_pub.publish(range_msg)
                return
            
            # Log selected contour
            self.get_logger().info(f"Selected horizon contour with top_y={min_y}, area={cv2.contourArea(horizon_contour):.0f}")
            
            # Extract top boundary points from the horizon contour
            top_boundary_points = []
            topmost_points = {}
            y_threshold = min_y + 5  # Only include points within 5 pixels of the topmost y
            
            # Find the topmost point for each x-coordinate, restricting to near the horizon
            for point in horizon_contour:
                x, y = point[0][0], point[0][1]
                if y <= y_threshold:  # Only include points near the top edge
                    if x not in topmost_points or y < topmost_points[x]:
                        topmost_points[x] = y
            
            # Convert to list of points
            for x, y in topmost_points.items():
                top_boundary_points.append((x, y))
            
            # Sort by x-coordinate for consistency
            top_boundary_points.sort(key=lambda p: p[0])
            
            if not top_boundary_points:
                self.get_logger().debug("No top boundary points found in horizon - publishing zero front depth")
                self.front_depth_pub.publish(range_msg)
                return
            
            # Focus on central ROI (80% of width) for stability
            roi_width = int(w * self.water_boundary_roi_width)
            roi_start_x = int((w - roi_width) / 2)
            roi_end_x = roi_start_x + roi_width
            
            # Corrected filtering: Use the entire (x, y) tuple
            filtered_top_points = [(x, y) for x, y in top_boundary_points 
                                if roi_start_x <= x <= roi_end_x]
            
            # If too few points in ROI, use all points
            if len(filtered_top_points) < 5:
                filtered_top_points = top_boundary_points
                self.get_logger().debug("Using all horizon points due to insufficient ROI points")
            
            # Get depths at boundary points
            valid_depths = []
            for x, y in filtered_top_points:
                if 0 <= y < h and 0 <= x < w:
                    depth_mm = depth_frame[y, x]
                    if self.min_depth_mm <= depth_mm <= self.max_depth_mm:  # Filter invalid depths
                        valid_depths.append((x, y, depth_mm))
            
            if not valid_depths:
                self.get_logger().debug("No valid depths at horizon - publishing zero front depth")
                self.front_depth_pub.publish(range_msg)
                return
            
            # Calculate depth (15th percentile for conservative estimate)
            depth_values = [d[2] for d in valid_depths]
            depth_mm = np.percentile(depth_values, 15)
            depth_m = depth_mm / 1000.0
            
            # Log depth points for debugging
            depth_log = [f"({x},{y}): {d/1000.0:.2f}m" for x, y, d in valid_depths[:5]]  # Log first 5 points
            self.get_logger().debug(f"Valid depth points (first 5): {'; '.join(depth_log)}")
            
            # Apply bounds
            depth_m = max(range_msg.min_range, min(range_msg.max_range, depth_m))
            
            # Smooth with previous value
            alpha = 0.3  # Smoothing factor
            self.last_water_boundary_depth = (1 - alpha) * self.last_water_boundary_depth + alpha * depth_m
            
            # Set the range to our calculated value
            range_msg.range = self.last_water_boundary_depth
            
            # Publish the range message
            self.front_depth_pub.publish(range_msg)
            
            # Visualize if requested
            if vis_frame is not None:
                # Draw only the horizon contour
                cv2.drawContours(vis_frame, [horizon_contour], -1, (0, 255, 255), 2)
                
                # Draw top boundary points
                for x, y in filtered_top_points:
                    cv2.circle(vis_frame, (x, y), 2, (0, 255, 0), -1)
                
                # Draw valid depth points with color-coded depth
                for x, y, d in valid_depths:
                    depth_m = d / 1000.0
                    normalized_depth = min(1.0, max(0.0, (depth_m - 0.5) / 7.5))
                    color = (
                        int(255 * normalized_depth),  # B
                        0,                            # G
                        int(255 * (1 - normalized_depth))  # R
                    )
                    cv2.circle(vis_frame, (x, y), 3, color, -1)
                
                # Add depth info
                cv2.putText(vis_frame, 
                            f"Horizon Depth: {self.last_water_boundary_depth:.2f}m", 
                            (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Add contour info
                cv2.putText(vis_frame, 
                            f"Horizon Contour (top_y={min_y}, {len(contours)} total)", 
                            (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Log periodically
            if self.frame_counter % 50 == 0:
                self.get_logger().info(f"Horizon depth: {self.last_water_boundary_depth:.2f}m, "
                                    f"top_y={min_y}, selected 1/{len(contours)} contours, "
                                    f"{len(valid_depths)} valid depth points")
        
        except Exception as e:
            self.get_logger().error(f"Error processing water boundary: {str(e)}")
            self.get_logger().error(traceback.format_exc())
            # Publish zero on error
            self.front_depth_pub.publish(range_msg)

    # Modify the process_data method to handle empty detections better
    def process_data(self):
        try:
            # Update collection sequence first
            self.update_collection_sequence()
            
            
            # Increment frame counter
            self.frame_counter += 1
            
            # Check if it's time to perform memory cleanup
            current_time = time.time()
            if current_time - self.last_memory_cleanup > self.memory_cleanup_interval:
                self.frame_buffer.clear_old_data()
                self.last_memory_cleanup = current_time
            
            # Check for frame validity (camera disconnect)
            if current_time - self.last_valid_frame_time > self.frame_validity_timeout:
                self.get_logger().warn("No valid frames received recently - camera may be disconnected")
                self.publish_empty_messages()
                return
            
            # Check if tracker needs to be reset due to timeouts
            # Just check and potentially reset, but don't return early - continue processing other data
            self.check_and_handle_tracker_reset()
            
            # Check for obstacle timeouts and clear them if necessary
            if self.frame_buffer.check_obstacle_timeout(self.obstacle_timeout):
                # Only clear if it's been a while since last clear
                if not hasattr(self, 'last_obstacle_clear_time'):
                    self.last_obstacle_clear_time = 0
                    
                # Clear obstacles at most once per timeout period
                if current_time - self.last_obstacle_clear_time > self.obstacle_timeout:
                    self.get_logger().info("Obstacles timed out - clearing obstacle data")
                    # If there were previously published obstacles, publish empty messages
                    if self.last_published_obstacles:
                        self.last_published_obstacles = []
                        self.publish_empty_messages()
                    self.last_obstacle_clear_time = current_time
            
            # Get latest data - ALWAYS process this data regardless of detection status
            rgb_frame, depth_frame, water_mask, detections, obstacles = self.frame_buffer.get_latest_data()
            
            if rgb_frame is None or depth_frame is None:
                return
            
            # Handle robot mask resizing if enabled
            if self.enable_robot_mask:
                if self.robot_mask is None:
                    self.create_default_robot_mask(rgb_frame.shape)
                else:
                    self.resize_robot_mask_if_needed(rgb_frame.shape)
            
            # Create a copy of the RGB frame for visualization
            vis_frame = rgb_frame.copy()
            
            # ALWAYS process water boundary regardless of detection status
            # This is crucial for navigation even when no objects are detected
            if depth_frame is not None:
                self.process_water_boundary(water_mask, depth_frame, vis_frame)
            
            # Process trackers separately for objects and obstacles
            if self.use_tracker:
                try:
                    # Process object detections
                    object_dets = [d for d in detections if d.get('type', 'object') == 'object']
                    if object_dets:
                        object_boxes = [[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['score']] 
                                    for d in object_dets]
                        tracked_objects = self.object_tracker.update(np.array(object_boxes))
                        
                        # Update object detections with track IDs
                        self._update_tracks(object_dets, tracked_objects)
                    else:
                        # Update with empty array to maintain tracking
                        _ = self.object_tracker.update(np.empty((0, 5)))
                        
                    # Process obstacle detections
                    obstacle_dets = [d for d in detections if d.get('type', 'obstacle') == 'obstacle']
                    if obstacle_dets:
                        obstacle_boxes = [[d['bbox'][0], d['bbox'][1], d['bbox'][2], d['bbox'][3], d['score']] 
                                        for d in obstacle_dets]
                        tracked_obstacles = self.obstacle_tracker.update(np.array(obstacle_boxes))
                        
                        # Update obstacle detections with track IDs
                        self._update_tracks(obstacle_dets, tracked_obstacles)
                    else:
                        # Update with empty array to maintain tracking
                        _ = self.obstacle_tracker.update(np.empty((0, 5)))
                        
                except Exception as e:
                    self.get_logger().error(f"Error in SORT tracking: {e}")
                    self.get_logger().error(traceback.format_exc())
            
            # Check if obstacles overlap with objects, and if so, remove the obstacles
            # as objects have priority (only if we have both)
            if obstacles and detections:
                filtered_obstacles = []
                for obstacle in obstacles:
                    obstacle_bbox = obstacle['bbox']
                    is_overlapping = False
                    
                    for obj in detections:
                        obj_bbox = obj['bbox']
                        iou = self.calculate_iou(obstacle_bbox, obj_bbox)
                        if iou > 0.1:  # Significant overlap
                            is_overlapping = True
                            break
                    
                    if not is_overlapping:
                        filtered_obstacles.append(obstacle)
                
                obstacles = filtered_obstacles
            
            # Draw robot mask boundary on visualization frame if enabled
            if self.enable_robot_mask and self.robot_mask is not None and vis_frame is not None:
                try:
                    # Create a visualization overlay for the robot mask
                    robot_overlay = np.zeros_like(vis_frame)
                    robot_overlay[:,:,0] = 0                # B
                    robot_overlay[:,:,1] = 0                # G
                    robot_overlay[:,:,2] = self.robot_mask  # R
                    
                    # Add semi-transparent overlay
                    alpha = 0.4  # Increased visibility
                    mask_vis = cv2.addWeighted(vis_frame, 1, robot_overlay, alpha, 0)
                    
                    # Apply to areas where mask is non-zero
                    mask_indices = self.robot_mask > 0
                    vis_frame[mask_indices] = mask_vis[mask_indices]
                    
                    # Also draw contours
                    contours, _ = cv2.findContours(self.robot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_frame, contours, -1, (255, 0, 255), 2)  # Magenta for robot mask
                    
                    # Add text label
                    cv2.putText(vis_frame, "Robot Mask", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    
                    # Publish robot mask if enabled
                    if self.robot_mask_publisher is not None:
                        try:
                            robot_mask_msg = self.bridge.cv2_to_imgmsg(self.robot_mask, encoding="mono8")
                            robot_mask_msg.header.stamp = self.get_clock().now().to_msg()
                            robot_mask_msg.header.frame_id = "camera"
                            self.robot_mask_publisher.publish(robot_mask_msg)
                        except Exception as e:
                            self.get_logger().error(f"Error publishing robot mask: {e}")
                except Exception as e:
                    self.get_logger().error(f"Error drawing robot mask: {e}")
                    self.get_logger().error(traceback.format_exc())
            
            # Combine detections and obstacles for processing
            combined_detections = detections.copy() if detections else []
            combined_detections.extend(obstacles if obstacles else [])
            
            # Filter detections that are no longer visible in frame
            frame_shape = rgb_frame.shape
            visible_detections = []
            for det in combined_detections:
                # Only keep detections that are in the frame
                if self.check_bbox_in_frame(det['bbox'], frame_shape):
                    visible_detections.append(det)
            
            # Fuse detections with depth and water mask
            object_positions, obstacle_positions, obstacle_sizes = self.fuse_detections(visible_detections, depth_frame, water_mask, vis_frame)
            
            # Update: Check if we need to publish empty messages
            if not object_positions and not obstacle_positions:
                # If we had published objects or obstacles before, but now none, publish empty arrays
                if self.last_published_objects or self.last_published_obstacles:
                    self.get_logger().info("No objects or obstacles detected - publishing empty messages")
                    self.publish_empty_messages()
                    # Also clear the stored data
                    self.last_published_objects = []
                    self.last_published_obstacles = []
            else:
                # We have valid positions, publish them
                # Publish point cloud with 3D positions using our new coordinate mapping
                self.publish_point_cloud(object_positions, obstacle_positions)
                
                # Update last published data
                self.last_published_objects = object_positions.copy() if object_positions else []
                self.last_published_obstacles = obstacle_positions.copy() if obstacle_positions else []
                
                # Publish object poses, obstacle poses, and sizes
                self.publish_object_poses(object_positions, obstacle_positions, obstacle_sizes)
                
            # Always publish empty messages for objects if no object positions, even if there are obstacles
            if not object_positions and self.last_published_objects:
                self.get_logger().info("No object positions - publishing empty object messages")
                # Publish empty object pose array
                empty_pose_array = PoseArray()
                empty_pose_array.header.stamp = self.get_clock().now().to_msg()
                empty_pose_array.header.frame_id = "map"
                self.object_poses_pub.publish(empty_pose_array)
                
                # Publish empty pose stamped
                empty_pose_stamped = PoseStamped()
                empty_pose_stamped.header.stamp = self.get_clock().now().to_msg()
                empty_pose_stamped.header.frame_id = "map"
                self.object_pose_stamped_pub.publish(empty_pose_stamped)
                
                # Clear last published objects
                self.last_published_objects = []
                
            # Always publish empty messages for obstacles if no obstacle positions, even if there are objects
            if not obstacle_positions and self.last_published_obstacles:
                self.get_logger().info("No obstacle positions - publishing empty obstacle messages")
                # Publish empty obstacle pose array
                empty_pose_array = PoseArray()
                empty_pose_array.header.stamp = self.get_clock().now().to_msg()
                empty_pose_array.header.frame_id = "map"
                self.obstacle_poses_pub.publish(empty_pose_array)
                
                # Publish empty size array
                empty_sizes = Float32MultiArray()
                self.obstacle_sizes_pub.publish(empty_sizes)
                
                # Clear last published obstacles
                self.last_published_obstacles = []
            
            # ALWAYS draw water line boundary on visualization frame if available
            if water_mask is not None:
                try:
                    contours, _ = cv2.findContours(water_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(vis_frame, contours, -1, (0, 255, 255), 2)
                except Exception as e:
                    self.get_logger().error(f"Error drawing water line: {e}")
            
            # ALWAYS publish visualization frames, water mask, etc.
            if vis_frame is not None:
                try:
                    vis_msg = self.bridge.cv2_to_imgmsg(vis_frame, "bgr8")
                    vis_msg.header.stamp = self.get_clock().now().to_msg()
                    vis_msg.header.frame_id = "camera"
                    self.fused_image_pub.publish(vis_msg)
                    
                    if self.frame_publisher is not None:
                        self.frame_publisher.publish(vis_msg)
                    
                    if water_mask is not None and self.mask_publisher is not None:
                        mask_msg = self.bridge.cv2_to_imgmsg(water_mask * 255, encoding="mono8")
                        mask_msg.header.stamp = vis_msg.header.stamp
                        mask_msg.header.frame_id = vis_msg.header.frame_id
                        self.mask_publisher.publish(mask_msg)
                    
                except Exception as e:
                    self.get_logger().error(f"Error publishing images: {e}")
            
        except Exception as e:
            self.get_logger().error(f"Error in process_data: {e}")
            self.get_logger().error(traceback.format_exc())

    def _update_tracks(self, detections, tracked_boxes):
        """Helper method to update detections with track IDs"""
        for track in tracked_boxes:
            tx1, ty1, tx2, ty2, track_id = track.astype(int)
            
            # Find best matching detection
            best_match = None
            best_iou = 0.0
            
            for i, det in enumerate(detections):
                x1, y1, x2, y2 = det['bbox']
                iou = self.calculate_iou((tx1, ty1, tx2, ty2), (x1, y1, x2, y2))
                
                if iou > best_iou:
                    best_iou = iou
                    best_match = i
            
            # Update detection with track ID if match found
            if best_match is not None and best_iou > self.tracker_iou_threshold:
                detections[best_match]['track_id'] = int(track_id)

    def calculate_iou(self, bbox1, bbox2):
        """Calculate IoU between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bb1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bb2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        
        return iou
        
    def calculate_depth_and_angle(self, bbox, depth_map, frame_center_x, max_angle):
        """
        Compute the depth of the object using the raw depth values.
        The depth_map now contains actual depth values in mm (not normalized 0-255).
        Prioritizes closer depth values, especially those near the center of the bounding box.
        Filters out extreme depth variations to avoid background influence.
        Returns None for depth if no valid depth data is available (object too far).
        """
        x1, y1, x2, y2 = map(int, bbox)
        h, w = depth_map.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w - 1))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h - 1))
        if x2 <= x1 or y2 <= y1:
            return None, 0.0, 0.0  # Invalid bbox - return None for depth

        # Calculate central ROI (60% of the bounding box)
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Define central region (about 60% of the original bbox)
        central_width = max(3, int(bbox_width * 0.6))
        central_height = max(3, int(bbox_height * 0.6))
        
        # Calculate central region bounds
        cx1 = max(0, center_x - central_width // 2)
        cx2 = min(w - 1, center_x + central_width // 2)
        cy1 = max(0, center_y - central_height // 2)
        cy2 = min(h - 1, center_y + central_height // 2)
        
        # Extract central ROI and full ROI
        central_roi = depth_map[cy1:cy2, cx1:cx2]
        full_roi = depth_map[y1:y2, x1:x2]
        
        # Get valid depths (non-zero) in meters for easier threshold comparison
        valid_central_depths_mm = central_roi[central_roi > 0]
        valid_full_depths_mm = full_roi[full_roi > 0]
        
        # Function to filter out extreme depth variations
        def filter_depth_outliers(depths_mm):
            if depths_mm.size == 0:
                return np.array([])
                
            # Convert to meters for easier threshold comparison
            depths_m = depths_mm / 1000.0
            
            # Find the dominant depth range using a histogram approach
            # First, get a rough estimate of foreground depth using lower percentile
            initial_depth_m = np.percentile(depths_m, 15)
            
            # Define what's considered a significant depth difference (in meters)
            # Objects closer than 1.5m: use 0.5m threshold
            # Objects further away: use 1/3 of the depth as threshold
            if initial_depth_m < 1.5:
                depth_threshold = 0.5  # 0.5m threshold for close objects
            else:
                depth_threshold = initial_depth_m / 3.0  # More tolerance for distant objects
                
            # Filter depths to keep only those within threshold of initial estimate
            filtered_depths_m = depths_m[np.abs(depths_m - initial_depth_m) < depth_threshold]
            
            # If filtering removed too many points, revert to original with a warning
            if filtered_depths_m.size < depths_m.size * 0.2:  # Less than 20% remaining
                self.get_logger().warn(f"Depth filtering too aggressive: {filtered_depths_m.size}/{depths_m.size} points remain")
                return depths_mm
                
            # Convert back to mm
            return filtered_depths_m * 1000.0
        
        # Apply outlier filtering
        filtered_central_depths = filter_depth_outliers(valid_central_depths_mm)
        filtered_full_depths = filter_depth_outliers(valid_full_depths_mm)
        
        # Calculate object depth in meters - prioritize closer objects
        object_depth_m = None  # Default to None for objects without valid depth
        
        if filtered_central_depths.size > 10:  # If we have enough central depth points
            # Use the 15th percentile to get closer values while avoiding noise/outliers
            object_depth_mm = np.percentile(filtered_central_depths, 15)
            object_depth_m = max(0.5, min(5.0, object_depth_mm / 1000.0))
            self.get_logger().debug(f"Using filtered central ROI depths: {filtered_central_depths.size}/{valid_central_depths_mm.size} points")
        elif filtered_full_depths.size > 0:
            # Fall back to full ROI if central has too few points
            object_depth_mm = np.percentile(filtered_full_depths, 15)
            object_depth_m = max(0.5, min(5.0, object_depth_mm / 1000.0))
            self.get_logger().debug(f"Using filtered full ROI depths: {filtered_full_depths.size}/{valid_full_depths_mm.size} points")
        else:
            # No valid depths found or all filtered out
            self.get_logger().debug("No valid depths found after filtering - object may be too far away")
            
        # Calculate angle regardless of depth availability
        bbox_center_x = (x1 + x2) / 2.0
        pixel_offset = frame_center_x - bbox_center_x
        normalized_offset = pixel_offset / frame_center_x
        angle_deg = normalized_offset * max_angle

        # Calculate real-world width if depth is available
        bbox_width_3d = 0.0
        if object_depth_m is not None:
            bbox_width_pixels = x2 - x1
            fov_rad = math.radians(self.fov_degrees)
            pixel_to_meter_factor = 2 * object_depth_m * math.tan(fov_rad / 2) / w
            bbox_width_3d = bbox_width_pixels * pixel_to_meter_factor

        if object_depth_m is not None:
            self.get_logger().debug(f"Depth calculation: object at {object_depth_m:.2f}m, angle {angle_deg:.1f}°, width {bbox_width_3d:.2f}m")
        else:
            self.get_logger().debug(f"Depth calculation: object too far or no valid depth, angle {angle_deg:.1f}°")
            
        return object_depth_m, angle_deg, bbox_width_3d

    def check_water_mask(self, bbox, water_mask, threshold=0.5):
        if water_mask is None:
            return True
            
        x1, y1, x2, y2 = map(int, bbox)
        h, w = water_mask.shape[:2]
        x1 = max(0, min(x1, w-1))
        y1 = max(0, min(y1, h-1))
        x2 = max(0, min(x2, w-1))
        y2 = max(0, min(y2, h-1))
        
        if x2 <= x1 or y2 <= y1:
            return False
            
        roi = water_mask[y1:y2, x1:x2]
        if roi.size == 0:
            return False
            
        water_ratio = np.sum(roi) / roi.size
        return water_ratio > threshold

    def check_overlap(self, bbox1, bbox2, iou_threshold=0.5):
        iou = self.calculate_iou(bbox1, bbox2)
        return iou > iou_threshold

    def fuse_detections(self, detections, depth_map, water_mask, vis_frame):
        """
        Fuse detections with depth and water mask data.
        Also check robot mask to filter out objects that are in the robot area.
        """
        # Initialize empty arrays for return
        object_positions = []
        obstacle_positions = []
        obstacle_sizes = []
        
        # If no detections, return early but log the condition
        if not detections:
            self.get_logger().debug("No detections to process")
            return object_positions, obstacle_positions, obstacle_sizes
            
        # Get frame dimensions for calculations
        frame_height, frame_width = vis_frame.shape[:2]
        center_x = frame_width / 2.0
        max_angle = self.fov_degrees / 2.0
        
        valid_objects = []
        valid_obstacles = []
        processed_bboxes = []
        
        # Sort detections by confidence score
        sorted_detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        # Process each detection
        for det in sorted_detections:
            bbox = det['bbox']
            score = det['score']
            track_id = det.get('track_id', None)
            entity_type = det.get('type', 'object')
            bbox_width = det.get('width', 0)
            
            x1, y1, x2, y2 = map(int, bbox)
            center_point = ((y1 + y2) // 2, (x1 + x2) // 2)
            
            # For obstacles, use the depth from VIO features if available
            if entity_type == 'obstacle' and 'vio_depth' in det and det['vio_depth'] > 0:
                # Use depth calculated from VIO features for obstacles
                depth = det['vio_depth'] / 1000.0  # Convert mm to meters
                # Use angle calculated from VIO features
                angle_deg = det.get('vio_angle_deg', 0.0)
                # Calculate 3D width based on depth and field of view
                bbox_width_3d = det.get('width', x2 - x1)
                pixel_to_meter_factor = 2 * depth * math.tan(math.radians(self.fov_degrees / 2)) / frame_width
                bbox_width_3d = bbox_width_3d * pixel_to_meter_factor
            else:
                # Use standard depth calculation for objects and fallback for obstacles
                depth, angle_deg, bbox_width_3d = self.calculate_depth_and_angle(bbox, depth_map, center_x, max_angle)
            
            # Apply different thresholds for obstacles vs objects
            threshold = self.obstacle_water_threshold if entity_type == 'obstacle' else self.object_water_threshold
            in_water = self.check_water_mask(bbox, water_mask, threshold)
            in_robot_mask = self.check_robot_mask(bbox)
            overlapping = any(self.check_overlap(bbox, pb) for pb in processed_bboxes)
            
            # Acceptance criteria:
            # - For obstacles: Accept regardless of water (they can be on shore)
            # - For objects: Must be in water and not in robot mask
            # - For both: Avoid overlapping detections
            
            # Debug the detection status
            status_parts = []
            if entity_type == 'obstacle':
                status_parts.append("obstacle")
            else:
                status_parts.append("object")
                
            if in_water:
                status_parts.append("in_water")
            if in_robot_mask:
                status_parts.append("in_robot_mask")
            if overlapping:
                status_parts.append("overlapping")
                
            status_str = f"{entity_type} [{','.join(status_parts)}]"
            
            # The main acceptance logic
            if ((entity_type == 'obstacle') or (in_water and not in_robot_mask)) and not overlapping:
                detection_data = {
                    'bbox': bbox,
                    'score': score,
                    'depth': depth,
                    'angle_deg': angle_deg,
                    'center': center_point,
                    'track_id': track_id,
                    'type': entity_type,
                    'y_position': (y1 + y2) / 2,
                    'width_3d': bbox_width_3d,
                }
                
                if entity_type == 'obstacle':
                    valid_obstacles.append(detection_data)
                    # Store obstacle size in meters
                    obstacle_sizes.append(bbox_width_3d)
                    self.get_logger().debug(f"ACCEPTED {status_str} at depth={depth:.2f}m, angle={angle_deg:.1f}°")
                else:
                    valid_objects.append(detection_data)
                    self.get_logger().debug(f"ACCEPTED {status_str} at depth={depth:.2f}m, angle={angle_deg:.1f}°")
                    
                processed_bboxes.append(bbox)
            else:
                # Log rejected detections for debugging
                self.get_logger().debug(f"REJECTED {status_str} at depth={depth:.2f}m, angle={angle_deg:.1f}°")
                
                # Pick color based on rejection reason for visualization
                if in_robot_mask:
                    color = (255, 0, 255)  # Magenta for robot mask
                    reason = "In robot mask"
                elif not in_water and entity_type != 'obstacle':
                    color = (0, 0, 255)  # Red for not in water
                    reason = "Not in water"
                else:
                    color = (0, 165, 255)  # Orange for overlapping
                    reason = "Overlapping"
                
                # Visualize rejected detection
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis_frame, reason, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Assign priorities for objects based on depth (closer objects get higher priority)
        if valid_objects:
            # Sort by depth first, then y-position (lower in frame if same depth)
            valid_objects.sort(key=lambda x: (x['depth'] if x['depth'] is not None else float('inf'), -x['y_position']))
            
            # Assign priorities and visualize accepted objects
            for i, obj in enumerate(valid_objects):
                obj['priority'] = i + 1
                x1, y1, x2, y2 = map(int, obj['bbox'])
                track_id = obj.get('track_id', None)
                priority = obj['priority']
                
                # Color based on priority (green for highest, yellow for others)
                if priority == 1:
                    color = (0, 255, 0)
                else:
                    color = (0, 255, 255)
                
                # Draw bounding box
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Add multiple text lines with information
                line_spacing = 15
                lines = [f"Object P{priority}", f"D:{obj['depth']:.1f}m, A:{obj['angle_deg']:.0f}°", 
                        f"W:{obj['width_3d']:.2f}m", f"C:{obj['score']:.2f}"]
                if track_id is not None:
                    lines.append(f"ID:{track_id}")
                
                for j, line in enumerate(lines):
                    y_text = y1 - 10 - (j * line_spacing)
                    if y_text > 0:
                        cv2.putText(vis_frame, line, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add colored depth circle (blue->red gradient based on depth)
                center_x_box = (x1 + x2) // 2
                center_y_box = (y1 + y2) // 2
                normalized_depth = max(0, min(1, (obj['depth'] - 0.5) / 4.5))
                blue = int(255 * (1 - normalized_depth))
                red = int(255 * normalized_depth)
                depth_color = (blue, 0, red)
                cv2.circle(vis_frame, (center_x_box, center_y_box), 5, depth_color, -1)
        
        # Visualize accepted obstacles
        for obstacle in valid_obstacles:
            x1, y1, x2, y2 = map(int, obstacle['bbox'])
            color = (0, 0, 255)  # Red for obstacles
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
            
            line_spacing = 15
            lines = ["Obstacle", f"D:{obstacle['depth']:.1f}m, A:{obstacle['angle_deg']:.0f}°", 
                    f"W:{obstacle['width_3d']:.2f}m"]
                    
            for i, line in enumerate(lines):
                y_text = y1 - 10 - (i * line_spacing)
                if y_text > 0:
                    cv2.putText(vis_frame, line, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
            # Add colored depth circle
            center_x_box = (x1 + x2) // 2
            center_y_box = (y1 + y2) // 2
            normalized_depth = max(0, min(1, (obstacle['depth'] - 0.5) / 4.5))
            blue = int(255 * (1 - normalized_depth))
            red = int(255 * normalized_depth)
            depth_color = (blue, 0, red)
            cv2.circle(vis_frame, (center_x_box, center_y_box), 5, depth_color, -1)
        
        # Convert valid objects and obstacles to position arrays for publishing
        object_positions = []
        for obj in valid_objects:
            depth = obj['depth']
            # Skip objects with invalid depth
            if depth is None:
                continue
                
            angle_deg = obj['angle_deg']
            angle_rad = math.radians(angle_deg)
            track_id = obj.get('track_id', -1)
            priority = obj.get('priority', 999)
            width_3d = obj.get('width_3d', 0)
            
            # Convert to robot coordinate system
            x_robot = depth                        # forward distance along x
            y_robot = depth * math.tan(angle_rad)  # lateral offset (left/right)
            z_robot = 0                            # height (up/down)
            
            # Filter objects by valid depth range
            if 0.5 <= depth <= 5.0:
                object_positions.append([x_robot, y_robot, z_robot, angle_deg, track_id, priority, width_3d])
                
        obstacle_positions = []
        for i, det in enumerate(valid_obstacles):
            depth = det['depth']
            # Skip obstacles with invalid depth
            if depth is None:
                continue
                
            angle_deg = det['angle_deg']
            angle_rad = math.radians(angle_deg)
            width_3d = det.get('width_3d', 0)
            
            # Convert to robot coordinate system
            x_robot = depth
            y_robot = depth * math.tan(angle_rad)
            z_robot = 0
            
            # Filter obstacles by valid depth range
            if 0.5 <= depth <= 5.0:
                obstacle_positions.append([x_robot, y_robot, z_robot, angle_deg, -1, width_3d])
                # Store the 3D width
                if i < len(obstacle_sizes):
                    obstacle_sizes[i] = width_3d
        
        # Log what we found
        self.get_logger().debug(f"Found {len(object_positions)} valid objects and {len(obstacle_positions)} valid obstacles")
        
        return object_positions, obstacle_positions, obstacle_sizes

    def publish_point_cloud(self, object_positions, obstacle_positions):
        pc_msg = PointCloud()
        pc_msg.header.stamp = self.get_clock().now().to_msg()
        pc_msg.header.frame_id = "map"
        
        # Use our new coordinate mapping for points.
        for pos in object_positions:
            x_robot, y_robot, z_robot = pos[0], pos[1], pos[2]
            point = Point32()
            point.x = float(x_robot)
            point.y = float(y_robot)
            point.z = float(z_robot)
            pc_msg.points.append(point)
            
        for pos in obstacle_positions:
            x_robot, y_robot, z_robot = pos[0], pos[1], pos[2]
            point = Point32()
            point.x = float(x_robot)
            point.y = float(y_robot)
            point.z = float(z_robot)
            pc_msg.points.append(point)
        
        self.point_cloud_pub.publish(pc_msg)


    def publish_object_poses(self, object_positions, obstacle_positions, obstacle_sizes):
        """
        SIMPLIFIED pose publishing - always publish arrays (empty or with data).
        Use NaN only when the vision system itself is completely offline.
        """
        current_time = self.get_clock().now().to_msg()
        frame_id = "map"
        
        # OBJECTS
        pose_array = PoseArray()
        pose_array.header.stamp = current_time
        pose_array.header.frame_id = frame_id
        
        if object_positions:
            for i, pos in enumerate(object_positions):
                x_robot, y_robot, z_robot, angle_deg, track_id, priority, width_3d = pos
                
                pose = Pose()
                pose.position.x = float(x_robot)
                pose.position.y = float(y_robot)
                pose.position.z = float(z_robot)
                
                object_yaw = math.radians(angle_deg)
                pose.orientation.w = math.cos(object_yaw / 2.0)
                pose.orientation.x = 0.0
                pose.orientation.y = float(width_3d)
                pose.orientation.z = math.sin(object_yaw / 2.0)
                
                pose_array.poses.append(pose)

                # Publish highest priority object as stamped pose
                if priority == 1:
                    pose_stamped = PoseStamped()
                    pose_stamped.header = pose_array.header
                    pose_stamped.pose = pose
                    self.object_pose_stamped_pub.publish(pose_stamped)
        else:
            # EMPTY OBJECTS: Don't add anything to the array
            # The behavior system will see an empty array and know no objects are detected
            
            # Also publish empty stamped pose
            empty_pose_stamped = PoseStamped()
            empty_pose_stamped.header = pose_array.header
            # Leave pose as all zeros (not NaN) to indicate "no object but system is working"
            self.object_pose_stamped_pub.publish(empty_pose_stamped)
        
        # OBSTACLES
        obstacle_pose_array = PoseArray()
        obstacle_pose_array.header = pose_array.header
        size_msg = Float32MultiArray()
        
        if obstacle_positions:
            for i, pos in enumerate(obstacle_positions):
                x_robot, y_robot, z_robot, angle_deg, _, width_3d = pos
                
                size_msg.data.append(float(width_3d))
                
                pose = Pose()
                pose.position.x = float(x_robot)
                pose.position.y = float(y_robot)
                pose.position.z = float(z_robot)
                
                obstacle_yaw = math.radians(angle_deg)
                pose.orientation.w = math.cos(obstacle_yaw / 2.0)
                pose.orientation.x = 0.0
                pose.orientation.y = float(width_3d)
                pose.orientation.z = math.sin(obstacle_yaw / 2.0)
                
                obstacle_pose_array.poses.append(pose)
        else:
            # EMPTY OBSTACLES: Don't add anything to the array
            pass
        
        # ALWAYS PUBLISH (empty arrays are valid and important!)
        self.object_poses_pub.publish(pose_array)
        self.obstacle_poses_pub.publish(obstacle_pose_array)
        self.obstacle_sizes_pub.publish(size_msg)
        
        # Update tracking
        self.last_published_objects = object_positions.copy() if object_positions else []
        self.last_published_obstacles = obstacle_positions.copy() if obstacle_positions else []
        
        # Create markers for visualization
        self.publish_markers(object_positions, obstacle_positions, pose_array.header)
        
        # Log what we published
        obj_count = len(object_positions) if object_positions else 0
        obs_count = len(obstacle_positions) if obstacle_positions else 0
        self.get_logger().debug(f"Published: {obj_count} objects, {obs_count} obstacles")

    def publish_markers(self, object_positions, obstacle_positions, header):
        """Create and publish visualization markers."""
        marker_array = MarkerArray()
        
        # Clear all markers first
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        delete_marker.header = header
        marker_array.markers.append(delete_marker)
        
        # Add object markers
        if object_positions:
            for i, pos in enumerate(object_positions):
                x_robot, y_robot, z_robot, angle_deg, track_id, priority, width_3d = pos
                
                # Create sphere marker
                marker = Marker()
                marker.header = header
                marker.ns = "object_markers"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                
                marker.pose.position.x = float(x_robot)
                marker.pose.position.y = float(y_robot)
                marker.pose.position.z = float(z_robot)
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = max(0.3, width_3d)
                marker.scale.y = max(0.3, width_3d)
                marker.scale.z = 0.3
                marker.color.a = 0.8
                
                # Color based on priority
                if priority == 1:
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                else:
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                
                marker_array.markers.append(marker)
                
                # Add text label
                text_marker = Marker()
                text_marker.header = header
                text_marker.ns = "object_labels"
                text_marker.id = i
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose = marker.pose
                text_marker.pose.position.z += 0.3
                
                label_text = f"Object P{priority}"
                if track_id is not None and track_id >= 0:
                    label_text += f" ID:{track_id}"
                text_marker.text = label_text
                text_marker.scale.z = 0.2
                text_marker.color.a = 1.0
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                
                marker_array.markers.append(text_marker)
        
        # Add obstacle markers
        if obstacle_positions:
            for i, pos in enumerate(obstacle_positions):
                x_robot, y_robot, z_robot, angle_deg, _, width_3d = pos
                
                # Create cube marker
                marker = Marker()
                marker.header = header
                marker.ns = "obstacle_markers"
                marker.id = i
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                marker.pose.position.x = float(x_robot)
                marker.pose.position.y = float(y_robot)
                marker.pose.position.z = float(z_robot)
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = max(0.3, width_3d)
                marker.scale.y = max(0.3, width_3d)
                marker.scale.z = 0.3
                marker.color.a = 0.8
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                
                marker_array.markers.append(marker)
                
                # Add text label
                text_marker = Marker()
                text_marker.header = header
                text_marker.ns = "obstacle_labels"
                text_marker.id = i
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose = marker.pose
                text_marker.pose.position.z += 0.3
                text_marker.text = "Obstacle"
                text_marker.scale.z = 0.2
                text_marker.color.a = 1.0
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.b = 1.0
                
                marker_array.markers.append(text_marker)
        
        # Always publish markers
        self.markers_pub.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = visionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down vision fusion node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()