#!/usr/bin/env python3

"""
Enhanced Behavior System with State Persistence

This adds hysteresis to prevent rapid state switching due to temporary data loss.
Key improvements:
1. Minimum time in state before switching
2. Grace period for target loss
3. Different timeouts for losing vs regaining targets
"""

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, Point, Quaternion
from std_msgs.msg import String, Float32MultiArray
from visualization_msgs.msg import MarkerArray, Marker
from sensor_msgs.msg import Range, Image
import numpy as np
import enum
import math
import time
import threading
from collections import deque
from copy import deepcopy
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String, Float32MultiArray, Float32

class BehaviorState(enum.Enum):
    IDLE = "idle"
    EXPLORING = "exploring"
    TARGET_SEEKING = "target_seeking"

class BehaviorSystem(Node):
    def __init__(self):
        super().__init__('behavior_system')
        
        # Parameters - simplified
        self.declare_parameter('target_distance_threshold', 1.0)
        self.declare_parameter('obstacle_detection_radius', 0.5)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('water_mask_timeout', 5.0)
        self.declare_parameter('target_timeout', 1.5)  # Normal timeout for acquiring targets
        
        # NEW: State persistence parameters
        self.declare_parameter('target_loss_grace_period', 3.0)  # Grace period before switching from TARGET_SEEKING
        self.declare_parameter('min_state_duration', 2.0)  # Minimum time to stay in a state before switching
        self.declare_parameter('rapid_switch_prevention', True)  # Enable/disable the feature
        
        self.declare_parameter('k_att', 1.0)
        self.declare_parameter('k_rep', 40.0)
        self.declare_parameter('rho_0', 1.0)
        self.declare_parameter('d_star', 2.0)
        self.declare_parameter('resolution', 0.1)
        self.declare_parameter('debug_obstacles', False)
        self.declare_parameter('robot_length', 1.0)
        self.declare_parameter('robot_width', 0.8)
        self.declare_parameter('camera_front_offset', 0.0)
        self.declare_parameter('safety_margin', 0.05)
        self.declare_parameter('visualization_level', 1)
        self.declare_parameter('visualization_interval', 0.5)
        self.declare_parameter('path_publish_rate', 10.0)
        self.declare_parameter('use_prediction', True)
        self.declare_parameter('prediction_horizon', 0.5)
        self.declare_parameter('enable_turn_in_place', True)
        
        # Get parameter values
        self.target_distance_threshold = self.get_parameter('target_distance_threshold').value
        self.obstacle_detection_radius = self.get_parameter('obstacle_detection_radius').value
        self.frame_id = self.get_parameter('frame_id').value
        self.water_mask_timeout = self.get_parameter('water_mask_timeout').value
        self.target_timeout = self.get_parameter('target_timeout').value
        
        # NEW: State persistence parameters
        self.target_loss_grace_period = self.get_parameter('target_loss_grace_period').value
        self.min_state_duration = self.get_parameter('min_state_duration').value
        self.rapid_switch_prevention = self.get_parameter('rapid_switch_prevention').value
        
        k_att = self.get_parameter('k_att').value
        k_rep = self.get_parameter('k_rep').value
        rho_0 = self.get_parameter('rho_0').value
        d_star = self.get_parameter('d_star').value
        resolution = self.get_parameter('resolution').value
        self.debug_obstacles = self.get_parameter('debug_obstacles').value
        self.robot_length = self.get_parameter('robot_length').value
        self.robot_width = self.get_parameter('robot_width').value
        self.camera_front_offset = self.get_parameter('camera_front_offset').value
        self.safety_margin = self.get_parameter('safety_margin').value
        visualization_level = self.get_parameter('visualization_level').value
        visualization_interval = self.get_parameter('visualization_interval').value
        self.path_publish_rate = self.get_parameter('path_publish_rate').value
        self.use_prediction = self.get_parameter('use_prediction').value
        self.prediction_horizon = self.get_parameter('prediction_horizon').value
        self.enable_turn_in_place = self.get_parameter('enable_turn_in_place').value

        # Initialize CvBridge for image conversions
        self.bridge = CvBridge()
        
        # Import VisionGuidedPlanner
        from path_planner.vision_planner import VisionGuidedPlanner
        
        # Initialize the refactored planner
        self.vision_planner = VisionGuidedPlanner(
            k_att=k_att,
            k_rep=k_rep,
            rho_0=rho_0,
            d_star=d_star,
            resolution=resolution,
            robot_length=self.robot_length,
            robot_width=self.robot_width,
            safety_margin=self.safety_margin,
            visualization_interval=visualization_interval
        )
        
        # Configure visualization parameters
        self.vision_planner.visualization_level = visualization_level
        
        # Configure the influence radius with reduced values
        self.vision_planner.wavefront_influence_base = 0.05
        self.vision_planner.wavefront_influence_max = 0.15
        
        # Create locks for thread safety
        self.path_lock = threading.Lock()
        self.target_lock = threading.Lock()
        self.obstacle_lock = threading.Lock()
        self.water_mask_lock = threading.Lock()
        self.depth_map_lock = threading.Lock()
        
        # Initialize the water explorer
        try:
            from path_planner.water_explorer import PixelBasedWaterExplorer
            self.water_explorer = PixelBasedWaterExplorer(node=self)
            
            # Configure water explorer for lightweight operation
            self.water_explorer.density_weight = 1.0
            self.water_explorer.grid_step = 20
            self.water_explorer.max_candidates = 50
            self.water_explorer.goal_timeout = 2.0
            
            # Set thread locks for water explorer
            self.water_explorer.mask_lock = self.water_mask_lock
            self.water_explorer.depth_lock = self.depth_map_lock
            
            self.get_logger().info("Water explorer initialized")
        except Exception as e:
            self.get_logger().error(f"Failed to initialize water explorer: {str(e)}")
            self.water_explorer = None
        
        # SIMPLIFIED State variables
        self.current_state = BehaviorState.IDLE
        self.current_target = None
        self.current_robot_pose = None
        self.current_path = None  # Main planned path from vision planner
        self.current_updated_path = None  # Incrementally updated path
        self.last_published_path = None
        self.last_published_updated_path = None
        self.current_obstacles = []
        self.obstacle_sizes = []
        
        # SIMPLIFIED timestamps - only what we need
        self.last_target_time = None  # When we last had a target
        self.last_water_mask_time = None  # When we last had a valid water mask
        self.state_entry_time = self.get_clock().now()
        
        # NEW: State persistence tracking
        self.last_state_change_time = self.get_clock().now()
        self.target_lost_time = None  # When we first lost the target (for grace period)
        self.last_valid_target_time = None  # When we last had a truly valid target
        
        # Water exploration state
        self.current_water_mask = None
        self.current_depth_map = None
        
        # Obstacle timestamps
        self.obstacle_timestamp = None
        
        # Target tracking for prediction
        self.target_velocity = np.zeros(3)
        self.previous_target = None
        self.previous_target_timestamp = None
        self.target_positions_history = deque(maxlen=5)
        
        # Path update control
        self.path_planning_in_progress = False
        self.planning_failures_count = 0
        self.max_planning_failures = 3
        self.min_path_update_interval = 0.1
        
        # Enhanced updater-path support with Hermite smoothing
        self.prev_goal_local = None
        self.updater_alpha = 0.55
        self.max_keep_points = 6
        
        # ROS interfaces
        self.setup_ros_interfaces()
        
        # Timer for behavior updates
        self.create_timer(0.1, self.behavior_update)  # 10Hz updates
        
        # Timer for continuous path publishing
        self.create_timer(1.0/self.path_publish_rate, self.publish_path_thread)
        
        self.get_logger().info("Enhanced Behavior System initialized with state persistence")
        self.get_logger().info(f"Target loss grace period: {self.target_loss_grace_period}s")
        self.get_logger().info(f"Minimum state duration: {self.min_state_duration}s")
        self.get_logger().info(f"Rapid switch prevention: {'enabled' if self.rapid_switch_prevention else 'disabled'}")

    def setup_ros_interfaces(self):
        """Setup ROS publishers and subscribers."""
        # Subscribers
        self.create_subscription(
            PoseStamped, '/robot_pose', self.robot_pose_callback, 10)
        self.create_subscription(
            PoseArray, '/object_poses', self.object_callback, 10)
        self.create_subscription(
            PoseArray, '/obstacle_poses', self.obstacle_callback, 10)
        self.create_subscription(
            Float32MultiArray, '/obstacle_sizes', self.obstacle_sizes_callback, 10)
            
        # Publishers
        self.behavior_state_pub = self.create_publisher(
            String, '/behavior/state', 10)
        self.behavior_viz_pub = self.create_publisher(
            MarkerArray, '/behavior/visualization', 10)
        self.path_pub = self.create_publisher(
            Path, '/planned_path', 10)
        self.updated_path_pub = self.create_publisher(
            Path, '/updated_path', 10)
        self.computation_time_pub = self.create_publisher(
            Float32, '/mfs_computation_time', 10)
        self.wavefront_time_pub = self.create_publisher(
            Float32, '/mfs_wavefront_time', 10)
        self.path_time_pub = self.create_publisher(
            Float32, '/mfs_path_time', 10)
        
        # Obstacle visualization publisher - only if debug enabled
        if self.debug_obstacles:
            self.obstacle_viz_pub = self.create_publisher(
                MarkerArray, '/behavior/obstacle_visualization', 10)
            
        # Add wavefront grid visualization publisher
        self.wavefront_grid_pub = self.create_publisher(
            MarkerArray, '/behavior/wavefront_grid', 10)
            
        # Add debug visualizations publisher - only if debug enabled
        if self.debug_obstacles:
            self.debug_viz_pub = self.create_publisher(
                MarkerArray, '/behavior/debug_visualization', 10)
        
        # Add water exploration visualization publishers
        if hasattr(self, 'water_explorer') and self.water_explorer:
            # Create publishers for exploration visualization
            self.exploration_viz_pub = self.create_publisher(
                MarkerArray, '/behavior/exploration_visualization', 10)
            self.heatmap_pub = self.create_publisher(
                Image, '/behavior/exploration_heatmap', 10)
            
            # Set the publisher references in the water explorer
            self.water_explorer.exploration_viz_pub = self.exploration_viz_pub
            self.water_explorer.heatmap_pub = self.heatmap_pub
            self.water_explorer.frame_id = self.frame_id
            
            # Create a timer for regular visualization updates
            self.create_timer(0.5, self.publish_water_explorer_visualization)
            
            self.get_logger().info("Created exploration visualization publishers")
            
        # Connect the visualization publishers to the planner
        self.vision_planner.set_visualization_publisher(self.wavefront_grid_pub)
        if self.debug_obstacles:
            self.vision_planner.set_debug_publisher(self.debug_viz_pub)
            
        # Add water mask and depth map subscribers for exploration
        self.create_subscription(
            Image, '/water_mask', self.water_mask_callback, 10)
        self.create_subscription(
            Image, '/depth', self.depth_map_callback, 10)

    def water_mask_callback(self, msg):
        """Process water mask and pass to the explorer."""
        if hasattr(self, 'water_explorer') and self.water_explorer:
            try:
                # Convert the ROS image to OpenCV format
                if hasattr(msg, "encoding"):
                    if msg.encoding == "mono8":
                        cv_mask = self.bridge.imgmsg_to_cv2(msg, "mono8")
                    elif msg.encoding == "32FC1":
                        float_mask = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                        cv_mask = (float_mask > 0.5).astype(np.uint8) * 255
                    elif msg.encoding == "bgr8" or msg.encoding == "rgb8":
                        color_mask = self.bridge.imgmsg_to_cv2(msg, msg.encoding)
                        cv_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2GRAY)
                    else:
                        cv_mask = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                else:
                    cv_mask = msg
                
                # Normalize if needed
                if np.max(cv_mask) > 1.0:
                    cv_mask = (cv_mask > 127).astype(np.uint8)
                
                # Store current mask
                self.current_water_mask = cv_mask
                
                # Update the mask in water explorer
                has_valid_water = self.water_explorer.update_water_mask(cv_mask)
                
                # SIMPLIFIED: Update water mask timestamp if valid
                if has_valid_water:
                    self.last_water_mask_time = self.get_clock().now()
                    
            except Exception as e:
                self.get_logger().error(f"Error processing water mask: {e}")

    def depth_map_callback(self, msg):
        """Process depth map and pass to the explorer."""
        if hasattr(self, 'water_explorer') and self.water_explorer:
            try:
                if hasattr(msg, "encoding"):
                    if msg.encoding == "16UC1":
                        cv_depth = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                        self.current_depth_map = cv_depth
                        success = self.water_explorer.update_depth_map(cv_depth)
                    elif msg.encoding == "32FC1":
                        depth_m = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                        depth_cm = (depth_m * 100.0).astype(np.uint16)
                        self.current_depth_map = depth_cm
                        success = self.water_explorer.update_depth_map(depth_cm)
                    else:
                        self.get_logger().warn(f"Unsupported depth encoding: {msg.encoding}")
                else:
                    self.water_explorer.update_depth_map(msg)
                    
            except Exception as e:
                self.get_logger().error(f"Error processing depth map: {e}")

    def behavior_update(self):
        """ENHANCED main behavior update loop with state persistence."""
        if not self.current_robot_pose:
            return

        # Check for state transitions with persistence logic
        new_state = self.determine_behavior_state_with_persistence()
        
        if new_state != self.current_state:
            self.get_logger().info(f"State transition: {self.current_state.value} -> {new_state.value}")
            self.current_state = new_state
            self.state_entry_time = self.get_clock().now()
            self.last_state_change_time = self.get_clock().now()
        
        # Update behavior based on current state
        if self.current_state == BehaviorState.TARGET_SEEKING:
            self.handle_target_seeking()
        elif self.current_state == BehaviorState.EXPLORING:
            self.handle_exploring()
        elif self.current_state == BehaviorState.IDLE:
            self.handle_idle()
        
        # Publish current state and visualization
        self.publish_behavior_state()
        self.publish_visualization()
        
        # Visualize obstacles for debugging only if enabled
        if self.debug_obstacles:
            self.visualize_obstacles()

    def determine_behavior_state_with_persistence(self):
        """
        ENHANCED behavior state determination with persistence to prevent rapid switching.
        """
        if not self.rapid_switch_prevention:
            # Fall back to original logic if persistence is disabled
            return self.determine_behavior_state_original()
        
        current_time = self.get_clock().now()
        time_in_current_state = (current_time.nanoseconds - self.last_state_change_time.nanoseconds) / 1e9
        
        # Check if we're still in the minimum state duration
        if time_in_current_state < self.min_state_duration:
            self.get_logger().debug(f"Maintaining {self.current_state.value} state (min duration: {time_in_current_state:.1f}/{self.min_state_duration:.1f}s)")
            return self.current_state
        
        # Check immediate raw conditions (without timeouts)
        has_immediate_target = self.has_immediate_target()
        has_valid_water = self.has_valid_water_mask()
        
        # Handle TARGET_SEEKING state with grace period
        if self.current_state == BehaviorState.TARGET_SEEKING:
            if has_immediate_target:
                # We have a target again, reset lost time and stay in TARGET_SEEKING
                self.target_lost_time = None
                self.last_valid_target_time = current_time
                return BehaviorState.TARGET_SEEKING
            else:
                # No immediate target - start or continue grace period
                if self.target_lost_time is None:
                    self.target_lost_time = current_time
                    self.get_logger().info(f"Target lost, starting grace period ({self.target_loss_grace_period}s)")
                
                # Check if grace period has expired
                grace_elapsed = (current_time.nanoseconds - self.target_lost_time.nanoseconds) / 1e9
                if grace_elapsed < self.target_loss_grace_period:
                    self.get_logger().debug(f"TARGET_SEEKING grace period: {grace_elapsed:.1f}/{self.target_loss_grace_period:.1f}s")
                    return BehaviorState.TARGET_SEEKING
                else:
                    # Grace period expired, allow state transition
                    self.get_logger().info(f"Target grace period expired ({grace_elapsed:.1f}s), allowing state transition")
                    self.target_lost_time = None
                    # Fall through to normal priority logic
        else:
            # Not in TARGET_SEEKING, reset target lost time
            self.target_lost_time = None
        
        # Normal priority-based logic (but immediate checks, not timeout-based)
        if has_immediate_target:
            return BehaviorState.TARGET_SEEKING
        elif has_valid_water:
            return BehaviorState.EXPLORING
        else:
            return BehaviorState.IDLE

    def determine_behavior_state_original(self):
        """
        Original behavior state determination (for fallback).
        """
        # Priority 1: TARGET_SEEKING - Do we have a target?
        if self.has_valid_target():
            return BehaviorState.TARGET_SEEKING
        
        # Priority 2: EXPLORING - Do we have water but no target?
        if self.has_valid_water_mask():
            return BehaviorState.EXPLORING
        
        # Priority 3: IDLE - Default state
        return BehaviorState.IDLE

    def has_immediate_target(self):
        """Check if we have an immediate target (no timeout check)."""
        return self.current_target is not None

    def has_valid_target(self):
        """Check if we have a valid target for seeking (with timeout)."""
        if self.current_target is None:
            return False
        
        # Check if target is too old
        if self.last_target_time is None:
            return False
            
        current_time = self.get_clock().now()
        target_age = (current_time.nanoseconds - self.last_target_time.nanoseconds) / 1e9
        
        return target_age < self.target_timeout

    def has_valid_water_mask(self):
        """Check if we have a valid water mask for exploration."""
        if self.last_water_mask_time is None:
            return False
            
        current_time = self.get_clock().now()
        water_age = (current_time.nanoseconds - self.last_water_mask_time.nanoseconds) / 1e9
        
        return water_age < self.water_mask_timeout

    def handle_idle(self):
        """Handle idle state behavior - clear all paths and wait."""
        with self.path_lock:
            self.current_path = None
            self.current_updated_path = None
        
        self.get_logger().debug("IDLE: Waiting for targets or water mask")

    def handle_exploring(self):
        """Handle water exploration behavior."""
        if not hasattr(self, 'water_explorer') or self.water_explorer is None:
            self.get_logger().warn("No water explorer available for exploration")
            return
            
        try:
            # Check if water mask exists
            has_water_mask = False
            with self.water_explorer.mask_lock:
                has_water_mask = self.water_explorer.water_mask is not None
                
            if not has_water_mask:
                self.get_logger().warn("No water mask available, skipping exploration")
                return
                
            # Get exploration goal from the water explorer
            current_time = time.time()
            goal_pose = self.water_explorer.get_exploration_goal(
                robot_pose=self.current_robot_pose.pose, 
                current_time=current_time
            )
            
            if goal_pose is None:
                self.get_logger().warn("Water explorer returned no goal")
                return
                
            # Create target from exploration goal
            target = Pose()
            target.position.x = goal_pose.position.x
            target.position.y = goal_pose.position.y
            target.position.z = 0.0
            
            # Check if this is a significantly different target
            is_new_target = True
            if self.current_target:
                dx = abs(target.position.x - self.current_target.position.x)
                dy = abs(target.position.y - self.current_target.position.y)
                if dx < 0.2 and dy < 0.2:  # Within 20cm
                    is_new_target = False
                    
            # Only update path if it's a new target
            if is_new_target:
                with self.target_lock:
                    self.current_target = target
                    self.last_target_time = self.get_clock().now()
                    
                self.get_logger().info(f"EXPLORING: New exploration target at ({target.position.x:.2f}, {target.position.y:.2f})")
                
                # Plan path to exploration target
                self.handle_target_seeking()
            
        except Exception as e:
            self.get_logger().error(f"Error in exploration behavior: {str(e)}")

    def handle_target_seeking(self):
        """Handle target seeking behavior with simplified logic."""
        if self.current_target is None or self.current_robot_pose is None:
            return

        # Check obstacle synchronization
        if not self.synchronize_obstacle_data():
            self.get_logger().warn("Target-seeking skipped (stale obstacle data)")
            return

        # Get target pose
        with self.target_lock:
            target_pose = deepcopy(self.current_target)

        # Predict future target position if enabled
        predicted = self.predict_target_position(target_pose)
        if predicted is not None:
            target_pose = predicted

        # Convert to local coordinates
        start_local = np.array([self.robot_length / 2.0, 0.0, 0.0])
        
        goal_local = np.array([
            target_pose.position.x,
            target_pose.position.y,
            target_pose.position.z
        ])

        # Transform goal to world frame for path blending
        goal_world = np.array(
            self.transform_to_world(
                goal_local[0], goal_local[1], goal_local[2],
                self.current_robot_pose.pose
            )
        )

        # If planning is in progress, just update existing path
        if self.path_planning_in_progress:
            with self.path_lock:
                if self.current_path:
                    # Update the incremental path, not the main path
                    self.current_updated_path = self._generate_updater_path(
                        self.current_path, goal_world
                    )
                    self.get_logger().debug("TARGET_SEEKING: Updating path incrementally (planning in progress)")
            return

        # Check minimum interval between replans
        now = self.get_clock().now()
        if hasattr(self, 'last_path_update_time') and self.last_path_update_time:
            dt = (now.nanoseconds - self.last_path_update_time.nanoseconds) / 1e9
            if dt < self.min_path_update_interval:
                with self.path_lock:
                    if self.current_path:
                        # Update the incremental path during rate limiting
                        self.current_updated_path = self._generate_updater_path(
                            self.current_path, goal_world
                        )
                        self.get_logger().debug("TARGET_SEEKING: Updating path incrementally (rate limited)")
                return

        # Gather obstacle data
        with self.obstacle_lock:
            obstacle_positions = [
                np.array([o.position.x, o.position.y, o.position.z])
                for o in self.current_obstacles
            ]
            obstacle_sizes = list(self.obstacle_sizes)

        # Start async planning
        self.path_planning_in_progress = True
        self.get_logger().debug("TARGET_SEEKING: Starting new path planning")
        threading.Thread(
            target=self._compute_path_async,
            args=(start_local, goal_local, obstacle_positions, obstacle_sizes),
            daemon=True
        ).start()

        # Update existing path while planning
        with self.path_lock:
            if self.current_path:
                # Generate updated path while waiting for new plan
                self.current_updated_path = self._generate_updater_path(
                    self.current_path, goal_world
                )
                self.get_logger().debug("TARGET_SEEKING: Updating path incrementally (while planning)")

    def object_callback(self, msg: PoseArray):
        """ENHANCED object detection callback with state persistence awareness."""
        try:
            with self.target_lock:
                current_time = self.get_clock().now()
                
                # Check for empty poses (no objects detected)
                if len(msg.poses) == 0:
                    self.get_logger().debug("No objects detected - clearing target")
                    self.current_target = None
                    self.last_target_time = None
                    # Note: We don't immediately clear target_lost_time here - let the state machine handle it
                    return
                
                # Find closest object
                closest_obj = None
                min_dist = float('inf')

                for pose in msg.poses:
                    # Skip NaN poses (indicates no detection)
                    if (math.isnan(pose.position.x) or 
                        math.isnan(pose.position.y) or 
                        math.isnan(pose.position.z)):
                        continue
                        
                    # Calculate distance
                    dist = math.sqrt(pose.position.x**2 + pose.position.y**2 + pose.position.z**2)
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_obj = deepcopy(pose)

                # Update target
                if closest_obj:
                    self.current_target = closest_obj
                    self.last_target_time = current_time
                    
                    # Reset target lost time since we have a target again
                    self.target_lost_time = None
                    self.last_valid_target_time = current_time
                    
                    # Update target history for velocity tracking
                    timestamp = current_time.nanoseconds / 1e9
                    self.target_positions_history.append((deepcopy(closest_obj), timestamp))
                    
                    self.get_logger().debug(f"TARGET: Object detected at distance {min_dist:.2f}m")
                else:
                    self.current_target = None
                    self.last_target_time = None
                    self.get_logger().debug("TARGET: No valid objects (all NaN)")

        except Exception as e:
            self.get_logger().error(f'Error in object callback: {str(e)}')

    def obstacle_callback(self, msg: PoseArray):
        """Handle obstacle updates with NaN detection."""
        try:
            with self.obstacle_lock:
                current_time = self.get_clock().now()
                
                # Check if we received valid obstacles or NaN
                valid_obstacles = []
                if len(msg.poses) > 0:
                    for pose in msg.poses:
                        if not self.is_nan_pose(pose):
                            valid_obstacles.append(deepcopy(pose))
                
                # Store valid obstacles
                self.current_obstacles = valid_obstacles
                
                # Store timestamp for synchronization
                self.obstacle_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec/1e9
                    
        except Exception as e:
            self.get_logger().error(f'Error in obstacle callback: {str(e)}')

    def obstacle_sizes_callback(self, msg: Float32MultiArray):
        """Handle obstacle size updates."""
        try:
            with self.obstacle_lock:
                self.obstacle_sizes = list(msg.data)
        except Exception as e:
            self.get_logger().error(f'Error in obstacle sizes callback: {str(e)}')

    def robot_pose_callback(self, msg: PoseStamped):
        """Handle robot pose updates."""
        self.current_robot_pose = deepcopy(msg)

    def publish_water_explorer_visualization(self):
        """Call the water explorer's visualization methods."""
        if hasattr(self, 'water_explorer') and self.water_explorer:
            try:
                self.water_explorer.publish_exploration_visualization()
            except Exception as e:
                self.get_logger().error(f"Error publishing water explorer visualization: {e}")

    def publish_behavior_state(self):
        """Publish current behavior state with additional debug info."""
        msg = String()
        
        # Add debug info about state persistence if enabled
        if self.rapid_switch_prevention:
            current_time = self.get_clock().now()
            time_in_state = (current_time.nanoseconds - self.last_state_change_time.nanoseconds) / 1e9
            
            state_info = self.current_state.value
            
            # Add grace period info if applicable
            if self.current_state == BehaviorState.TARGET_SEEKING and self.target_lost_time is not None:
                grace_elapsed = (current_time.nanoseconds - self.target_lost_time.nanoseconds) / 1e9
                state_info += f" (grace: {grace_elapsed:.1f}/{self.target_loss_grace_period:.1f}s)"
            elif time_in_state < self.min_state_duration:
                state_info += f" (min: {time_in_state:.1f}/{self.min_state_duration:.1f}s)"
            
            msg.data = state_info
        else:
            msg.data = self.current_state.value
            
        self.behavior_state_pub.publish(msg)

    def publish_visualization(self):
        """Publish visualization markers."""
        marker_array = MarkerArray()
        
        # State text marker
        text_marker = Marker()
        text_marker.header.frame_id = self.frame_id
        text_marker.header.stamp = self.get_clock().now().to_msg()
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.id = 0
        
        if self.current_robot_pose:
            text_marker.pose.position = self.current_robot_pose.pose.position
            text_marker.pose.position.z += 1.0
            
        text_marker.scale.z = 0.5
        text_marker.color.a = 1.0
        
        # Color based on state
        if self.current_state == BehaviorState.IDLE:
            text_marker.color.r = 0.5
            text_marker.color.g = 0.5
            text_marker.color.b = 0.5
        elif self.current_state == BehaviorState.EXPLORING:
            text_marker.color.g = 1.0
        elif self.current_state == BehaviorState.TARGET_SEEKING:
            text_marker.color.b = 1.0
            
        # Enhanced text with persistence info
        text_content = self.current_state.value
        if self.rapid_switch_prevention:
            current_time = self.get_clock().now()
            if self.current_state == BehaviorState.TARGET_SEEKING and self.target_lost_time is not None:
                grace_elapsed = (current_time.nanoseconds - self.target_lost_time.nanoseconds) / 1e9
                text_content += f"\nGrace: {grace_elapsed:.1f}s"
                text_marker.color.r = 0.5  # Add some red to indicate grace period
        
        text_marker.text = text_content
        marker_array.markers.append(text_marker)
        
        # Target visualization
        if self.current_target and self.current_robot_pose:
            world_x, world_y, world_z = self.transform_to_world(
                self.current_target.position.x,
                self.current_target.position.y,
                self.current_target.position.z,
                self.current_robot_pose.pose
            )
            
            target_marker = Marker()
            target_marker.header.frame_id = self.frame_id
            target_marker.header.stamp = self.get_clock().now().to_msg()
            target_marker.type = Marker.SPHERE
            target_marker.action = Marker.ADD
            target_marker.id = 1
            
            target_marker.pose.position.x = world_x
            target_marker.pose.position.y = world_y
            target_marker.pose.position.z = world_z
            target_marker.pose.orientation.w = 1.0
            
            target_marker.scale.x = 0.3
            target_marker.scale.y = 0.3
            target_marker.scale.z = 0.3
            target_marker.color.a = 1.0
            target_marker.color.r = 0.0
            target_marker.color.g = 0.0
            target_marker.color.b = 1.0
            marker_array.markers.append(target_marker)
        
        self.behavior_viz_pub.publish(marker_array)

    def publish_path_thread(self):
        """
        Thread function to publish both main planned paths and updated paths.
        - /planned_path: Main paths from the vision planner
        - /updated_path: Incrementally updated paths during planning
        """
        if not self.current_robot_pose:
            return
            
        current_time = self.get_clock().now().to_msg()
        
        # Publish main planned path
        with self.path_lock:
            main_path = self.current_path
            updated_path = self.current_updated_path
        
        # Create and publish main planned path
        if main_path is not None:
            planned_path_msg = self._create_path_message(main_path, current_time, "Main Planned Path")
            self.path_pub.publish(planned_path_msg)
            self.last_published_path = planned_path_msg
        elif self.last_published_path is not None:
            # Republish last path with updated timestamp
            self.last_published_path.header.stamp = current_time
            self.path_pub.publish(self.last_published_path)
        else:
            # Publish empty path
            empty_path = Path()
            empty_path.header.frame_id = self.frame_id
            empty_path.header.stamp = current_time
            self.path_pub.publish(empty_path)
        
        # Create and publish updated path (if available)
        if updated_path is not None:
            updated_path_msg = self._create_path_message(updated_path, current_time, "Updated Path")
            self.updated_path_pub.publish(updated_path_msg)
            self.last_published_updated_path = updated_path_msg
            self.get_logger().debug(f"Published updated path with {len(updated_path)} points")
        elif self.last_published_updated_path is not None:
            # Republish last updated path with updated timestamp
            self.last_published_updated_path.header.stamp = current_time
            self.updated_path_pub.publish(self.last_published_updated_path)
        else:
            # Publish empty updated path
            empty_updated_path = Path()
            empty_updated_path.header.frame_id = self.frame_id
            empty_updated_path.header.stamp = current_time
            self.updated_path_pub.publish(empty_updated_path)

    def _create_path_message(self, path_points, timestamp, path_type):
        """Helper method to create a Path message from path points."""
        path_msg = Path()
        path_msg.header.frame_id = self.frame_id
        path_msg.header.stamp = timestamp
        
        for point in path_points:
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(point[2])
            
            # Set orientation to look along path direction
            if len(path_msg.poses) > 0:
                prev_pos = path_msg.poses[-1].pose.position
                dx = pose.pose.position.x - prev_pos.x
                dy = pose.pose.position.y - prev_pos.y
                
                if dx != 0 or dy != 0:
                    theta = math.atan2(dy, dx)
                    q = self.quaternion_from_euler(0, 0, theta)
                    pose.pose.orientation = q
                else:
                    pose.pose.orientation.w = 1.0
            else:
                pose.pose.orientation.w = 1.0
            
            path_msg.poses.append(pose)
        
        return path_msg

    # [Include all the helper methods from the original code]
    def transform_to_world(self, local_x, local_y, local_z, robot_pose):
        """Transform a point from robot's local frame to world frame."""
        robot_x = robot_pose.position.x
        robot_y = robot_pose.position.y
        robot_z = robot_pose.position.z
        robot_yaw = self.get_robot_yaw(robot_pose)
        
        cos_yaw = math.cos(robot_yaw)
        sin_yaw = math.sin(robot_yaw)
        x_rot = local_x * cos_yaw - local_y * sin_yaw
        y_rot = local_x * sin_yaw + local_y * cos_yaw
        
        world_x = robot_x + x_rot
        world_y = robot_y + y_rot
        world_z = robot_z + local_z
        
        return world_x, world_y, world_z

    def is_nan_pose(self, pose):
        """Check if a pose contains NaN values."""
        return (math.isnan(pose.position.x) or 
                math.isnan(pose.position.y) or 
                math.isnan(pose.position.z) or
                math.isnan(pose.orientation.w))

    def synchronize_obstacle_data(self):
        """Check if obstacle data is synchronized with robot pose."""
        if not self.current_robot_pose:
            return False

        if not self.current_obstacles:
            return True

        try:
            robot_time = (
                self.current_robot_pose.header.stamp.sec
                + self.current_robot_pose.header.stamp.nanosec / 1e9
            )

            if self.obstacle_timestamp is None:
                return True

            time_diff = abs(robot_time - self.obstacle_timestamp)
            return (time_diff <= 0.3)

        except Exception as e:
            self.get_logger().error(f"Error checking data synchronization: {str(e)}")
            return True

    def predict_target_position(self, target_pose):
        """Predict future target position if enabled."""
        if not self.use_prediction or len(self.target_positions_history) < 2:
            return None
            
        try:
            avg_velocity = np.zeros(3)
            positions = []
            timestamps = []
            
            for pos, timestamp in self.target_positions_history:
                positions.append(np.array([pos.position.x, pos.position.y, pos.position.z]))
                timestamps.append(timestamp)
            
            velocities = []
            for i in range(1, len(positions)):
                delta_pos = positions[i] - positions[i-1]
                delta_time = timestamps[i] - timestamps[i-1]
                if delta_time > 0:
                    velocity = delta_pos / delta_time
                    velocities.append(velocity)
            
            if velocities:
                avg_velocity = sum(velocities) / len(velocities)
                
                current_time = self.get_clock().now().nanoseconds / 1e9
                last_timestamp = timestamps[-1]
                time_delta = current_time - last_timestamp
                
                velocity_magnitude = np.linalg.norm(avg_velocity)
                if time_delta > 0 and time_delta < 1.0 and velocity_magnitude > 0.1:
                    prediction_time = min(time_delta, self.prediction_horizon)
                    prediction_offset = avg_velocity * prediction_time
                    
                    predicted_pose = deepcopy(target_pose)
                    predicted_pose.position.x += prediction_offset[0]
                    predicted_pose.position.y += prediction_offset[1]
                    predicted_pose.position.z += prediction_offset[2]
                    
                    return predicted_pose
        except Exception as e:
            self.get_logger().error(f"Error in target prediction: {str(e)}")
        
        return None

    def visualize_obstacles(self):
        """Visualize obstacles for debugging."""
        if not hasattr(self, 'obstacle_viz_pub'):
            return
            
        marker_array = MarkerArray()
        
        with self.obstacle_lock:
            obstacles = self.current_obstacles.copy()
            sizes = self.obstacle_sizes.copy()
            
            for i, obstacle in enumerate(obstacles):
                obstacle_size = 0.3
                if i < len(sizes):
                    obstacle_size = max(0.3, sizes[i])
                
                world_x, world_y, world_z = self.transform_to_world(
                    obstacle.position.x,
                    obstacle.position.y,
                    obstacle.position.z,
                    self.current_robot_pose.pose
                )
                    
                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = "obstacle_debugging"
                marker.id = i
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                marker.pose.position.x = world_x
                marker.pose.position.y = world_y
                marker.pose.position.z = world_z
                marker.pose.orientation.w = 1.0
                
                marker.scale.x = obstacle_size
                marker.scale.y = obstacle_size
                marker.scale.z = 0.4
                marker.color.a = 0.7
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
                
                marker_array.markers.append(marker)
        
        if marker_array.markers:
            self.obstacle_viz_pub.publish(marker_array)

    @staticmethod
    def get_robot_yaw(pose):
        """Extract yaw from quaternion."""
        siny_cosp = 2.0 * (pose.orientation.w * pose.orientation.z + 
                          pose.orientation.x * pose.orientation.y)
        cosy_cosp = 1.0 - 2.0 * (pose.orientation.y**2 + pose.orientation.z**2)
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def quaternion_from_euler(roll, pitch, yaw):
        """Convert Euler angles to quaternion."""
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        q = Quaternion()
        q.w = cy * cp * cr + sy * sp * sr
        q.x = cy * cp * sr - sy * sp * cr
        q.y = sy * cp * sr + cy * sp * cr
        q.z = sy * cp * cr - cy * sp * sr
        return q

    # Include remaining path planning methods from original code...
    def _apply_hermite_smoothing_to_initial_path(self, raw_path):
        """Apply Hermite spline smoothing to initial path."""
        if not raw_path or len(raw_path) < 3:
            return raw_path
        
        # Keep the original implementation
        path_points = [np.array(point) for point in raw_path]
        tangents = self._calculate_hermite_tangents(path_points)
        
        smooth_path = []
        
        for i in range(len(path_points) - 1):
            p0 = path_points[i][:2]
            p1 = path_points[i + 1][:2]
            t0 = tangents[i]
            t1 = tangents[i + 1]
            
            segment_length = np.linalg.norm(p1 - p0)
            num_points = max(2, int(segment_length / 0.2))
            
            for j in range(num_points):
                t = j / (num_points - 1) if num_points > 1 else 0
                
                h00 = 2*t**3 - 3*t**2 + 1
                h10 = t**3 - 2*t**2 + t
                h01 = -2*t**3 + 3*t**2
                h11 = t**3 - t**2
                
                smooth_point_2d = (h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1)
                
                z_interp = path_points[i][2] + t * (path_points[i + 1][2] - path_points[i][2])
                smooth_point = np.array([smooth_point_2d[0], smooth_point_2d[1], z_interp])
                
                if not smooth_path or np.linalg.norm(smooth_point - smooth_path[-1]) > 0.05:
                    smooth_path.append(smooth_point)
        
        if smooth_path and np.linalg.norm(smooth_path[-1] - path_points[-1]) > 0.05:
            smooth_path.append(path_points[-1])
        
        return smooth_path

    def _calculate_hermite_tangents(self, path_points):
        """Calculate tangent vectors for Hermite interpolation."""
        tangents = []
        
        for i in range(len(path_points)):
            if i == 0:
                if len(path_points) > 1:
                    tangent = (path_points[1][:2] - path_points[0][:2]) * 0.5
                else:
                    tangent = np.zeros(2)
            elif i == len(path_points) - 1:
                tangent = (path_points[i][:2] - path_points[i-1][:2]) * 0.5
            else:
                prev_segment = path_points[i][:2] - path_points[i-1][:2]
                next_segment = path_points[i+1][:2] - path_points[i][:2]
                
                tangent = (next_segment + prev_segment) * 0.5
                
                segment_length = (np.linalg.norm(prev_segment) + np.linalg.norm(next_segment)) / 2
                if segment_length > 0:
                    tangent_length = np.linalg.norm(tangent)
                    if tangent_length > 0:
                        max_tangent_length = segment_length * 0.6
                        if tangent_length > max_tangent_length:
                            tangent = tangent * (max_tangent_length / tangent_length)
            
            tangents.append(tangent)
        
        return tangents

    def _generate_updater_path(self, old_path, new_goal_world):
        """Enhanced path updater for smooth Hermite paths."""
        if not old_path or len(old_path) < 2:
            return old_path

        updated = [p.copy() for p in old_path]
        keep = min(self.max_keep_points, len(updated) - 1)
        
        if keep >= len(updated) - 1:
            updated[-1] = new_goal_world.copy()
            return updated

        old_goal = updated[-1]
        goal_change = np.linalg.norm(new_goal_world - old_goal)
        
        if goal_change < 0.3:
            updated = self._apply_gentle_endpoint_adjustment(updated, keep, new_goal_world)
        elif goal_change < 1.0:
            updated = self._apply_smooth_curve_modification(updated, keep, new_goal_world)
        else:
            updated = self._apply_partial_path_regeneration(updated, keep, new_goal_world)
        
        updated = self._apply_minimal_smoothing(updated, keep)
        updated[-1] = new_goal_world.copy()
        
        return updated

    def _apply_gentle_endpoint_adjustment(self, updated, keep, new_goal_world):
        """Apply gentle adjustment for small goal changes."""
        if len(updated) - keep < 2:
            return updated
        
        old_goal = updated[-1]
        goal_shift = new_goal_world - old_goal
        
        for i in range(len(updated) - 1, keep, -1):
            distance_from_goal = (len(updated) - 1 - i) / (len(updated) - 1 - keep)
            adjustment_factor = math.exp(-distance_from_goal * 3.0)
            adjustment = goal_shift * adjustment_factor * 0.7
            updated[i] += adjustment
        
        return updated

    def _apply_smooth_curve_modification(self, updated, keep, new_goal_world):
        """Apply smooth curve modification for medium goal changes."""
        if len(updated) - keep < 3:
            return updated
        
        goal_shift = new_goal_world - updated[-1]
        update_length = len(updated) - keep
        
        for i in range(keep, len(updated)):
            t = (i - keep) / (update_length - 1) if update_length > 1 else 1.0
            smooth_t = 3 * t**2 - 2 * t**3
            
            if i > keep:
                current_direction = updated[i] - updated[i-1]
                current_direction_norm = np.linalg.norm(current_direction)
                
                if current_direction_norm > 0.01:
                    velocity_preservation = 0.7
                    goal_attraction = 1.0 - velocity_preservation
                    
                    target_direction = new_goal_world - updated[i]
                    target_direction_norm = np.linalg.norm(target_direction)
                    
                    if target_direction_norm > 0.01:
                        current_dir_unit = current_direction / current_direction_norm
                        target_dir_unit = target_direction / target_direction_norm
                        
                        blended_direction = (velocity_preservation * current_dir_unit + 
                                           goal_attraction * target_dir_unit)
                        blended_direction_norm = np.linalg.norm(blended_direction)
                        
                        if blended_direction_norm > 0:
                            blended_direction /= blended_direction_norm
                            
                            adjustment_magnitude = smooth_t * np.linalg.norm(goal_shift) * 0.3
                            adjustment = blended_direction[:len(updated[i])] * adjustment_magnitude
                            updated[i] += adjustment
            else:
                adjustment = goal_shift * smooth_t * 0.2
                updated[i] += adjustment
        
        return updated

    def _apply_partial_path_regeneration(self, updated, keep, new_goal_world):
        """Apply partial path regeneration for large goal changes."""
        if len(updated) - keep < 2:
            return updated
        
        start_pt = updated[keep]
        
        if keep > 0:
            start_tangent = (updated[keep] - updated[keep-1])[:2] * 0.5
        else:
            start_tangent = np.zeros(2)
        
        goal_direction = new_goal_world[:2] - start_pt[:2]
        goal_distance = np.linalg.norm(goal_direction)
        
        if goal_distance > 0.1:
            end_tangent = goal_direction / goal_distance * goal_distance * 0.3
        else:
            end_tangent = np.zeros(2)
        
        update_length = len(updated) - keep
        
        for i in range(keep, len(updated)):
            t = (i - keep) / (update_length - 1) if update_length > 1 else 1.0
            
            h00 = 2*t**3 - 3*t**2 + 1
            h10 = t**3 - 2*t**2 + t
            h01 = -2*t**3 + 3*t**2
            h11 = t**3 - t**2
            
            new_pos_2d = (h00 * start_pt[:2] + 
                          h10 * start_tangent + 
                          h01 * new_goal_world[:2] + 
                          h11 * end_tangent)
            
            new_z = start_pt[2] + t * (new_goal_world[2] - start_pt[2])
            
            blend_factor = self.updater_alpha * (0.5 + 0.5 * t)
            
            original_pos = updated[i]
            new_pos = np.array([new_pos_2d[0], new_pos_2d[1], new_z])
            
            updated[i] = (1.0 - blend_factor) * original_pos + blend_factor * new_pos
        
        return updated

    def _apply_minimal_smoothing(self, updated, keep):
        """Apply minimal smoothing to maintain path quality."""
        if len(updated) - keep < 3:
            return updated
        
        smoothed = updated.copy()
        
        for i in range(keep + 1, len(updated) - 1):
            prev_point = smoothed[i - 1]
            curr_point = smoothed[i]
            next_point = smoothed[i + 1]
            
            smoothed_point = 0.2 * prev_point + 0.6 * curr_point + 0.2 * next_point
            
            change_magnitude = np.linalg.norm(smoothed_point - curr_point)
            if change_magnitude < 0.1:
                smoothed[i] = smoothed_point
        
        return smoothed

    def _compute_path_async(self, start_pos, goal_pos, obstacle_positions, obstacle_sizes):
        """Enhanced async path computation with Hermite smoothing."""
        try:
            self.get_logger().debug(f"Planning path from {start_pos} to {goal_pos} with {len(obstacle_positions)} obstacles")
            
            planning_start = time.time()
            
            local_path = self.vision_planner.plan_path(
                start_pos,
                goal_pos,
                obstacle_positions,
                obstacle_sizes,
                object_positions=None,
                robot_pose=self.current_robot_pose.pose
            )
            
            planning_end = time.time()
            planning_duration = planning_end - planning_start
            
            # Publish overall computation time
            actual_time = self.vision_planner.last_planning_time if self.vision_planner.last_planning_time is not None else planning_duration
            time_msg = Float32()
            time_msg.data = float(actual_time)
            self.computation_time_pub.publish(time_msg)
            
            # Publish wavefront computation time
            if self.vision_planner.last_wavefront_time is not None:
                wavefront_msg = Float32()
                wavefront_msg.data = float(self.vision_planner.last_wavefront_time)
                self.wavefront_time_pub.publish(wavefront_msg)
            
            # Publish path computation time
            if self.vision_planner.last_path_computation_time is not None:
                path_msg = Float32()
                path_msg.data = float(self.vision_planner.last_path_computation_time)
                self.path_time_pub.publish(path_msg)
            
            self.get_logger().info(f"Published MFS timings - Overall: {actual_time*1000:.2f} ms, Wavefront: {self.vision_planner.last_wavefront_time*1000:.2f} ms, Path: {self.vision_planner.last_path_computation_time*1000:.2f} ms")
            
            if not local_path:
                self.get_logger().warn("Asynchronous planning returned no path")
                return
                
            world_path = []
            
            for point in local_path:
                world_x, world_y, world_z = self.transform_to_world(
                    point[0], point[1], point[2],
                    self.current_robot_pose.pose
                )
                world_path.append(np.array([world_x, world_y, world_z]))
            
            self.get_logger().debug("Applying Hermite smoothing to initial path")
            smooth_path = self._apply_hermite_smoothing_to_initial_path(world_path)
            
            with self.path_lock:
                self.current_path = smooth_path
                self.current_updated_path = None
                if not hasattr(self, 'last_path_update_time'):
                    self.last_path_update_time = None
                self.last_path_update_time = self.get_clock().now()
                
            self.planning_failures_count = 0
            self.get_logger().info(f"New planned path with {len(smooth_path)} points")
            self.get_logger().debug(f"Smoothed path with {len(smooth_path)} points")
            
        except Exception as e:
            self.get_logger().error(f"Error in async path planning: {str(e)}")
            import traceback
            traceback.print_exc()
            self.planning_failures_count += 1
        finally:
            self.path_planning_in_progress = False



def main(args=None):
    rclpy.init(args=args)
    node = BehaviorSystem()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()