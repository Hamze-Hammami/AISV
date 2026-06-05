#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist, PoseArray, Pose
from nav_msgs.msg import Path
from sensor_msgs.msg import Range, Image
from std_msgs.msg import String, Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
import enum
import math
import time
import traceback

from controller.fuzzy_controller import FuzzyController
from controller.dwa_controller import DWAController

class ControlState(enum.Enum):
    IDLE = "idle"
    FUZZY_CONTROL = "fuzzy_control"
    DWA_CONTROL = "dwa_control"
    EMERGENCY_STOP = "emergency_stop"
    TURN_IN_PLACE = "turn_in_place"  # Added dedicated state for turn in place

class ControlSystem(Node):
    def __init__(self):
        super().__init__('control_system')
        
        # Parameters initialization
        self.init_ros_parameters()
        self.get_ros_parameters()

        # Initialize controllers
        self.fuzzy_controller = FuzzyController(
            max_linear_speed=self.max_linear_speed,
            max_angular_speed=self.max_angular_speed,
            goal_tolerance=self.goal_tolerance
        )
        
        self.dwa_controller = DWAController(
            max_speed=self.max_speed,
            min_speed=self.min_speed,
            max_yaw_rate=self.max_yaw_rate,
            max_accel=self.max_accel,
            velocity_resolution=self.velocity_resolution,
            predict_time=self.predict_time
        )

        # State initialization
        self.current_state = ControlState.IDLE
        self.previous_state = None
        self.current_pose = None
        self.current_path = []
        self.current_path_index = 0
        self.current_goal = None
        self.sonar_data = {}
        self.detected_objects = []
        self.detected_obstacles = []
        self.last_control_time = self.get_clock().now()
        self.last_obstacle_check_time = self.get_clock().now()
        
        # Front depth tracking (water boundary)
        self.front_depth_value = None
        self.front_depth_timestamp = None
        self.front_depth_timeout = 2.0
        
        # Water mask status tracking
        self.water_mask_status = "normal"  # "normal", "small", "multiple", "lost"
        self.water_mask_timestamp = None
        self.water_mask_status_timeout = 2.0
        
        # Path completion tracking
        self.path_completed = False
        self.goal_reached_time = None
        self.goal_reached_duration = 5.0
        
        # Obstacle tracking - MODIFIED: Separate sonar and critical obstacle tracking
        self.obstacle_check_counter = 0
        self.sonar_obstacle_detected_count = 0  # For DWA switching only
        self.sonar_obstacle_free_count = 0     # For DWA switching only
        self.critical_obstacle_detected_count = 0  # For emergency behaviors only
        self.critical_obstacle_free_count = 0      # For emergency behaviors only
        
        # Robot velocity for adaptive lookahead
        self.current_linear_velocity = 0.0
        self.current_angular_velocity = 0.0
        self.last_cmd_time = None
        self.last_cmd_vel = None
        
        # Emergency stop handling - ENHANCED (only for non-sonar triggers)
        self.emergency_stop_active = False
        self.emergency_stop_time = None
        self.emergency_recovery_duration = 4.0
        self.emergency_turn_direction = None
        self.consecutive_emergency_count = 0
        self.max_consecutive_emergencies = 5
        
        # Path tracking statistics
        self.cross_track_error_avg = 0.0
        self.cross_track_error_samples = 0
        
        # Turn-in-place handling - ENHANCED (only for non-sonar triggers)
        self.turn_in_place_active = False
        self.turn_in_place_speed = 2.0
        self.emergency_turn_speed = 2.5
        self.turning_in_progress = False
        self.turning_start_time = None
        self.turn_lock_duration = 4.0
        self.min_turn_duration = 1.0
        self.turn_direction_history = []
        self.turn_direction_history_size = 2
        self.last_emergency_direction = None
        self.turn_target_heading = None
        self.turn_completed = False
        self.force_turn_completion = False
        self.new_path_during_turn = False
        
        # Servo handling
        self.servo_state = 0  # 0: idle, 1: collecting, 2: retracting
        self.servo_state_time = None
        
        # Setup ROS interfaces
        self.setup_ros_interfaces()

        # Timer for control loop
        self.create_timer(self.control_period, self.control_loop_callback)

        # Timer for statistics
        self.create_timer(5.0, self.print_statistics)

        self.get_logger().info('Control System initialized with sonar-only DWA mode (no emergency triggers)')
        self.get_logger().info('IMPORTANT: Sonars trigger DWA switching ONLY - NO emergency stops')
        self.get_logger().info('Emergency triggers: front_depth < 0.3m, water_mask issues ONLY')
        self.get_logger().info(f'Controller emergency filtering: {"ENABLED" if self.filter_controller_emergency_commands else "DISABLED"}')
        self.get_logger().info('This will filter out emergency commands from fuzzy/DWA controllers for sonar-only obstacles')

    def init_ros_parameters(self):
        """Initialize all ROS parameters."""
        self.declare_parameter('goal_tolerance', 0.1)
        self.declare_parameter('lookahead_distance_min', 0.3)
        self.declare_parameter('lookahead_distance_max', 1.5)
        self.declare_parameter('control_frequency', 20.0)
        self.declare_parameter('state_change_timeout', 1.0)
        self.declare_parameter('path_following_enabled', True)
        self.declare_parameter('path_lookahead_points', 3)
        self.declare_parameter('goal_reached_duration', 5.0)
        self.declare_parameter('adaptive_lookahead', True)
        self.declare_parameter('obstacle_hysteresis', 5)
        self.declare_parameter('control_period', 0.05)
        
        self.declare_parameter('max_linear_speed', 3.0)
        self.declare_parameter('max_angular_speed', 2.5)
        
        self.declare_parameter('max_speed', 20.0)
        self.declare_parameter('min_speed', 10.0)
        self.declare_parameter('max_yaw_rate', 11.0)
        self.declare_parameter('max_accel', 20.0)
        self.declare_parameter('velocity_resolution', 0.2)
        self.declare_parameter('predict_time', 1.0)
        
        self.declare_parameter('sonar_threshold', 0.2)
        self.declare_parameter('obstacle_threshold', 0.5)
        self.declare_parameter('obstacle_check_frequency', 2)
        self.declare_parameter('emergency_stop_threshold', 0.5)  # NOT USED FOR SONARS
        self.declare_parameter('use_vision_obstacles', True)
        
        # MODIFIED: Front depth parameters - these WILL trigger emergency behaviors
        self.declare_parameter('front_depth_threshold', 1.0)
        self.declare_parameter('front_depth_timeout', 2.0)
        self.declare_parameter('use_front_depth', True)
        self.declare_parameter('front_depth_emergency_threshold', 0.3)  # NEW: Critical threshold for emergency
        
        # Enhanced parameters for faster turning (only for non-sonar emergencies)
        self.declare_parameter('turn_in_place_speed', 10.0)
        self.declare_parameter('turn_lock_duration', 4.0)
        self.declare_parameter('emergency_turn_speed', 2.5)
        
        # Parameters for DWA obstacle avoidance (sonar-friendly)
        self.declare_parameter('obstacle_turn_bias', 1.5)
        self.declare_parameter('dwa_fix_enabled', True)
        self.declare_parameter('sonar_speed_reduction_factor', 0.8)  # NEW: Less aggressive speed reduction for sonars
        
        # NEW: Parameters for emergency command filtering
        self.declare_parameter('filter_controller_emergency_commands', True)  # Enable filtering
        self.declare_parameter('emergency_linear_threshold', 0.1)  # Threshold for detecting emergency stop commands
        self.declare_parameter('emergency_angular_threshold', 0.8)  # Threshold for detecting emergency turn commands

    def get_ros_parameters(self):
        """Get parameter values."""
        self.goal_tolerance = self.get_parameter('goal_tolerance').value
        self.lookahead_distance_min = self.get_parameter('lookahead_distance_min').value
        self.lookahead_distance_max = self.get_parameter('lookahead_distance_max').value
        self.control_frequency = self.get_parameter('control_frequency').value
        self.state_change_timeout = self.get_parameter('state_change_timeout').value
        self.path_following_enabled = self.get_parameter('path_following_enabled').value
        self.path_lookahead_points = self.get_parameter('path_lookahead_points').value
        self.goal_reached_duration = self.get_parameter('goal_reached_duration').value
        self.adaptive_lookahead = self.get_parameter('adaptive_lookahead').value
        self.obstacle_hysteresis = self.get_parameter('obstacle_hysteresis').value
        self.control_period = self.get_parameter('control_period').value
        
        self.max_linear_speed = self.get_parameter('max_linear_speed').value
        self.max_angular_speed = self.get_parameter('max_angular_speed').value
        
        self.max_speed = self.get_parameter('max_speed').value
        self.min_speed = self.get_parameter('min_speed').value
        self.max_yaw_rate = self.get_parameter('max_yaw_rate').value
        self.max_accel = self.get_parameter('max_accel').value
        self.velocity_resolution = self.get_parameter('velocity_resolution').value
        self.predict_time = self.get_parameter('predict_time').value
        
        self.sonar_threshold = self.get_parameter('sonar_threshold').value
        self.obstacle_threshold = self.get_parameter('obstacle_threshold').value
        self.obstacle_check_frequency = self.get_parameter('obstacle_check_frequency').value
        self.emergency_stop_threshold = self.get_parameter('emergency_stop_threshold').value
        self.use_vision_obstacles = self.get_parameter('use_vision_obstacles').value
        self.obstacle_check_counter = 0
        
        self.front_depth_threshold = self.get_parameter('front_depth_threshold').value
        self.front_depth_timeout = self.get_parameter('front_depth_timeout').value
        self.use_front_depth = self.get_parameter('use_front_depth').value
        self.front_depth_emergency_threshold = self.get_parameter('front_depth_emergency_threshold').value
        
        # New parameters for enhanced turning (non-sonar only)
        self.turn_in_place_speed = self.get_parameter('turn_in_place_speed').value
        self.turn_lock_duration = self.get_parameter('turn_lock_duration').value
        self.sonar_speed_reduction_factor = self.get_parameter('sonar_speed_reduction_factor').value
        
        # NEW: Emergency command filtering parameters
        self.filter_controller_emergency_commands = self.get_parameter('filter_controller_emergency_commands').value
        self.emergency_linear_threshold = self.get_parameter('emergency_linear_threshold').value
        self.emergency_angular_threshold = self.get_parameter('emergency_angular_threshold').value

    def setup_ros_interfaces(self):
        """Setup ROS publishers and subscribers."""
        self.pose_sub = self.create_subscription(
            PoseStamped, 'robot_pose', self.pose_callback, 10)
        self.path_sub = self.create_subscription(
            Path, 'planned_path', self.path_callback, 10)
        self.servo_sub = self.create_subscription(
            String, '/servo', self.servo_callback, 10)
            
        sonar_topics = [
            ('/sonar/left/front', 'left_front'),
            ('/sonar/left/rear', 'left_rear'),
            ('/sonar/right/front', 'right_front'),
            ('/sonar/right/rear', 'right_rear')
        ]
        
        self.sonar_subs = []
        for topic, name in sonar_topics:
            sub = self.create_subscription(
                Range,
                topic,
                lambda msg, name=name: self.sonar_callback(msg, name),
                10
            )
            self.sonar_subs.append(sub)

        if self.use_front_depth:
            self.front_depth_sub = self.create_subscription(
                Range,
                'front_depth',
                self.front_depth_callback,
                10
            )
            self.get_logger().info("Subscribed to front_depth topic for water boundary detection")

        self.water_mask_sub = self.create_subscription(
            Image,
            '/water_mask',
            self.water_mask_callback,
            10
        )
        self.get_logger().info("Subscribed to raw water_mask topic")
        
        self.water_mask_status_pub = self.create_publisher(
            String,
            'water_mask_status',
            10
        )

        self.object_poses_sub = self.create_subscription(
            PoseArray,
            'object_poses',
            self.object_poses_callback,
            10
        )
        
        if self.use_vision_obstacles:
            self.obstacle_poses_sub = self.create_subscription(
                PoseArray,
                'obstacle_poses',
                self.obstacle_poses_callback,
                10
            )
            
        self.cmd_vel_pub = self.create_publisher(
            Twist, '/turtle1/cmd_vel', 10)
        self.state_pub = self.create_publisher(
            String, '/control_state', 10)
        
        self.current_target_pub = self.create_publisher(
            PoseStamped, '/control/current_target', 10)
        
        self.tracking_error_pub = self.create_publisher(
            String, '/control/tracking_error', 10)

    def servo_callback(self, msg):
        """Handle servo state updates."""
        try:
            new_state = int(msg.data)
            if new_state != self.servo_state:
                self.get_logger().info(f"Servo state changed from {self.servo_state} to {new_state}")
                self.servo_state = new_state
                self.servo_state_time = self.get_clock().now()
                
                if new_state == 1:
                    self.get_logger().info("Collection started - ensuring forward motion")
                elif new_state == 2:
                    self.get_logger().info("Collection finished - returning to normal control")
        except ValueError:
            self.get_logger().error(f"Invalid servo state: {msg.data}")

    def water_mask_callback(self, msg: Image):
        """Handle raw water mask updates and determine status."""
        try:
            bridge = CvBridge()
            water_mask = bridge.imgmsg_to_cv2(msg, "mono8")
            water_mask = (water_mask > 128).astype(np.uint8)
            
            self.water_mask_timestamp = self.get_clock().now().nanoseconds / 1e9
            
            status, mask_info = self.analyze_water_mask(water_mask)
            
            # When status changes, update and publish
            if status != self.water_mask_status:
                self.water_mask_status = status
                status_msg = String()
                status_msg.data = status
                self.water_mask_status_pub.publish(status_msg)
                self.dwa_controller.update_water_mask_status(status)
                self.get_logger().info(f"Water mask status updated: {status}")
            
            # Handle critical conditions - WATER MASK ISSUES STILL TRIGGER EMERGENCY
            if status in ["lost", "multiple", "small"]:
                # Always update emergency turn direction even during turns
                if status == "lost":
                    self.emergency_turn_direction = 0.0
                else:
                    # Store turn direction but with a strong persistence
                    if self.emergency_turn_direction is None:
                        self.emergency_turn_direction = mask_info['turn_direction']
                    elif self.turning_in_progress and hasattr(self, 'turning_start_time'):
                        # Don't change direction during active turns unless we've been turning
                        # in the same direction for a while (prevents oscillation)
                        current_time = self.get_clock().now().nanoseconds / 1e9
                        if (current_time - self.turning_start_time) < (self.turn_lock_duration * 2):
                            # During active turn, keep existing direction for stability
                            pass
                        else:
                            # After long enough in one direction, allow direction update
                            if 'direction_confidence' in mask_info and mask_info['direction_confidence'] > 0.4:
                                self.emergency_turn_direction = mask_info['turn_direction']
                    else:
                        # Update direction (not mid-turn)
                        self.emergency_turn_direction = mask_info['turn_direction']
                
                # Log the condition
                self.get_logger().warn(f"Critical water mask condition: {status.upper()}")
                
                # Only trigger state change if not already turning
                if not self.turn_in_place_active and not self.current_state == ControlState.TURN_IN_PLACE:
                    # Convert to turn-in-place for faster, more controlled response
                    self.emergency_stop_active = True
                    self.emergency_stop_time = self.get_clock().now()
                    self.turn_in_place_active = True  # Activate turn mode instead of emergency stop
                    self.turning_in_progress = True
                    self.turning_start_time = self.get_clock().now().nanoseconds / 1e9
                    self.transition_to_state(ControlState.TURN_IN_PLACE)
            
            # Return to normal state if conditions clear
            elif status == "normal" and self.emergency_stop_active and not self.turn_in_place_active:
                self.get_logger().info("Water mask returned to normal - exiting emergency mode")
                self.emergency_stop_active = False
                self.emergency_turn_direction = None
        
        except Exception as e:
            self.get_logger().error(f"Error in water mask callback: {str(e)}")
            self.get_logger().error(traceback.format_exc())

    def get_sonar_value(self, sonar_name):
        """Get a single sonar reading with proper validation."""
        if sonar_name not in self.sonar_data:
            return float('inf')
            
        data = self.sonar_data[sonar_name]
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if isinstance(data, dict) and 'timestamp' in data and 'range' in data:
            if current_time - data['timestamp'] > 1.0:
                return float('inf')
            return data['range']
        elif isinstance(data, (int, float)) and not math.isnan(data):
            return data
        
        return float('inf')
    
    def analyze_water_mask(self, water_mask):
        """Analyze water mask to determine status and turn direction with stability."""
        status = "lost"
        mask_info = {'turn_direction': 0.0}
        
        try:
            if water_mask is None:
                return status, mask_info
                
            h, w = water_mask.shape[:2]
            
            contours, _ = cv2.findContours(water_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.get_logger().debug("No water contours found - status: lost")
                return "lost", mask_info
                
            significant_contours = []
            min_significant_area = h * w * 0.01
            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= min_significant_area:
                    significant_contours.append(contour)
        
            num_significant_contours = len(significant_contours)
            
            if num_significant_contours == 0:
                return "lost", mask_info
    
            # Determine raw turn direction from mask
            left_half = water_mask[:, :w//2]
            right_half = water_mask[:, w//2:]
            left_area = np.sum(left_half) / 255.0
            right_area = np.sum(right_half) / 255.0
    
            raw_turn_direction = 1.0 if left_area > right_area else -1.0
            
            # Calculate the ratio difference to determine confidence
            total_area = left_area + right_area
            if total_area > 0:
                ratio_diff = abs(left_area - right_area) / total_area
                mask_info['direction_confidence'] = ratio_diff
            else:
                mask_info['direction_confidence'] = 0.0
            
            # Apply turn direction stability logic
            turn_direction = self.stabilize_turn_direction(raw_turn_direction, mask_info['direction_confidence'])
            mask_info['turn_direction'] = turn_direction
            
            # Determine water mask status
            if num_significant_contours >= 2:
                status = "multiple"
            else:
                largest_area = cv2.contourArea(max(contours, key=cv2.contourArea))
                min_area = h * w * 0.1
                if largest_area < min_area:
                    status = "small"
                else:
                    status = "normal"
    
            return status, mask_info
            
        except Exception as e:
            self.get_logger().error(f"Error analyzing water mask: {str(e)}")
            self.get_logger().error(traceback.format_exc())
            return status, mask_info

    def stabilize_turn_direction(self, raw_direction, confidence=0.0):
        """
        Stabilize turn direction to prevent oscillation with confidence weighting.
        Higher confidence values (closer to 1.0) mean we trust the raw direction more.
        """
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # If we're in turn-in-place mode, handle specially
        if self.turn_in_place_active:
            # If we haven't started a turn yet, initialize
            if not self.turning_in_progress:
                self.turning_in_progress = True
                self.turning_start_time = current_time
                
                # Use target heading if available (for goal-oriented turns)
                if self.turn_target_heading is not None and self.current_pose is not None:
                    current_yaw = self.get_robot_yaw(self.current_pose)
                    angle_diff = self.normalize_angle(self.turn_target_heading - current_yaw)
                    self.emergency_turn_direction = 1.0 if angle_diff > 0 else -1.0
                    self.get_logger().info(f"Starting goal-oriented turn to {math.degrees(self.turn_target_heading):.1f}° " +
                                         f"(currently at {math.degrees(current_yaw):.1f}°, " +
                                         f"turning {'LEFT' if self.emergency_turn_direction > 0 else 'RIGHT'})")
                else:
                    # Otherwise use water mask direction
                    self.emergency_turn_direction = raw_direction
                    self.get_logger().info(f"Starting turn-in-place with direction: {'LEFT' if raw_direction > 0 else 'RIGHT'}")
                
                return self.emergency_turn_direction
            
            # If we're in the middle of a turn, maintain direction for lock_duration
            elapsed = current_time - self.turning_start_time
            if elapsed < self.turn_lock_duration:
                # Keep the same direction during the lock period
                return self.emergency_turn_direction
            
            # After lock duration, check if we need to update direction based on target heading
            if self.turn_target_heading is not None and self.current_pose is not None:
                current_yaw = self.get_robot_yaw(self.current_pose)
                angle_diff = self.normalize_angle(self.turn_target_heading - current_yaw)
                
                # If we're close to target heading, we might flip direction
                if abs(angle_diff) < 0.2:  # ~11 degrees
                    self.get_logger().info(f"Very close to target heading, fine-tuning turn")
                    return 1.0 if angle_diff > 0 else -1.0
                
                # If we're turning in the wrong direction relative to target, update direction
                wrong_direction = (angle_diff > 0 and self.emergency_turn_direction < 0) or \
                                 (angle_diff < 0 and self.emergency_turn_direction > 0)
                                 
                if wrong_direction:
                    new_direction = 1.0 if angle_diff > 0 else -1.0
                    self.get_logger().info(f"Correcting turn direction: {self.emergency_turn_direction:.1f} -> {new_direction:.1f}")
                    self.emergency_turn_direction = new_direction
                    self.turning_start_time = current_time  # Reset lock timer
            
            # Only change if we have high confidence in new direction
            elif confidence > 0.3 and raw_direction * self.emergency_turn_direction < 0:
                self.get_logger().info(f"Updating turn direction after lock with confidence {confidence:.2f}: " +
                                     f"{self.emergency_turn_direction:.1f} -> {raw_direction:.1f}")
                self.turning_start_time = current_time  # Reset lock timer
                self.emergency_turn_direction = raw_direction
            
            return self.emergency_turn_direction
        
        # Not in turn-in-place mode - use history with weighted confidence
        # Add to history with confidence weighting
        confidence_threshold = 0.2
        if confidence > confidence_threshold:
            self.turn_direction_history.append(raw_direction)
            if len(self.turn_direction_history) > self.turn_direction_history_size:
                self.turn_direction_history.pop(0)
        
        # If history is empty, just use raw direction
        if not self.turn_direction_history:
            return raw_direction
        
        # Use majority vote for stability
        left_votes = len([d for d in self.turn_direction_history if d > 0])
        right_votes = len([d for d in self.turn_direction_history if d < 0])
        
        if left_votes > right_votes:
            return 1.0
        elif right_votes > left_votes:
            return -1.0
        else:
            # Tie - use most recent with sufficient confidence
            return raw_direction

    def front_depth_callback(self, msg: Range):
        """Handle front depth data for water boundary detection - STILL TRIGGERS EMERGENCY."""
        try:
            if math.isnan(msg.range):
                self.get_logger().debug(f"Invalid front depth reading: NaN")
                return

            if msg.range <= 0.01:
                self.get_logger().warn(f"Critical front depth reading: {msg.range:.2f}m (zero or near-zero, treating as emergency)")
                self.front_depth_value = 0.01
            elif msg.range < msg.min_range or msg.range > msg.max_range:
                self.get_logger().debug(f"Invalid front depth reading: {msg.range:.2f}m (outside range [{msg.min_range}, {msg.max_range}])")
                return
            else:
                self.front_depth_value = msg.range

            self.front_depth_timestamp = self.get_clock().now().nanoseconds / 1e9
            self.get_logger().debug(f"Received water boundary depth: {self.front_depth_value:.2f}m")
            self.dwa_controller.update_front_depth(self.front_depth_value)

            # FRONT DEPTH STILL TRIGGERS EMERGENCY ACTIONS (unlike sonars)
            if self.front_depth_value <= self.front_depth_emergency_threshold:
                is_mask_small = (self.water_mask_status == "small" and 
                                self.water_mask_timestamp is not None and
                                (self.front_depth_timestamp - self.water_mask_timestamp) < self.water_mask_status_timeout)
                
                if self.turn_in_place_active or self.current_state == ControlState.TURN_IN_PLACE:
                    if self.front_depth_value <= 0.1:
                        self.emergency_stop_active = True
                        self.emergency_stop_time = self.get_clock().now()
                    return
                    
                if is_mask_small and self.front_depth_value > 0.1:
                    self.get_logger().warn(f"Water boundary close: {self.front_depth_value:.2f}m with small mask - proceeding with caution")
                else:
                    self.get_logger().warn(f"Water boundary critical: {self.front_depth_value:.2f}m")
                    
                    if self.front_depth_value <= 0.1:  # Very critical condition
                        self.emergency_stop_active = True
                        self.emergency_stop_time = self.get_clock().now()
                        self.turn_in_place_active = True
                        self.turning_in_progress = True
                        self.turning_start_time = self.get_clock().now().nanoseconds / 1e9
                        if self.emergency_turn_direction is None:
                            self.emergency_turn_direction = 1.0  # Default left turn
                            self.get_logger().info("Emergency turn direction set to LEFT for critical depth")
                        self.transition_to_state(ControlState.TURN_IN_PLACE)
                    elif self.front_depth_value <= self.front_depth_emergency_threshold:
                        self.emergency_stop_active = True
                        self.emergency_stop_time = self.get_clock().now()
                        if self.emergency_turn_direction is None:
                            self.emergency_turn_direction = 1.0  # Default left turn
                            self.get_logger().info("Emergency turn direction set to LEFT for close depth")
                        self.transition_to_state(ControlState.EMERGENCY_STOP)
        except Exception as e:
            self.get_logger().error(f"Error in front depth callback: {str(e)}")

    def check_front_depth_obstacle(self, strict_threshold=False):
        """Check if front depth indicates an obstacle."""
        if not self.use_front_depth or self.front_depth_value is None:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if self.use_front_depth and (
                self.front_depth_timestamp is None or 
                (current_time - self.front_depth_timestamp) > self.front_depth_timeout
            ):
                self.get_logger().warn("Front depth data missing or stale - assuming close to water boundary")
                return True
            return False

        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.front_depth_timestamp and (current_time - self.front_depth_timestamp) < self.front_depth_timeout:
            threshold = self.front_depth_threshold * 0.8 if strict_threshold else self.front_depth_threshold
            
            if (self.water_mask_status == "small" and 
                self.water_mask_timestamp is not None and
                (current_time - self.water_mask_timestamp) < self.water_mask_status_timeout):
                threshold *= 0.8
                
            if self.front_depth_value <= threshold:
                self.get_logger().debug(f"{'VERY CLOSE to' if strict_threshold else 'Close to'} water boundary: {self.front_depth_value:.2f}m vs threshold {threshold:.2f}m")
                return True
            elif self.front_depth_value <= 0.01:
                self.get_logger().warn(f"Zero/near-zero water boundary detected: {self.front_depth_value:.2f}m")
                return True

        return False

    def sonar_callback(self, msg: Range, sonar_name: str):
        """Handle sonar range updates - MODIFIED: ZERO emergency triggers from sonars."""
        try:
            if math.isnan(msg.range) or msg.range < msg.min_range or msg.range > msg.max_range:
                self.get_logger().debug(f"Invalid sonar reading from {sonar_name}: {msg.range}")
                return
                
            self.sonar_data[sonar_name] = {
                'range': msg.range,
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            }
            
            self.update_controllers_sensor_data()
            
            # COMPLETELY REMOVED: Any emergency check calls from sonar callback
            # Sonars ONLY affect DWA state switching via check_for_obstacles()
            # NO emergency stops, NO emergency turns, NO emergency anything from sonars
            
            # Log very close obstacles for debugging but do NOT trigger emergency
            if msg.range < 0.3:
                self.get_logger().debug(f"Sonar {sonar_name}: {msg.range:.2f}m (close but NO emergency - DWA will handle)")
            
        except Exception as e:
            self.get_logger().error(f"Error in sonar callback: {str(e)}")

    def check_critical_emergency_conditions(self):
        """Check for CRITICAL emergency conditions - ONLY FRONT DEPTH AND WATER MASK."""
        # Skip emergency checks during turns - we want to complete the turn without interruptions
        if self.turn_in_place_active or self.current_state == ControlState.TURN_IN_PLACE:
            return
            
        # COMPLETELY REMOVED: Any obstacle-based emergency logic
        # ONLY front depth triggers emergency (water boundary detection)
        # NOTE: Water mask emergencies are handled in water_mask_callback()
        
        if self.use_front_depth and self.front_depth_value is not None:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if self.front_depth_timestamp and (current_time - self.front_depth_timestamp) < 0.5:
                if self.front_depth_value < self.front_depth_emergency_threshold:
                    self.get_logger().warn(f"EMERGENCY STOP triggered by water boundary: {self.front_depth_value:.2f}m")
                    
                    # For severe water boundary issues, use turn-in-place for better recovery
                    if self.front_depth_value < 0.1:
                        self.emergency_stop_active = True
                        self.emergency_stop_time = self.get_clock().now()
                        
                        # Activate turn-in-place mode
                        self.turn_in_place_active = True
                        self.turning_in_progress = True
                        self.turning_start_time = current_time
                        
                        # Use water mask direction if available, otherwise turn left
                        if self.emergency_turn_direction is None:
                            self.emergency_turn_direction = 1.0  # Default left turn
                            
                        self.transition_to_state(ControlState.TURN_IN_PLACE)
                    else:
                        # Standard emergency stop for less critical cases
                        self.emergency_stop_active = True
                        self.emergency_stop_time = self.get_clock().now()
                        self.transition_to_state(ControlState.EMERGENCY_STOP)
                    return

    def object_poses_callback(self, msg: PoseArray):
        """Handle detected object poses."""
        self.detected_objects = msg.poses
        self.update_controllers_sensor_data()

    def obstacle_poses_callback(self, msg: PoseArray):
        """Handle detected obstacle poses."""
        self.detected_obstacles = msg.poses
        
        if self.use_vision_obstacles:
            obstacle_count = len(msg.poses)
            if obstacle_count > 0:
                self.get_logger().debug(f"Detected {obstacle_count} obstacles from vision")
                for i in range(min(3, obstacle_count)):
                    pose = msg.poses[i]
                    self.get_logger().debug(f"  Obstacle {i}: ({pose.position.x:.2f}, {pose.position.y:.2f})")
        
            self.update_controllers_sensor_data()

    def update_controllers_sensor_data(self):
        """Update both controllers with the latest sensor data."""
        sonar_ranges = {}
        for name, data in self.sonar_data.items():
            if isinstance(data, dict) and 'range' in data:
                sonar_ranges[name] = data['range']
            elif isinstance(data, (int, float)) and not math.isnan(data):
                sonar_ranges[name] = data
            
        self.fuzzy_controller.update_sonar_data(sonar_ranges)
        self.dwa_controller.update_sensor_data(sonar_ranges, None)
        
        if self.use_front_depth and self.front_depth_value is not None:
            self.dwa_controller.update_front_depth(self.front_depth_value)
        
        if self.use_vision_obstacles and self.detected_obstacles:
            obstacle_positions = []
            for obs in self.detected_obstacles:
                obstacle_positions.append([obs.position.x, obs.position.y])
            self.fuzzy_controller.update_obstacle_positions(obstacle_positions)

    def pose_callback(self, msg: PoseStamped):
        """Handle robot pose updates."""
        self.current_pose = msg.pose
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        if hasattr(self, 'last_pose_time') and self.last_pose_time is not None:
            dt = current_time - self.last_pose_time
            if dt > 0:
                dx = msg.pose.position.x - self.last_pose.position.x
                dy = msg.pose.position.y - self.last_pose.position.y
                
                alpha = 0.3
                self.current_linear_velocity = (1-alpha) * self.current_linear_velocity + alpha * math.sqrt(dx*dx + dy*dy) / dt
                
                prev_yaw = self.get_robot_yaw(self.last_pose)
                current_yaw = self.get_robot_yaw(msg.pose)
                dyaw = self.normalize_angle(current_yaw - prev_yaw)
                self.current_angular_velocity = (1-alpha) * self.current_angular_velocity + alpha * dyaw / dt
                
                # Check if we've reached our target heading during turn-in-place
                if self.turn_in_place_active and self.turn_target_heading is not None:
                    angle_diff = self.normalize_angle(self.turn_target_heading - current_yaw)
                    if abs(angle_diff) < math.radians(5):  # Within 5 degrees
                        self.get_logger().info(f"Reached target heading: {math.degrees(current_yaw):.1f}° (target: {math.degrees(self.turn_target_heading):.1f}°)")
                        self.turn_in_place_active = False
                        self.turning_in_progress = False
                        self.turn_target_heading = None
                        if self.current_path and not self.path_completed:
                            self.transition_to_state(ControlState.FUZZY_CONTROL)
                        else:
                            self.transition_to_state(ControlState.IDLE)
                
                if self.last_pose_time % 5 < dt:
                    self.get_logger().debug(f"Current velocity: linear={self.current_linear_velocity:.2f} m/s, angular={math.degrees(self.current_angular_velocity):.2f} deg/s")
        
        self.last_pose = msg.pose
        self.last_pose_time = current_time

    def path_callback(self, msg: Path):
        """Handle path updates with special handling for turn-in-place."""
        has_turn_marker = ":turn" in msg.header.frame_id
        
        # Track if we're receiving a new path during an active turn
        if (self.turn_in_place_active or self.current_state == ControlState.TURN_IN_PLACE) and not has_turn_marker:
            self.new_path_during_turn = True
            
            # Don't interrupt critical turns (water boundary recovery)
            if self.force_turn_completion or self.front_depth_value is not None and self.front_depth_value <= 0.1:
                if not self.turning_in_progress or (
                   self.turning_start_time is not None and 
                   (self.get_clock().now().nanoseconds / 1e9 - self.turning_start_time) < self.min_turn_duration):
                    self.get_logger().warn("Ignoring new path - critical water boundary turn in progress")
                    return
        
        if has_turn_marker:
            self.turn_in_place_active = True
            self.get_logger().info("Turn-in-place mode activated")
            
            # Get target information if available (frame_id format: "map:turn:X.XX")
            parts = msg.header.frame_id.split(":")
            msg.header.frame_id = parts[0]  # Keep base frame
            
            # Reset turning state for new turn command
            self.turning_in_progress = False
            self.turning_start_time = None
            self.turn_direction_history = []
            self.turn_completed = False
            self.force_turn_completion = "force" in msg.header.frame_id.lower()
            
            # If we have a target heading in the frame_id
            if len(parts) >= 3:
                try:
                    target_heading = float(parts[2])
                    self.turn_target_heading = math.radians(target_heading)
                    self.get_logger().info(f"Turn-in-place target heading: {target_heading}°" +
                                         f"{' (FORCED COMPLETION)' if self.force_turn_completion else ''}")
                except ValueError:
                    self.turn_target_heading = None
            else:
                self.turn_target_heading = None
            
            # Switch to dedicated turn-in-place state
            self.transition_to_state(ControlState.TURN_IN_PLACE)
        else:
            # If we're not in a critical turn, reset turn flags
            if not self.force_turn_completion:
                self.turn_in_place_active = False
                self.turn_target_heading = None
                self.turn_completed = False

        # Handle path reception during cool-down period
        current_time = self.get_clock().now()
        if (self.path_completed and self.goal_reached_time and 
            (current_time.nanoseconds - self.goal_reached_time.nanoseconds) / 1e9 < self.goal_reached_duration):
            self.get_logger().info("Ignoring new path - in goal reached cool-down period")
            return
        
        # Reset emergency state with new path unless we're in critical water boundary condition
        if self.emergency_stop_active and not has_turn_marker:
            if self.front_depth_value is not None and self.front_depth_value <= 0.1:
                self.get_logger().warn("New path received but maintaining emergency state - critical water boundary")
            else:
                self.get_logger().info("Received new path - clearing emergency stop state")
                self.emergency_stop_active = False
                self.consecutive_emergency_count = 0  # Reset emergency counter
        
        if msg.poses:
            if len(msg.poses) < 2 and not has_turn_marker:
                self.get_logger().warn("Received path with only one waypoint - need at least two")
                return
            
            self.current_path = msg.poses
            self.current_path_index = 0
            
            self.cross_track_error_avg = 0.0
            self.cross_track_error_samples = 0
            
            self.path_completed = False
            self.goal_reached_time = None
            
            self.current_goal = self.get_lookahead_point()
            
            # Don't change state during critical turns
            if self.force_turn_completion and self.current_state == ControlState.TURN_IN_PLACE:
                self.get_logger().warn("New path received but maintaining turn-in-place - force completion enabled")
            elif has_turn_marker:
                # Already handled above
                pass
            elif self.check_for_obstacles(force_check=True):
                self.transition_to_state(ControlState.DWA_CONTROL)
            else:
                self.transition_to_state(ControlState.FUZZY_CONTROL)
            
            self.get_logger().info(f"Received {'turn-in-place command' if has_turn_marker else 'new path with ' + str(len(msg.poses)) + ' waypoints'}")
            
            if not has_turn_marker and len(msg.poses) > 0:
                start = msg.poses[0].pose.position
                end = msg.poses[-1].pose.position
                self.get_logger().info(f"Path from ({start.x:.2f}, {start.y:.2f}) to ({end.x:.2f}, {end.y:.2f})")
        else:
            self.current_path = []
            self.current_goal = None
            self.path_completed = False
            
            # Don't exit turn-in-place for empty paths during critical turns
            if not (self.force_turn_completion and self.current_state == ControlState.TURN_IN_PLACE):
                self.transition_to_state(ControlState.IDLE)
                self.get_logger().info("Received empty path - stopping robot")

    def get_lookahead_point(self):
        """Get the appropriate target point from the path."""
        if self.path_completed or not self.current_path:
            return None
        
        if not self.path_following_enabled:
            return self.current_path[-1].pose
        
        if self.current_pose:
            min_dist = float('inf')
            closest_idx = 0
            
            for i, pose_stamped in enumerate(self.current_path):
                pose = pose_stamped.pose
                dx = pose.position.x - self.current_pose.position.x
                dy = pose.position.y - self.current_pose.position.y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            
            lookahead_points = self.path_lookahead_points
            
            if self.adaptive_lookahead:
                speed_factor = self.current_linear_velocity / self.max_linear_speed
                
                curvature_factor = 1.0
                if closest_idx + 1 < len(self.current_path) and closest_idx > 0:
                    try:
                        p1 = self.current_path[closest_idx-1].pose.position
                        p2 = self.current_path[closest_idx].pose.position
                        p3 = self.current_path[closest_idx+1].pose.position
                        
                        v1 = [p2.x - p1.x, p2.y - p1.y]
                        v2 = [p3.x - p2.x, p3.y - p2.y]
                        
                        v1_mag = math.sqrt(v1[0]**2 + v1[1]**2)
                        v2_mag = math.sqrt(v2[0]**2 + v2[1]**2)
                        
                        if v1_mag > 0 and v2_mag > 0:
                            dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                            angle = math.acos(max(-1.0, min(1.0, dot_product / (v1_mag * v2_mag))))
                            curvature_factor = max(0.5, 1.0 - angle / math.pi)
                    except Exception as e:
                        self.get_logger().warn(f"Error calculating path curvature: {str(e)}")
                
                lookahead_distance = self.lookahead_distance_min + (self.lookahead_distance_max - self.lookahead_distance_min) * speed_factor * curvature_factor
                
                if closest_idx + 1 < len(self.current_path):
                    next_point = self.current_path[closest_idx+1].pose.position
                    point_distance = math.sqrt((next_point.x - self.current_path[closest_idx].pose.position.x)**2 + 
                                            (next_point.y - self.current_path[closest_idx].pose.position.y)**2)
                    if point_distance > 0:
                        lookahead_points = max(1, int(lookahead_distance / point_distance))
            
            lookahead_points = max(1, min(lookahead_points, len(self.current_path) - closest_idx - 1))
            
            lookahead_idx = min(closest_idx + lookahead_points, len(self.current_path) - 1)
            
            self.current_path_index = closest_idx
            
            self.publish_current_target(self.current_path[lookahead_idx].pose)
            
            if self.cross_track_error_samples % 50 == 0:
                self.get_logger().debug(f"Lookahead: {lookahead_points} points ahead " +
                                      f"(speed={self.current_linear_velocity:.2f}m/s, idx={closest_idx}/{len(self.current_path)})")
            
            return self.current_path[lookahead_idx].pose
        
        return self.current_path[-1].pose

    def publish_current_target(self, target_pose):
        """Publish the current target pose for visualization."""
        target_msg = PoseStamped()
        target_msg.header.frame_id = "map"
        target_msg.header.stamp = self.get_clock().now().to_msg()
        target_msg.pose = target_pose
        self.current_target_pub.publish(target_msg)

    def check_for_obstacles(self, force_check=False):
        """MODIFIED: Check for obstacles with hysteresis - SONARS ONLY FOR DWA SWITCHING."""
        current_time = self.get_clock().now()
        
        if hasattr(self, 'last_obstacle_check_time') and self.last_obstacle_check_time:
            time_diff = (current_time.nanoseconds - self.last_obstacle_check_time.nanoseconds) / 1e9
        else:
            time_diff = float('inf')
            self.last_obstacle_check_time = current_time
            
        if not force_check and time_diff < self.control_period * self.obstacle_check_frequency:
            return self.sonar_obstacle_detected_count > self.obstacle_hysteresis
        
        self.last_obstacle_check_time = current_time
        
        has_obstacle = False
        
        # Check sonar obstacles for DWA switching (not emergency)
        sonar_obstacles = self.check_sonar_obstacles(strict_threshold=True)
        
        # Check vision obstacles
        vision_obstacles = False
        if self.use_vision_obstacles:
            vision_obstacles = self.check_vision_obstacles(strict_threshold=True)
        
        # Check front depth obstacles (for switching to DWA, not emergency)
        front_depth_obstacle = False
        if self.use_front_depth:
            front_depth_obstacle = self.check_front_depth_obstacle(strict_threshold=True)
        
        has_obstacle = sonar_obstacles or vision_obstacles or front_depth_obstacle
        
        if has_obstacle:
            self.sonar_obstacle_detected_count += 1
            self.sonar_obstacle_free_count = 0
        else:
            self.sonar_obstacle_detected_count = 0
            self.sonar_obstacle_free_count += 1
        
        obstacles_confirmed = self.sonar_obstacle_detected_count > self.obstacle_hysteresis
        obstacles_cleared = self.sonar_obstacle_free_count > self.obstacle_hysteresis
        
        if obstacles_confirmed:
            if obstacles_confirmed and not self.current_state == ControlState.DWA_CONTROL:
                self.get_logger().info("Obstacles detected - activating DWA controller for avoidance")
            return True
        elif obstacles_cleared:
            return False
        else:
            return self.sonar_obstacle_detected_count > self.sonar_obstacle_free_count

    def check_sonar_obstacles(self, strict_threshold=False):
        """Check for sonar obstacles - FOR DWA SWITCHING ONLY - NO EMERGENCY TRIGGERS."""
        valid_sonar_count = 0
        current_time = self.get_clock().now()
        
        threshold = self.sonar_threshold if not strict_threshold else self.sonar_threshold
        
        for sonar_name, data in self.sonar_data.items():
            if isinstance(data, dict) and 'timestamp' in data and 'range' in data:
                if isinstance(data['timestamp'], float):
                    if (current_time.nanoseconds / 1e9) - data['timestamp'] > 1.0:
                        continue
                elif hasattr(data['timestamp'], 'nanoseconds'):
                    if (current_time.nanoseconds - data['timestamp'].nanoseconds) / 1e9 > 1.0:
                        continue
                    
                if data['range'] < threshold:
                    self.get_logger().debug(f'Sonar obstacle detected by {sonar_name} at {data["range"]:.2f}m - FOR DWA ONLY (no emergency)')
                    return True
                    
                valid_sonar_count += 1
            elif isinstance(data, (int, float)) and not math.isnan(data):
                if data < threshold:
                    self.get_logger().debug(f'Sonar obstacle detected by {sonar_name} at {data:.2f}m - FOR DWA ONLY (no emergency)')
                    return True
                    
                valid_sonar_count += 1
        
        if valid_sonar_count == 0:
            self.get_logger().warn('No valid sonar data available')
            
        return False

    def check_vision_obstacles(self, strict_threshold=False):
        """Check for vision obstacles."""
        if not self.detected_obstacles or not self.current_pose:
            return False
            
        threshold = self.obstacle_threshold if not strict_threshold else self.obstacle_threshold * 0.6
            
        for obstacle in self.detected_obstacles:
            dx = obstacle.position.x - self.current_pose.position.x
            dy = obstacle.position.y - self.current_pose.position.y
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < threshold:
                if strict_threshold:
                    self.get_logger().debug(f'VERY CLOSE vision obstacle detected at {distance:.2f}m')
                else:
                    self.get_logger().debug(f'Vision obstacle detected at {distance:.2f}m')
                return True
                
        return False

    def determine_control_state(self):
        """Determine which control state to use - NO OBSTACLE EMERGENCIES."""
        if not self.current_pose:
            self.transition_to_state(ControlState.IDLE)
            return
            
        # Honor current state for special cases
        if self.current_state == ControlState.EMERGENCY_STOP and self.emergency_stop_active:
            return
        
        # Turn-in-place state has priority when active
        if self.turn_in_place_active:
            if self.current_state != ControlState.TURN_IN_PLACE:
                self.transition_to_state(ControlState.TURN_IN_PLACE)
            return
            
        if self.path_completed:
            self.transition_to_state(ControlState.IDLE)
            return

        # ONLY check for obstacles for DWA switching - NOT for emergency
        obstacle_detected = self.check_for_obstacles()
        has_path = self.current_path and not self.path_completed

        # REMOVED: Any obstacle-based emergency state transitions
        # Emergency state is ONLY set by front depth and water mask callbacks
        if self.emergency_stop_active:
            self.transition_to_state(ControlState.EMERGENCY_STOP)
        elif obstacle_detected and has_path:
            if self.current_state != ControlState.DWA_CONTROL:
                self.get_logger().info("Obstacles detected - switching to DWA for smooth avoidance")
            self.transition_to_state(ControlState.DWA_CONTROL)
        elif has_path:
            if self.current_state != ControlState.FUZZY_CONTROL:
                if self.current_state == ControlState.DWA_CONTROL:
                    self.get_logger().info("No more obstacles - returning to Fuzzy Control for path following")
                else:
                    self.get_logger().info("Using Fuzzy Control for path following")
            self.transition_to_state(ControlState.FUZZY_CONTROL)
        else:
            self.transition_to_state(ControlState.IDLE)

    def transition_to_state(self, new_state: ControlState):
        """Handle state transitions."""
        if new_state != self.current_state:
            self.get_logger().info(f'Transitioning from {self.current_state.value} to {new_state.value}')
            self.previous_state = self.current_state
            self.current_state = new_state
            self.publish_state()

    def publish_state(self):
        """Publish current control state."""
        msg = String()
        msg.data = self.current_state.value
        self.state_pub.publish(msg)

    def control_loop_callback(self):
        """Main control loop."""
        if not self.current_pose:
            self.publish_cmd_vel(0.0, 0.0)
            return

        if not self.path_completed and self.current_path:
            self.current_goal = self.get_lookahead_point()

        self.determine_control_state()

        if self.current_state == ControlState.IDLE:
            self.publish_cmd_vel(0.0, 0.0)
            return
        elif self.current_state == ControlState.EMERGENCY_STOP:
            self.handle_emergency_stop()
            return
        elif self.current_state == ControlState.TURN_IN_PLACE:
            self.handle_turn_in_place()
            return

        if not self.current_goal:
            self.publish_cmd_vel(0.0, 0.0)
            return

        target_pose = self.current_goal

        current_path_points = None
        if self.current_path:
            current_path_points = []
            for pose_stamped in self.current_path:
                current_path_points.append([
                    pose_stamped.pose.position.x,
                    pose_stamped.pose.position.y
                ])
        
        if self.current_state == ControlState.FUZZY_CONTROL:
            cmd_vel, goal_reached, cross_track_error = self.compute_fuzzy_control(
                target_pose, current_path_points)
            if cross_track_error is not None:
                self.update_cross_track_stats(cross_track_error)
                
        elif self.current_state == ControlState.DWA_CONTROL:
            have_valid_sensors = self.have_valid_sonar_data()
            if not have_valid_sensors:
                self.get_logger().warn('No valid sonar data for DWA, falling back to Fuzzy Control')
                cmd_vel, goal_reached, _ = self.compute_fuzzy_control(
                    target_pose, current_path_points)
            else:
                cmd_vel, goal_reached = self.compute_dwa_control(target_pose)
        else:
            self.publish_cmd_vel(0.0, 0.0)
            return

        if (self.current_path and 
            self.current_path_index >= len(self.current_path) - 1 and 
            goal_reached):
            self.get_logger().info('Final goal reached')
            self.path_completed = True
            self.goal_reached_time = self.get_clock().now()
            self.current_path = []
            self.current_goal = None
            self.transition_to_state(ControlState.IDLE)
            self.publish_cmd_vel(0.0, 0.0)
        else:
            self.publish_cmd_vel(cmd_vel.linear.x, cmd_vel.angular.z)

    def handle_turn_in_place(self):
        """Enhanced handler for turn-in-place state with higher speeds for boundary conditions."""
        if not self.current_pose:
            self.publish_cmd_vel(0.0, 0.0)
            return
            
        twist = Twist()
        twist.linear.x = 0.0  # No forward motion during turn-in-place
        
        # Check if we're in an emergency situation with water boundary
        is_emergency = (self.emergency_stop_active or 
                        self.water_mask_status in ["multiple", "small", "lost"] or
                        (self.front_depth_value is not None and self.front_depth_value <= 0.1))
        
        # Set a higher turn speed for emergency situations
        emergency_turn_speed = 2.5  # Much faster than normal
        normal_turn_speed = self.turn_in_place_speed
        
        # Determine turn direction using either target heading or water mask
        if self.turn_target_heading is not None:
            current_yaw = self.get_robot_yaw(self.current_pose)
            angle_diff = self.normalize_angle(self.turn_target_heading - current_yaw)
            
            # Adjust turn rate based on how far we are from target and emergency status
            base_speed = emergency_turn_speed if is_emergency else normal_turn_speed
            
            if abs(angle_diff) > math.radians(30):
                turn_rate = base_speed
            elif abs(angle_diff) > math.radians(10):
                turn_rate = base_speed * (0.8 if is_emergency else 0.7)
            else:
                turn_rate = base_speed * (0.5 if is_emergency else 0.3)
                
            turn_direction = 1.0 if angle_diff > 0 else -1.0
            
            # Log occasionally
            current_time = self.get_clock().now().nanoseconds / 1e9
            if not hasattr(self, 'last_turn_log_time') or current_time - self.last_turn_log_time > 1.0:
                self.get_logger().info(f"Turning to heading {math.degrees(self.turn_target_heading):.1f}°, " +
                                    f"current {math.degrees(current_yaw):.1f}°, " +
                                    f"diff {math.degrees(angle_diff):.1f}°, " +
                                    f"rate {turn_rate:.2f}" +
                                    f"{' (EMERGENCY SPEED)' if is_emergency else ''}")
                self.last_turn_log_time = current_time
            
            twist.angular.z = turn_direction * turn_rate
            
            # Check if we're done
            target_threshold = math.radians(5 if is_emergency else 2)
            if abs(angle_diff) < target_threshold:
                self.get_logger().info(f"Reached target heading {math.degrees(self.turn_target_heading):.1f}°")
                self.turn_in_place_active = False
                self.turning_in_progress = False
                self.turn_target_heading = None
                self.emergency_turn_direction = None  # Reset
                if self.current_path and not self.path_completed:
                    self.transition_to_state(ControlState.FUZZY_CONTROL)
                else:
                    self.transition_to_state(ControlState.IDLE)
                self.publish_cmd_vel(0.0, 0.0)
                return
        else:
            # Initialize emergency_turn_direction if None
            if self.emergency_turn_direction is None:
                self.emergency_turn_direction = 1.0  # Default to left
                self.get_logger().info("Emergency turn direction not set - defaulting to LEFT")
            
            # Use the direction with emergency-aware speed
            base_speed = emergency_turn_speed if is_emergency else normal_turn_speed
            
            # Use higher speed for critical conditions
            if self.front_depth_value is not None and self.front_depth_value <= 0.05:
                base_speed = emergency_turn_speed * 1.2  # Even faster
                self.get_logger().warn(f"Critical water boundary ({self.front_depth_value:.2f}m) - using max turn speed")
            
            twist.angular.z = self.emergency_turn_direction * base_speed
            direction_str = "LEFT" if self.emergency_turn_direction > 0 else "RIGHT"
            
            current_time = self.get_clock().now().nanoseconds / 1e9
            if not hasattr(self, 'last_turn_log_time') or current_time - self.last_turn_log_time > 1.0:
                self.get_logger().info(f"Turning in place {direction_str} at {base_speed:.2f} rad/s" +
                                    f"{' (EMERGENCY SPEED)' if is_emergency else ''}")
                self.last_turn_log_time = current_time
        
        # Apply velocity limits
        max_speed = self.max_angular_speed * (1.5 if is_emergency else 1.0)
        twist.angular.z = max(-max_speed, min(twist.angular.z, max_speed))
        
        # Send command
        self.publish_cmd_vel(twist.linear.x, twist.angular.z)

    def compute_fuzzy_control(self, target_pose, path_points=None):
        """Compute control using fuzzy logic - FILTER OUT SONAR EMERGENCY COMMANDS."""
        cmd_vel, goal_reached, cte = self.fuzzy_controller.compute_control(
            self.current_pose, target_pose, path_points)

        # FILTER OUT EMERGENCY COMMANDS FROM CONTROLLERS WHEN ONLY SONARS DETECT OBSTACLES
        if self.should_filter_emergency_commands():
            cmd_vel = self.filter_emergency_commands(cmd_vel, "FUZZY")

        if self.turn_in_place_active:
            cmd_vel.linear.x = 0.0
        
        if self.servo_state == 1:
            min_collection_speed = 0.3
            if cmd_vel.linear.x < min_collection_speed:
                self.get_logger().info(f"Boosting collection speed from {cmd_vel.linear.x:.2f} to {min_collection_speed:.2f}")
                cmd_vel.linear.x = min_collection_speed
            cmd_vel.angular.z *= 0.7
            
            current_time = self.get_clock().now()
            if self.servo_state_time:
                collection_elapsed = (current_time.nanoseconds - self.servo_state_time.nanoseconds) / 1e9
                if int(collection_elapsed) % 1 == 0:
                    self.get_logger().info(f"Collection in progress: {collection_elapsed:.1f}s, speed={cmd_vel.linear.x:.2f}")
        
        return cmd_vel, goal_reached, cte

    def compute_dwa_control(self, target_pose):
        """MODIFIED: Compute control using DWA with FILTERED emergency commands."""
        # Get DWA control output
        cmd_vel, goal_reached = self.dwa_controller.compute_control(
            self.current_pose, target_pose, with_goal=True)
        
        # FILTER OUT EMERGENCY COMMANDS FROM CONTROLLERS WHEN ONLY SONARS DETECT OBSTACLES
        if self.should_filter_emergency_commands():
            cmd_vel = self.filter_emergency_commands(cmd_vel, "DWA")
        
        # Get detailed obstacle information by directly accessing sonar data
        left_front = self.get_min_sonar_by_name('left_front')
        right_front = self.get_min_sonar_by_name('right_front')
        left_rear = self.get_min_sonar_by_name('left_rear') 
        right_rear = self.get_min_sonar_by_name('right_rear')
        
        # Calculate minimum distances for different areas
        min_front = min(left_front, right_front)
        min_left = min(left_front, left_rear)  
        min_right = min(right_front, right_rear)
        min_sonar_distance = min(min_front, min_left, min_right)
        
        # Detect side wall vs front obstacle scenarios
        front_obstacle = min_front < self.sonar_threshold
        left_wall = min_left < self.sonar_threshold * 1.5 and min_front > self.sonar_threshold
        right_wall = min_right < self.sonar_threshold * 1.5 and min_front > self.sonar_threshold
        
        # FIX for DWA turning toward obstacles
        dwa_fix_enabled = self.get_parameter('dwa_fix_enabled').value if hasattr(self, 'get_parameter') else True
        if dwa_fix_enabled and cmd_vel.angular.z != 0:
            # Side wall specific behavior
            if left_wall and cmd_vel.angular.z > 0:
                # Left wall but turning left - turn right instead
                self.get_logger().info(f"DWA TURN FIX: Left wall detected - turning right")
                cmd_vel.angular.z = -abs(cmd_vel.angular.z)
                # Allow forward motion along wall at reduced speed
                cmd_vel.linear.x = max(cmd_vel.linear.x, 0.5) if cmd_vel.linear.x > 0 else cmd_vel.linear.x
                
            elif right_wall and cmd_vel.angular.z < 0:
                # Right wall but turning right - turn left instead
                self.get_logger().info(f"DWA TURN FIX: Right wall detected - turning left")
                cmd_vel.angular.z = abs(cmd_vel.angular.z)
                # Allow forward motion along wall at reduced speed
                cmd_vel.linear.x = max(cmd_vel.linear.x, 0.5) if cmd_vel.linear.x > 0 else cmd_vel.linear.x
                
            # Front obstacle handling - more cautious
            elif front_obstacle:
                if left_front < right_front and cmd_vel.angular.z > 0:
                    # Front-left obstacle but turning left - turn right instead
                    self.get_logger().warn(f"DWA TURN FIX: Front-left obstacle - turning right")
                    cmd_vel.angular.z = -abs(cmd_vel.angular.z) * 1.2  # Sharper turn
                    
                elif right_front < left_front and cmd_vel.angular.z < 0:
                    # Front-right obstacle but turning right - turn left instead
                    self.get_logger().warn(f"DWA TURN FIX: Front-right obstacle - turning left")
                    cmd_vel.angular.z = abs(cmd_vel.angular.z) * 1.2  # Sharper turn
        
        # MODIFIED: LESS AGGRESSIVE speed scaling for sonars
        if front_obstacle:
            # Front obstacles require moderate speed reduction (less aggressive than before)
            speed_scale = max(self.sonar_speed_reduction_factor, min_front / self.sonar_threshold)
            cmd_vel.linear.x *= speed_scale
            self.get_logger().debug(f"Moderate speed reduction to {cmd_vel.linear.x:.2f} due to front obstacle at {min_front:.2f}m")
        elif left_wall or right_wall:
            # Side walls allow even more forward motion
            wall_distance = min_left if left_wall else min_right
            speed_scale = max(0.9, wall_distance / (self.sonar_threshold * 1.5))  # Much less reduction
            cmd_vel.linear.x *= speed_scale
            wall_side = "left" if left_wall else "right"
            self.get_logger().debug(f"Minimal speed reduction {cmd_vel.linear.x:.2f} while following {wall_side} wall at {wall_distance:.2f}m")
        
        # Water boundary handling (front depth) - THIS STILL DOES SPEED REDUCTION
        if self.use_front_depth and self.front_depth_value is not None:
            current_time = self.get_clock().now().nanoseconds / 1e9
            if self.front_depth_timestamp and (current_time - self.front_depth_timestamp) < self.front_depth_timeout:
                if self.front_depth_value < self.front_depth_threshold:
                    water_scale = max(0.3, self.front_depth_value / self.front_depth_threshold)
                    cmd_vel.linear.x *= water_scale
                    self.get_logger().debug(f"Reduced speed to {cmd_vel.linear.x:.2f} due to water boundary at {self.front_depth_value:.2f}m")
        
        # Servo collection handling
        if self.servo_state == 1:
            min_collection_speed = 0.3
            if cmd_vel.linear.x < min_collection_speed:
                self.get_logger().info(f"Boosting collection speed from {cmd_vel.linear.x:.2f} to {min_collection_speed:.2f}")
                cmd_vel.linear.x = min_collection_speed
            cmd_vel.angular.z *= 0.7
        
        return cmd_vel, goal_reached

    def handle_emergency_stop(self):
        """MODIFIED: Handle emergency stop behavior - ONLY WATER MASK AND FRONT DEPTH."""
        if self.servo_state == 1:
            # Only abort collection for very critical non-sonar conditions
            critical_front_depth = self.front_depth_value is not None and self.front_depth_value < 0.1
            
            if critical_front_depth:
                self.get_logger().warn("Critical water boundary emergency during collection - aborting collection")
            else:
                self.get_logger().info("Overriding emergency stop to continue collection sequence")
                self.emergency_stop_active = False
                twist = Twist()
                twist.linear.x = 0.2
                twist.angular.z = 0.0
                self.publish_cmd_vel(twist.linear.x, twist.angular.z)
                return

        twist = Twist()
        
        # Handle ONLY water mask and front depth conditions - NO OBSTACLE LOGIC
        if self.water_mask_status == "lost":
            twist.linear.x = -0.2
            twist.angular.z = 0.0
            self.get_logger().warn("Emergency: Water mask LOST - backing up straight")
        
        elif self.water_mask_status in ["small", "multiple"]:
            twist.linear.x = 0.0
            turn_speed = 0.6
            if self.emergency_turn_direction is None:
                self.emergency_turn_direction = 1.0  # Default to left
                self.get_logger().info("Emergency turn direction not set for water mask - defaulting to LEFT")
            twist.angular.z = self.emergency_turn_direction * turn_speed
            direction_str = "LEFT" if self.emergency_turn_direction > 0 else "RIGHT"
            self.get_logger().warn(f"Emergency: Water mask {self.water_mask_status.upper()} - turning in place to {direction_str}")
        
        elif self.front_depth_value is not None and self.front_depth_value < self.front_depth_emergency_threshold:
            twist.linear.x = 0.0
            turn_speed = 0.6
            if self.emergency_turn_direction is None:
                self.emergency_turn_direction = 1.0  # Default to left
                self.get_logger().info("Emergency turn direction not set for front depth - defaulting to LEFT")
            twist.angular.z = self.emergency_turn_direction * turn_speed
            direction_str = "LEFT" if self.emergency_turn_direction > 0 else "RIGHT"
            self.get_logger().warn(f"Emergency: Water boundary close ({self.front_depth_value:.2f}m) - turning in place to {direction_str}")
        
        else:
            # No emergency condition detected - stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().debug("Emergency state active but no specific emergency condition - stopping")
        
        self.publish_cmd_vel(twist.linear.x, twist.angular.z)
        
        current_time = self.get_clock().now()
        time_in_emergency = (current_time.nanoseconds - self.emergency_stop_time.nanoseconds) / 1e9
        
        # MODIFIED: Check only non-sonar conditions for emergency recovery
        non_sonar_obstacles_clear = (
            self.water_mask_status == "normal" and 
            (self.front_depth_value is None or self.front_depth_value > self.front_depth_emergency_threshold)
        )
        
        if (time_in_emergency >= self.emergency_recovery_duration and non_sonar_obstacles_clear):
            self.get_logger().info("Emergency conditions cleared - resuming normal operation")
            self.emergency_stop_active = False
            self.emergency_turn_direction = None
            if self.current_path:
                self.current_goal = self.get_lookahead_point()
                self.transition_to_state(ControlState.FUZZY_CONTROL)
            else:
                self.transition_to_state(ControlState.IDLE)

    def get_min_sonar_distance(self):
        """Get minimum valid sonar distance."""
        min_distance = float('inf')
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        for name, data in self.sonar_data.items():
            if isinstance(data, dict) and 'timestamp' in data:
                if current_time - data['timestamp'] > 1.0:
                    continue
                if data['range'] < min_distance:
                    min_distance = data['range']
            elif isinstance(data, (int, float)) and not math.isnan(data):
                min_distance = min(min_distance, data)
                
        return min_distance if min_distance < float('inf') else 3.0
    
    def get_min_sonar_by_name(self, name_pattern):
        """Get minimum sonar reading by name."""
        min_distance = float('inf')
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        for name, data in self.sonar_data.items():
            if name_pattern in name:
                if isinstance(data, dict) and 'timestamp' in data:
                    if current_time - data['timestamp'] > 1.0:
                        continue
                    if data['range'] < min_distance:
                        min_distance = data['range']
                elif isinstance(data, (int, float)) and not math.isnan(data):
                    min_distance = min(min_distance, data)
                    
        return min_distance if min_distance < float('inf') else 3.0

    def should_filter_emergency_commands(self):
        """
        Determine if we should filter emergency commands from controllers.
        Returns True if ONLY sonars detect obstacles (no front depth or water mask issues).
        """
        # Check if filtering is enabled
        if not self.filter_controller_emergency_commands:
            return False
        
        # Check if we have critical emergency conditions (front depth or water mask)
        has_front_depth_emergency = (
            self.use_front_depth and 
            self.front_depth_value is not None and 
            self.front_depth_value < self.front_depth_emergency_threshold
        )
        
        has_water_mask_emergency = (
            self.water_mask_status in ["lost", "small", "multiple"]
        )
        
        # Check if we have sonar obstacles
        has_sonar_obstacles = self.check_sonar_obstacles()
        
        # Filter commands if we have sonar obstacles BUT no critical emergencies
        should_filter = has_sonar_obstacles and not has_front_depth_emergency and not has_water_mask_emergency
        
        if should_filter:
            self.get_logger().debug("Filtering emergency commands - only sonar obstacles detected")
        
        return should_filter

    def filter_emergency_commands(self, cmd_vel, controller_name):
        """
        Filter out emergency-like commands from controllers when only sonars detect obstacles.
        """
        from geometry_msgs.msg import Twist
        
        # Use configured thresholds
        emergency_linear_threshold = self.emergency_linear_threshold
        emergency_angular_threshold = self.emergency_angular_threshold
        
        is_emergency_command = (
            cmd_vel.linear.x < emergency_linear_threshold and 
            abs(cmd_vel.angular.z) > emergency_angular_threshold
        )
        
        if is_emergency_command:
            # Override emergency command with smooth avoidance
            filtered_cmd = Twist()
            
            # Allow moderate forward motion instead of stopping
            filtered_cmd.linear.x = max(0.3, cmd_vel.linear.x)
            
            # Reduce angular velocity to reasonable levels
            if abs(cmd_vel.angular.z) > emergency_angular_threshold:
                # Keep the direction but reduce intensity
                direction = 1.0 if cmd_vel.angular.z > 0 else -1.0
                filtered_cmd.angular.z = direction * min(abs(cmd_vel.angular.z), 0.6)
            else:
                filtered_cmd.angular.z = cmd_vel.angular.z
            
            self.get_logger().info(f"FILTERED {controller_name} emergency command: " +
                                 f"linear {cmd_vel.linear.x:.2f}→{filtered_cmd.linear.x:.2f}, " +
                                 f"angular {cmd_vel.angular.z:.2f}→{filtered_cmd.angular.z:.2f}")
            
            return filtered_cmd
        
        return cmd_vel

    def have_valid_sonar_data(self):
        """Check if we have valid sonar data."""
        valid_count = 0
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        for name, data in self.sonar_data.items():
            if isinstance(data, dict) and 'timestamp' in data:
                if current_time - data['timestamp'] <= 1.0:
                    valid_count += 1
            elif isinstance(data, (int, float)) and not math.isnan(data):
                valid_count += 1
                
        return valid_count > 0

    def have_valid_front_depth_data(self):
        """Check if we have valid front depth data."""
        if not self.use_front_depth or self.front_depth_value is None:
            return False
            
        current_time = self.get_clock().now().nanoseconds / 1e9
        if self.front_depth_timestamp and (current_time - self.front_depth_timestamp) < self.front_depth_timeout:
            return True
            
        return False

    def publish_cmd_vel(self, lin_x, ang_z):
        """Publish velocity commands with safety limits."""
        twist = Twist()
        twist.linear.x = max(-self.max_linear_speed, min(lin_x, self.max_linear_speed))
        twist.angular.z = max(-self.max_angular_speed, min(ang_z, self.max_angular_speed))
        
        current_time = self.get_clock().now().nanoseconds / 1e9
        self.last_cmd_vel = twist
        self.last_cmd_time = current_time
        
        self.cmd_vel_pub.publish(twist)

    def update_cross_track_stats(self, cross_track_error):
        """Update path tracking statistics."""
        if self.cross_track_error_samples == 0:
            self.cross_track_error_avg = abs(cross_track_error)
        else:
            alpha = 0.05
            self.cross_track_error_avg = (1-alpha) * self.cross_track_error_avg + alpha * abs(cross_track_error)
            
        self.cross_track_error_samples += 1
        
        if self.cross_track_error_samples % 10 == 0:
            msg = String()
            msg.data = f"CTE: {cross_track_error:.3f}m, Avg: {self.cross_track_error_avg:.3f}m"
            self.tracking_error_pub.publish(msg)

    def print_statistics(self):
        """Print regular statistics."""
        if self.current_state == ControlState.IDLE:
            return
            
        self.get_logger().info(f"Control State: {self.current_state.value}")
        
        if self.current_path:
            self.get_logger().info(f"Path progress: {self.current_path_index}/{len(self.current_path)}")
            if self.cross_track_error_samples > 0:
                self.get_logger().info(f"Path tracking error (avg): {self.cross_track_error_avg:.3f}m")
        
        min_sonar = self.get_min_sonar_distance()
        self.get_logger().info(f"Minimum sonar distance: {min_sonar:.2f}m (DWA-switching only, NO emergency)")
        
        if self.use_front_depth and self.front_depth_value is not None:
            emergency_risk = "EMERGENCY RISK" if self.front_depth_value < self.front_depth_emergency_threshold else "safe"
            self.get_logger().info(f"Water boundary: {self.front_depth_value:.2f}m ({emergency_risk})")
            
        if hasattr(self, 'water_mask_status') and self.water_mask_status:
            emergency_risk = "EMERGENCY RISK" if self.water_mask_status != "normal" else "safe"
            self.get_logger().info(f"Water mask: {self.water_mask_status} ({emergency_risk})")
        
        # Emergency state status
        if self.emergency_stop_active:
            self.get_logger().info("EMERGENCY STATE: ACTIVE")
        else:
            self.get_logger().info("Emergency state: inactive")
        
        self.get_logger().info(f"Current velocity: linear={self.current_linear_velocity:.2f}m/s, " +
                             f"angular={math.degrees(self.current_angular_velocity):.1f}deg/s")
        
        # Warning about external emergency messages
        if min_sonar < 0.3:
            self.get_logger().info("NOTE: If you see 'EMERGENCY: Obstacle' messages, they are from fuzzy/DWA controllers, NOT this control system")

    @staticmethod
    def get_robot_yaw(pose):
        """Extract yaw from quaternion."""
        siny_cosp = 2.0 * (pose.orientation.w * pose.orientation.z + 
                          pose.orientation.x * pose.orientation.y)
        cosy_cosp = 1.0 - 2.0 * (pose.orientation.y**2 + pose.orientation.z**2)
        return math.atan2(siny_cosp, cosy_cosp)
        
    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))

def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = ControlSystem()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node is not None and rclpy.ok():
            node.get_logger().error(f'Unexpected error: {e}')
        else:
            print(f'Unexpected error: {e}', file=sys.stderr)
    finally:
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()