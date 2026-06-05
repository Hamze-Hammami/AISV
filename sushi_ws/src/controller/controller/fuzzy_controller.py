#!/usr/bin/env python3

import math
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from geometry_msgs.msg import Twist

class FuzzyController:
    def __init__(self, max_linear_speed=1.0, max_angular_speed=1.0, goal_tolerance=0.1):
        self.MAX_LINEAR_SPEED = max_linear_speed
        self.MAX_ANGULAR_SPEED = max_angular_speed
        self.goal_tolerance = goal_tolerance
        
        # Path following parameters
        self.path_following_enabled = True
        self.lookahead_distance = 0.5
        self.path_error_threshold = 0.3  # Max acceptable cross-track error
        
        # Obstacle representation from camera data
        self.obstacle_positions = []
        self.obstacle_timestamp = None
        self.obstacle_influence_radius = 2.0  # Detection radius for obstacles
        
        # Logger for debugging
        self.logger = self.get_logger()
        
        # Path tracking statistics
        self.cross_track_error_history = []
        self.max_error_history_size = 10

        # Setup fuzzy control system
        self.setup_fuzzy_control()
    
    # ============ COMPATIBILITY STUB METHODS ============
    # These methods exist only for API compatibility with the control_system
    # They do not use or process sonar data in any way
    
    def update_sonar_data(self, sonar_data, min_obstacle_distance=None):
        """COMPATIBILITY STUB - Fuzzy controller ignores all sonar data."""
        pass
        
    def get_min_sonar_distance(self):
        """COMPATIBILITY STUB - Fuzzy controller doesn't use sonar data."""
        return 999.0  # Return large value to indicate "no obstacle"
        
    def get_front_sonar_distance(self):
        """COMPATIBILITY STUB - Fuzzy controller doesn't use sonar data."""
        return 999.0  # Return large value to indicate "no obstacle"
        
    def get_left_right_readings(self):
        """COMPATIBILITY STUB - Fuzzy controller doesn't use sonar data."""
        return 999.0, 999.0  # Return large values to indicate "no obstacle"
        
    def get_front_reading(self):
        """COMPATIBILITY STUB - Fuzzy controller doesn't use sonar data."""
        return 999.0  # Return large value to indicate "no obstacle"
        
    def check_front_obstacle(self, threshold=0.4):
        """COMPATIBILITY STUB - Fuzzy controller doesn't use sonar data."""
        return False  # Always return False (no obstacle)
    # ============ END COMPATIBILITY STUBS ============

    def update_obstacle_positions(self, obstacle_positions):
        """Update obstacle positions from vision system with timestamp."""
        self.obstacle_positions = obstacle_positions
        self.obstacle_timestamp = self.get_current_time()
        
    def get_current_time(self):
        """Get current time in seconds (implementation dependent)."""
        # In a real ROS node, this would use self.get_clock().now().nanoseconds / 1e9
        # For testing, we'll use a simple approach
        import time
        return time.time()
        
    def update_path_parameters(self, lookahead_distance=None, path_error_threshold=None):
        """Update path following parameters dynamically."""
        if lookahead_distance is not None:
            self.lookahead_distance = lookahead_distance
        if path_error_threshold is not None:
            self.path_error_threshold = path_error_threshold

    def setup_fuzzy_control(self):
        """Setup fuzzy control system for path following with obstacle awareness."""
        # ──────────────────── 1.  UNIVERSES ────────────────────
        self.distance           = ctrl.Antecedent(np.arange(0, 5,   0.1), 'distance')
        self.heading_error      = ctrl.Antecedent(np.arange(-180, 181,  1), 'heading_error')
        self.obstacle_proximity = ctrl.Antecedent(np.arange(0, 3.1, 0.1), 'obstacle_proximity')
        self.cross_track_error  = ctrl.Antecedent(np.arange(0, 2.1, 0.1), 'cross_track_error')

        self.speed    = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'speed')
        self.steering = ctrl.Consequent(np.arange(-1, 1.1, 0.1), 'steering')

        # ──────────────────── 2.  MEMBERSHIP FUNCTIONS ────────────────────

        # Distance
        self.distance['very_close'] = fuzz.trimf(self.distance.universe, [0,   0,   0.3])
        self.distance['close']      = fuzz.trimf(self.distance.universe, [0.2, 0.4, 0.6])
        self.distance['medium']     = fuzz.trimf(self.distance.universe, [0.5, 1.0, 1.5])
        self.distance['far']        = fuzz.trapmf(self.distance.universe, [1.2, 2.0, 5.0, 5.0])

        # Heading error – **EXTREME at ±20° now**
        self.heading_error['extreme_left']  = fuzz.trapmf(self.heading_error.universe, [-180, -180, -30, -20])
        self.heading_error['far_left']      = fuzz.trimf(self.heading_error.universe, [-40, -25, -10])
        self.heading_error['left']          = fuzz.trimf(self.heading_error.universe, [-25, -12,  -5])
        self.heading_error['center']        = fuzz.trimf(self.heading_error.universe, [-10,   0,  10])
        self.heading_error['right']         = fuzz.trimf(self.heading_error.universe, [  5,  12,  25])
        self.heading_error['far_right']     = fuzz.trimf(self.heading_error.universe, [10,  25,  40])
        self.heading_error['extreme_right'] = fuzz.trapmf(self.heading_error.universe, [20,  30, 180, 180])

        # Obstacle proximity
        self.obstacle_proximity['very_close'] = fuzz.trimf(self.obstacle_proximity.universe, [0,   0,   0.4])
        self.obstacle_proximity['close']      = fuzz.trimf(self.obstacle_proximity.universe, [0.3, 0.6, 0.9])
        self.obstacle_proximity['medium']     = fuzz.trimf(self.obstacle_proximity.universe, [0.7, 1.1, 1.5])
        self.obstacle_proximity['far']        = fuzz.trapmf(self.obstacle_proximity.universe, [1.3, 1.8, 3.0, 3.0])

        # Cross-track error
        self.cross_track_error['negligible'] = fuzz.trimf(self.cross_track_error.universe, [0,   0,   0.1])
        self.cross_track_error['small']      = fuzz.trimf(self.cross_track_error.universe, [0.05, 0.2, 0.4])
        self.cross_track_error['medium']     = fuzz.trimf(self.cross_track_error.universe, [0.3,  0.6, 0.9])
        self.cross_track_error['large']      = fuzz.trapmf(self.cross_track_error.universe, [0.7, 1.2, 2.0, 2.0])

        # Speed
        self.speed['stop']      = fuzz.trimf(self.speed.universe, [0,    0,   0.05])
        self.speed['crawl']     = fuzz.trimf(self.speed.universe, [0.02, 0.08, 0.15])
        self.speed['very_slow'] = fuzz.trimf(self.speed.universe, [0.1,  0.18, 0.25])
        self.speed['slow']      = fuzz.trimf(self.speed.universe, [0.2,  0.3,  0.4])
        self.speed['medium']    = fuzz.trimf(self.speed.universe, [0.35, 0.5,  0.65])
        self.speed['fast']      = fuzz.trimf(self.speed.universe, [0.6,  0.8,  1.0])

        # Steering
        self.steering['extreme_left']  = fuzz.trimf(self.steering.universe, [-1.0, -1.0, -0.8])
        self.steering['hard_left']     = fuzz.trimf(self.steering.universe, [-0.9, -0.7, -0.5])
        self.steering['soft_left']     = fuzz.trimf(self.steering.universe, [-0.6, -0.3, -0.1])
        self.steering['center']        = fuzz.trimf(self.steering.universe, [-0.15, 0.0, 0.15])
        self.steering['soft_right']    = fuzz.trimf(self.steering.universe, [0.1,  0.3,  0.6])
        self.steering['hard_right']    = fuzz.trimf(self.steering.universe, [0.5,  0.7,  0.9])
        self.steering['extreme_right'] = fuzz.trimf(self.steering.universe, [0.8,  1.0,  1.0])

        # ──────────────────── 3.  RULE‑BASE ────────────────────
        rules = [
            # SPEED by distance
            ctrl.Rule(self.distance['very_close'], self.speed['very_slow']),
            ctrl.Rule(self.distance['close'],      self.speed['slow']),
            ctrl.Rule(self.distance['medium'],     self.speed['medium']),
            ctrl.Rule(self.distance['far'],        self.speed['fast']),

            # TURN‑IN‑PLACE overrides
            # 0) unconditional stop if |heading_error| ≥ 20°
            ctrl.Rule(self.heading_error['extreme_left'],  self.speed['stop']),
            ctrl.Rule(self.heading_error['extreme_right'], self.speed['stop']),

            # 1) stop if also very close & a bit less extreme
            ctrl.Rule((self.heading_error['extreme_left']  | self.heading_error['far_left'])  &
                    (self.distance['very_close']         | self.distance['close']),
                    self.speed['stop']),
            ctrl.Rule((self.heading_error['extreme_right'] | self.heading_error['far_right']) &
                    (self.distance['very_close']         | self.distance['close']),
                    self.speed['stop']),

            # STEERING from heading
            ctrl.Rule(self.heading_error['extreme_left'],  self.steering['extreme_left']),
            ctrl.Rule(self.heading_error['far_left'],      self.steering['hard_left']),
            ctrl.Rule(self.heading_error['left'],          self.steering['soft_left']),
            ctrl.Rule(self.heading_error['center'],        self.steering['center']),
            ctrl.Rule(self.heading_error['right'],         self.steering['soft_right']),
            ctrl.Rule(self.heading_error['far_right'],     self.steering['hard_right']),
            ctrl.Rule(self.heading_error['extreme_right'], self.steering['extreme_right']),

            # CROSS‑TRACK correction
            ctrl.Rule(self.cross_track_error['negligible'],                  self.steering['center']),
            ctrl.Rule(self.cross_track_error['small']   & self.heading_error['left'],  self.steering['soft_left']),
            ctrl.Rule(self.cross_track_error['small']   & self.heading_error['right'], self.steering['soft_right']),
            ctrl.Rule(self.cross_track_error['medium']  & self.heading_error['left'],  self.steering['hard_left']),
            ctrl.Rule(self.cross_track_error['medium']  & self.heading_error['center'],self.steering['soft_left']),
            ctrl.Rule(self.cross_track_error['medium']  & self.heading_error['right'], self.steering['hard_right']),
            ctrl.Rule(self.cross_track_error['large']   & self.heading_error['left'],  self.steering['extreme_left']),
            ctrl.Rule(self.cross_track_error['large']   & self.heading_error['center'],self.steering['hard_left']),
            ctrl.Rule(self.cross_track_error['large']   & self.heading_error['right'], self.steering['extreme_right']),

            # slow-down for path error
            ctrl.Rule(self.cross_track_error['medium'], self.speed['slow']),
            ctrl.Rule(self.cross_track_error['large'],  self.speed['very_slow']),

            # OBSTACLE avoidance – speed based on camera
            ctrl.Rule(self.obstacle_proximity['very_close'], self.speed['stop']),
            ctrl.Rule(self.obstacle_proximity['close'],      self.speed['very_slow']),
            ctrl.Rule(self.obstacle_proximity['medium'],     self.speed['slow']),
            ctrl.Rule(self.obstacle_proximity['medium'] & self.distance['far'], self.speed['medium']),

            # OBSTACLE avoidance – steering based on camera
            ctrl.Rule(self.obstacle_proximity['very_close'] & self.heading_error['center'], self.steering['extreme_right']),
            ctrl.Rule(self.obstacle_proximity['very_close'] & self.heading_error['left'],   self.steering['extreme_right']),
            ctrl.Rule(self.obstacle_proximity['very_close'] & self.heading_error['right'],  self.steering['extreme_left']),

            ctrl.Rule(self.obstacle_proximity['close']      & self.heading_error['center'], self.steering['extreme_right']),
            ctrl.Rule(self.obstacle_proximity['close']      & self.heading_error['left'],   self.steering['extreme_right']),
            ctrl.Rule(self.obstacle_proximity['close']      & self.heading_error['right'],  self.steering['extreme_left']),

            ctrl.Rule(self.obstacle_proximity['medium']     & self.heading_error['center'], self.steering['hard_right']),
            ctrl.Rule(self.obstacle_proximity['medium']     & self.heading_error['left'],   self.steering['hard_right']),
            ctrl.Rule(self.obstacle_proximity['medium']     & self.heading_error['right'],  self.steering['hard_left']),

            # path/obstacle combined
            ctrl.Rule(self.cross_track_error['large'] & self.obstacle_proximity['far'],    self.steering['extreme_left']),
            ctrl.Rule(self.cross_track_error['large'] & self.obstacle_proximity['medium'], self.steering['hard_left']),
            ctrl.Rule(self.cross_track_error['large'] & self.obstacle_proximity['close'],  self.steering['soft_left']),
        ]

        self.control_system     = ctrl.ControlSystem(rules)
        self.control_simulation = ctrl.ControlSystemSimulation(self.control_system)

    def get_min_obstacle_distance(self, current_pose):
        """
        Calculate minimum distance to obstacles detected by vision (camera).
        Returns the minimum distance and angle to the closest obstacle.
        """
        if not self.obstacle_positions or current_pose is None:
            return float('inf'), 0.0
            
        min_distance = float('inf')
        min_angle = 0.0
        
        robot_x = current_pose.position.x
        robot_y = current_pose.position.y
        robot_yaw = self.get_robot_yaw(current_pose)
        
        for obstacle in self.obstacle_positions:
            if len(obstacle) < 2:
                continue
                
            # Calculate distance to obstacle
            dx = obstacle[0] - robot_x
            dy = obstacle[1] - robot_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate angle to obstacle relative to robot heading
            obstacle_angle = math.atan2(dy, dx)
            relative_angle = self.normalize_angle(obstacle_angle - robot_yaw)
            
            # Only consider obstacles in front of the robot (within ±120 degrees)
            if abs(relative_angle) < 2*math.pi/3 and distance < min_distance:
                min_distance = distance
                min_angle = relative_angle
                
        return min_distance, min_angle

    def compute_path_cross_track_error(self, current_pose, path_points):
        """
        Calculate cross-track error (perpendicular distance to the path).
        Returns the signed error (positive = right of path, negative = left of path).
        """
        if not path_points or len(path_points) < 2 or current_pose is None:
            return None
            
        # Find the closest line segment in the path
        robot_pos = np.array([current_pose.position.x, current_pose.position.y])
        min_dist = float('inf')
        closest_segment = None
        segment_idx = 0
        closest_t = 0.0
        closest_point = None
        
        for i in range(len(path_points) - 1):
            p1 = np.array([path_points[i][0], path_points[i][1]])
            p2 = np.array([path_points[i+1][0], path_points[i+1][1]])
            
            # Vector from p1 to p2
            v_path = p2 - p1
            path_length = np.linalg.norm(v_path)
            
            if path_length < 0.01:  # Skip very short segments
                continue
                
            # Vector from p1 to robot
            v_robot = robot_pos - p1
            
            # Project robot onto path segment
            t = np.dot(v_robot, v_path) / (path_length * path_length)
            t = max(0, min(1, t))  # Clamp to segment
            
            # Nearest point on path segment
            nearest = p1 + t * v_path
            
            # Distance to nearest point
            dist = np.linalg.norm(robot_pos - nearest)
            
            if dist < min_dist:
                min_dist = dist
                closest_segment = (p1, p2)
                segment_idx = i
                closest_t = t
                closest_point = nearest
        
        if closest_segment is None:
            return None
            
        # Calculate signed cross-track error
        p1, p2 = closest_segment
        path_direction = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        path_direction = path_direction / np.linalg.norm(path_direction)
        
        # Vector perpendicular to path (right-hand side)
        perp_direction = np.array([-path_direction[1], path_direction[0]])
        
        # Vector from closest point to robot
        error_vector = robot_pos - closest_point
        
        # Sign of cross-track error
        signed_error = np.dot(error_vector, perp_direction)
        
        # Update error history for trend analysis
        self.update_error_history(signed_error * min_dist)
        
        # Return signed error as a single value
        return signed_error
        
    def update_error_history(self, error):
        """Update the cross-track error history for trend detection."""
        self.cross_track_error_history.append(error)
        
        # Keep history limited to max size
        if len(self.cross_track_error_history) > self.max_error_history_size:
            self.cross_track_error_history.pop(0)
            
    def get_error_trend(self):
        """Analyze the cross-track error trend to detect if it's improving or worsening."""
        if len(self.cross_track_error_history) < 3:
            return 0.0  # Not enough history to determine trend
            
        # Calculate trend as average rate of change
        changes = [abs(self.cross_track_error_history[i]) - abs(self.cross_track_error_history[i-1]) 
                  for i in range(1, len(self.cross_track_error_history))]
        
        # Negative trend means error is decreasing (improving)
        # Positive trend means error is increasing (worsening)
        return sum(changes) / len(changes)
        
    def compute_control(self, current_pose, target_pose, path_points=None):
        """Compute control commands using fuzzy logic."""
        # Calculate heading error from poses
        dx = target_pose.position.x - current_pose.position.x
        dy = target_pose.position.y - current_pose.position.y
        
        # Use target's x value as angle if it's from sushi (in degrees)
        if hasattr(target_pose, 'is_angle') and target_pose.is_angle:
            heading_error_deg = target_pose.position.x  # Already in degrees from sushi
        else:
            # Calculate heading error normally
            current_yaw = self.get_robot_yaw(current_pose)
            target_yaw = math.atan2(dy, dx)
            heading_error = self.normalize_angle(target_yaw - current_yaw)
            heading_error_deg = math.degrees(heading_error)

        # Get minimum distance to vision-detected obstacles (camera only)
        min_obstacle_distance, obstacle_angle = self.get_min_obstacle_distance(current_pose)
        
        # Calculate cross-track error if path points provided
        cross_track_error = 0.0
        
        if path_points and len(path_points) >= 2:
            cross_track_result = self.compute_path_cross_track_error(current_pose, path_points)
            if cross_track_result is not None:
                cross_track_error = abs(cross_track_result)  # Use absolute value for fuzzy

        # Fuzzy control
        try:
            # Always provide values for all antecedents
            self.control_simulation.input['distance'] = 0.0  # Distance is zero for angular-only control
            self.control_simulation.input['heading_error'] = heading_error_deg
            self.control_simulation.input['obstacle_proximity'] = min_obstacle_distance
            
            # Always set cross_track_error (default to smallest value if not available)
            # This ensures the antecedent always has a value
            if self.cross_track_error.universe.size > 0:
                min_cte_value = min(self.cross_track_error.universe)
                max_cte_value = max(self.cross_track_error.universe)
                # Cap cross-track error to range of the universe
                capped_error = min(max(cross_track_error, min_cte_value), max_cte_value)
                self.control_simulation.input['cross_track_error'] = capped_error
            
            # Compute fuzzy control
            self.control_simulation.compute()
            f_speed = self.control_simulation.output['speed']
            f_steer = self.control_simulation.output['steering'] 
        except Exception as e:
            self.logger.error(f"Fuzzy control error: {e}")
            # Fall back to simple proportional control
            f_speed = 0.0  # No linear speed for angular-only control
            f_steer = heading_error / math.pi  # Simple proportional control
            
            # Log detailed info for debugging
            self.logger.warn(f"Fallback control: heading_error={heading_error_deg:.2f}, " +
                           f"min_distance={min_obstacle_distance:.2f}, cross_track_error={cross_track_error:.2f}")

        # Enhanced obstacle avoidance - using camera data
        if min_obstacle_distance < self.obstacle_influence_radius:
            # Use obstacle angle to determine avoidance direction
            if obstacle_angle > 0:  # Obstacle on the right
                # Steer left more aggressively
                f_steer = max(-1.0, f_steer - 0.4)
            else:  # Obstacle on the left
                # Steer right more aggressively
                f_steer = min(1.0, f_steer + 0.4)
            
            # Reduce speed based on obstacle proximity
            if min_obstacle_distance < 1.0:
                f_speed *= 0.5  # 50% speed reduction when close to obstacle
            elif min_obstacle_distance < 1.5:
                f_speed *= 0.7  # 30% speed reduction when moderately close
                    
        # Special case: if we have path tracking error and it's not improving, boost steering
        if path_points is not None and cross_track_error > 0.5:
            error_trend = self.get_error_trend()
            # If error is not decreasing or increasing
            if error_trend >= 0.0:
                # Increase steering correction based on current direction and error magnitude
                steering_boost = min(0.3, cross_track_error * 0.2)  # Cap the boost
                
                if path_points and len(path_points) >= 2:
                    cross_track_result = self.compute_path_cross_track_error(current_pose, path_points)
                    if cross_track_result is not None:
                        signed_error = cross_track_result
                        
                        if signed_error > 0:  # Robot is to the right of path, steer left
                            f_steer = max(-1.0, f_steer - steering_boost)
                        else:  # Robot is to the left of path, steer right
                            f_steer = min(1.0, f_steer + steering_boost)
                    
                # Also reduce speed proportional to cross-track error
                speed_reduction = min(0.5, cross_track_error * 0.3)
                f_speed = max(0.1, f_speed * (1.0 - speed_reduction))
                        
        # Create and return twist command
        twist = Twist()
        twist.linear.x = f_speed * self.MAX_LINEAR_SPEED
        twist.angular.z = f_steer * self.MAX_ANGULAR_SPEED

        # Get the cross-track error for return (signed if available)
        return_cte = 0.0
        if path_points and len(path_points) >= 2:
            cross_track_result = self.compute_path_cross_track_error(current_pose, path_points)
            if cross_track_result is not None:
                return_cte = cross_track_result

        return twist, False, return_cte  # Goal not reached for angular-only control
    
    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]."""
        return math.atan2(math.sin(angle), math.cos(angle))
        
    def get_logger(self):
        """Simple logger interface for stand-alone operation."""
        class SimpleLogger:
            @staticmethod
            def warn(msg):
                print(f"WARNING: {msg}")
                
            @staticmethod
            def info(msg):
                print(f"INFO: {msg}")
                
            @staticmethod
            def error(msg):
                print(f"ERROR: {msg}")
                
            @staticmethod
            def debug(msg):
                print(f"DEBUG: {msg}")
                
        return SimpleLogger()

    @staticmethod
    def get_robot_yaw(pose):
        """Extract yaw from quaternion."""
        siny_cosp = 2.0 * (pose.orientation.w * pose.orientation.z + 
                          pose.orientation.x * pose.orientation.y)
        cosy_cosp = 1.0 - 2.0 * (pose.orientation.y**2 + pose.orientation.z**2)
        return math.atan2(siny_cosp, cosy_cosp)