#!/usr/bin/env python3

import math
import numpy as np
from geometry_msgs.msg import Twist, Quaternion

class DWAController:
    def __init__(self,
                 max_speed=4.0,
                 min_speed=2.0,
                 max_yaw_rate=1.0,
                 max_accel=2.0,
                 max_yaw_accel=1.5,
                 velocity_resolution=0.1,
                 yaw_rate_resolution=0.1,
                 dt=0.1,
                 predict_time=1.0,
                 heading_weight=0.8,
                 distance_weight=0.2,
                 obstacle_weight=0.1):
        
        # Ensure min_speed is non-negative
        self.min_speed_forward = max(0.0, min_speed)
        
        self.config = {
            'max_speed': max_speed,
            'min_speed': 0.0,  # Allow low speeds but never reverse
            'max_yaw_rate': max_yaw_rate,
            'max_accel': max_accel,
            'max_yaw_accel': max_yaw_accel,
            'velocity_resolution': velocity_resolution,
            'yaw_rate_resolution': yaw_rate_resolution,
            'dt': dt,
            'predict_time': predict_time,
            'heading_weight': heading_weight,
            'distance_weight': distance_weight,
            'obstacle_weight': obstacle_weight,
            'safe_distance': 0.5,     # Minimum safety distance
            'obstacle_range': 4.0,    # Detection range for obstacles
            'goal_attraction': 1.5,   # New parameter for goal attraction strength
            'path_bias': 0.5,         # New parameter for path following bias
            'emergency_threshold': 0.4, # Threshold for emergency maneuvers
            'allow_reversing': False,   # Flag to control reversing behavior
            'front_depth_threshold': 1.8, # Increased threshold from 1.2m to 1.8m
            'front_depth_weight': 1.5    # Weight for front depth in scoring (higher than obstacles)
        }
        
        # Initialize sonar data with timestamps
        self.sonar_data = {}
        self.sonar_timestamp = None
        
        # Front depth data (water boundary)
        self.front_depth_value = None
        self.front_depth_timestamp = None
        
        # Add water mask status tracking
        self.water_mask_status = "normal"  # Can be "normal", "small", "multiple", "lost"
        self.water_mask_timestamp = None
        
        # Obstacle confidence tracking
        self.obstacle_positions = []
        self.obstacle_timestamp = None
        
        # Previous motion commands for continuity
        self.prev_velocity = 0.0
        self.prev_yaw_rate = 0.0
        
        # Goal and path information
        self.path_points = None
        self.path_timestamp = None
        
        # Emergency recovery state
        self.in_emergency = False
        self.emergency_start_time = None
        self.emergency_duration = 2.0  # seconds
        
        # Add emergency turn direction tracking
        self.emergency_turn_direction = None
        
        # Debug data
        self.last_trajectory = None
        self.last_score = None
        
        # Logger for debugging
        self.logger = self.get_logger()

    def update_sensor_data(self, sonar_readings, depth_points=None):
        """Update obstacle data from sensors with timing info."""
        # Clean sonar data (remove NaN values)
        cleaned_data = {}
        for name, reading in sonar_readings.items():
            if reading is not None and not math.isnan(reading):
                cleaned_data[name] = reading
        
        # Store with timestamp
        self.sonar_data = cleaned_data
        self.sonar_timestamp = self.get_current_time()
        
        # Update obstacle positions from depth points (if provided)
        if depth_points is not None and len(depth_points) > 0:
            self.obstacle_positions = depth_points
            self.obstacle_timestamp = self.get_current_time()
    
    def update_front_depth(self, front_depth_value):
        """Update front depth (water boundary) data with special handling for zero values."""
        if front_depth_value is not None and not math.isnan(front_depth_value):
            if front_depth_value <= 0.01:
                self.logger.warn("Water boundary zero or near-zero detected - treating as critical emergency")
                self.front_depth_value = 0.01  # Small non-zero to avoid division by zero
            else:
                self.front_depth_value = front_depth_value
            self.front_depth_timestamp = self.get_current_time()
            self.logger.debug(f"Updated front depth: {self.front_depth_value:.2f}m")
        else:
            self.logger.debug("Invalid front depth value received (NaN or None)")

    def update_water_mask_status(self, status):
        """Update the water mask status for specialized avoidance behaviors."""
        self.water_mask_status = status
        self.water_mask_timestamp = self.get_current_time()
        self.logger.info(f"Updated water mask status: {status}")

    def update_path_data(self, path_points):
        """Update path data for better goal-directed navigation."""
        if path_points and len(path_points) > 1:
            self.path_points = path_points
            self.path_timestamp = self.get_current_time()
        else:
            self.path_points = None

    def get_current_time(self):
        """Get current time in seconds."""
        import time
        return time.time()

    def compute_control(self, current_pose, goal_pose=None, with_goal=True):
        """Compute optimal velocity and yaw rate with enhanced water mask behavior."""
        # Initialize robot state [x, y, theta, v, omega]
        x = [
            current_pose.position.x,
            current_pose.position.y,
            self.get_yaw(current_pose),
            max(0.0, self.prev_velocity),  # Ensure non-negative velocity
            self.prev_yaw_rate
        ]

        # Set goal based on mode
        if with_goal and goal_pose:
            goal = [goal_pose.position.x, goal_pose.position.y]
        else:
            # Without goal, create a virtual goal ahead of robot
            dist_ahead = 2.0  # Look 2 meters ahead
            goal = [x[0] + dist_ahead * math.cos(x[2]), 
                x[1] + dist_ahead * math.sin(x[2])]

        # Log debug info
        self.logger.debug(f"Current position: ({x[0]:.2f}, {x[1]:.2f}), heading: {math.degrees(x[2]):.1f}°")
        if with_goal and goal_pose:
            self.logger.debug(f"Goal position: ({goal[0]:.2f}, {goal[1]:.2f})")
        
        # Calculate distance to goal
        goal_distance = math.sqrt((goal[0] - x[0])**2 + (goal[1] - x[1])**2)
        
        # Initialize goal_reached flag here to ensure it's always defined
        goal_reached = goal_distance < 0.1
        
        # Check water mask status for special behaviors
        current_time = self.get_current_time()
        is_mask_status_fresh = (self.water_mask_timestamp is not None and 
                            (current_time - self.water_mask_timestamp) < 1.0)
        
        if is_mask_status_fresh:
            # Handle special water mask conditions
            if self.water_mask_status == "lost":
                # Lost mask - back up while turning
                self.logger.warn("Water mask LOST - backing up while turning")
                twist = Twist()
                twist.linear.x = -0.2  # Slow reverse
                twist.angular.z = 0.5  # Moderate turn
                return twist, False
                
            elif self.water_mask_status == "multiple":
                # Multiple contours - turn in place
                self.logger.warn("Multiple water contours detected - turning in place")
                twist = Twist()
                twist.linear.x = 0.0  # No forward motion
                # Determine turn direction based on goal
                goal_angle = math.atan2(goal[1] - x[1], goal[0] - x[0])
                angle_diff = self.normalize_angle(goal_angle - x[2])
                twist.angular.z = 0.5 if angle_diff > 0 else -0.5  # Turn toward goal
                return twist, False
                
            elif self.water_mask_status == "small":
                # Small mask - slow down but keep moving
                self.logger.warn("Water mask SMALL - proceeding with caution")
                # Continue with normal processing but we'll scale down velocity later
                # Let the normal DWA algorithm run but with reduced speed
        
        # Check front depth (water boundary) - separate handling
        front_depth_emergency = False
        if self.front_depth_value is not None:
            current_time = self.get_current_time()
            if self.front_depth_timestamp and (current_time - self.front_depth_timestamp) < 2.0:
                # Invalid mask or below threshold triggers emergency
                if self.front_depth_value <= self.config['front_depth_threshold']:
                    front_depth_emergency = True
                    self.logger.warn(f"Water boundary close: {self.front_depth_value:.2f}m vs threshold {self.config['front_depth_threshold']}m")
        
        # Handle emergencies
        if front_depth_emergency:
            # For small mask, we'll continue with caution instead of full emergency stop
            if self.water_mask_status == "small":
                self.logger.warn("Water boundary close with small mask - proceeding with caution")
                # Continue with normal algorithm but with limited speed
            else:
                self.in_emergency = True
                self.emergency_start_time = self.get_current_time()
                return self.get_emergency_maneuver(x, goal, front_depth_emergency), False
        
        # Check for regular obstacle emergencies (only if no front depth emergency)
        min_sonar = self.get_min_sonar_distance()
        obstacle_emergency = False
        if not front_depth_emergency and min_sonar < self.config['emergency_threshold']:
            obstacle_emergency = True
            self.logger.warn(f"EMERGENCY: Obstacle very close: {min_sonar:.2f}m - initiating emergency maneuver")

        # Handle emergencies
        if obstacle_emergency:
            self.in_emergency = True
            self.emergency_start_time = self.get_current_time()
            return self.get_emergency_maneuver(x, goal, front_depth_emergency), False

        # Handle emergency recovery state
        if self.in_emergency:
            current_time = self.get_current_time()
            if current_time - self.emergency_start_time < self.emergency_duration:
                # Still in emergency recovery
                return self.get_emergency_maneuver(x, goal, front_depth_emergency), False
            else:
                # Exit emergency mode
                self.in_emergency = False
                self.logger.info("Exiting emergency state")

        # Reset emergency turn direction when exiting emergency mode
        if not (front_depth_emergency or obstacle_emergency) and self.in_emergency:
            self.emergency_turn_direction = None
            self.logger.info("Clearing emergency turn direction")

        # Calculate dynamic window based on current state and obstacles
        dynamic_window = self.calculate_dynamic_window(x[3], x[4], front_depth_emergency)
        
        # Initialize best trajectory
        best_u = [0.0, 0.0]
        best_trajectory = None
        best_score = float('-inf')
        
        # Get closest point on path for path-based evaluation
        path_point = None
        if self.path_points and len(self.path_points) > 1:
            path_point = self.find_closest_path_point(x, self.path_points)
        
        # Set velocity sampling ranges based on dynamic window
        # Ensure we're only sampling non-negative velocities
        v_min = max(0.01, dynamic_window[0])
        v_max = dynamic_window[1]
        w_min = dynamic_window[2]
        w_max = dynamic_window[3]
        
        # Safety check for valid velocity ranges
        if v_max <= v_min or w_max <= w_min:
            self.logger.warn("Invalid velocity ranges! Using fallback values.")
            v_min = 0.01
            v_max = max(0.1, min(0.5, self.config['max_speed'] * 0.5))
            w_min = -self.config['max_yaw_rate'] * 0.5
            w_max = self.config['max_yaw_rate'] * 0.5
            
        v_samples = np.arange(v_min, v_max, self.config['velocity_resolution'])
        w_samples = np.arange(w_min, w_max, self.config['yaw_rate_resolution'])
        
        # Safety check for empty velocity samples
        if len(v_samples) == 0 or len(w_samples) == 0:
            self.logger.warn("Empty velocity samples! Using minimum values.")
            v_samples = np.array([0.1])  # Default safe forward speed
            w_samples = np.array([-0.1, 0.0, 0.1])  # Default yaw rate options
        
        # Log the velocity search space
        self.logger.debug(f"Velocity space: v=[{v_samples[0]:.2f}, {v_samples[-1]:.2f}], w=[{w_samples[0]:.2f}, {w_samples[-1]:.2f}]")
        
        # Track all evaluated trajectories for debugging
        all_scores = []
        
        # Evaluate each velocity and yaw rate combination
        for v in v_samples:
            for w in w_samples:
                # Predict trajectory
                trajectory = self.predict_trajectory(x.copy(), v, w)
                
                # Skip if trajectory couldn't be generated (e.g., due to collision)
                if not trajectory or len(trajectory) < 2:
                    continue
                
                # Calculate comprehensive score
                score, subscores = self.evaluate_trajectory(trajectory, goal, path_point)
                
                # Record trajectory score for debugging
                all_scores.append((v, w, score, subscores))
                
                # Check if this is the best trajectory so far
                if score > best_score:
                    best_score = score
                    best_u = [v, w]
                    best_trajectory = trajectory
        
        # If we have no valid trajectory, use a safety fallback
        if best_score == float('-inf') or not best_trajectory:
            self.logger.warn("No valid trajectory found! Using safety fallback.")
            
            # Use a gentle turn with slow forward motion as fallback
            # The turn direction is based on goal position relative to current heading
            goal_angle = math.atan2(goal[1] - x[1], goal[0] - x[0])
            angle_diff = self.normalize_angle(goal_angle - x[2])
            
            # Determine turn direction (positive = left, negative = right)
            turn_direction = 1.0 if angle_diff > 0 else -1.0
            
            # Special handling for invalid water mask
            if self.front_depth_value is not None and self.front_depth_value <= 0.1:
                # Very slow with cautious turn if water mask is invalid
                twist = Twist()
                twist.linear.x = 0.05  # Very slow
                twist.angular.z = turn_direction * 0.3  # Gentle turn
                self.logger.warn("Using extra cautious fallback due to invalid water mask")
                return twist, goal_reached
            else:
                # Normal fallback for other cases
                twist = Twist()
                twist.linear.x = 0.1  # Very slow forward
                twist.angular.z = turn_direction * 0.5  # Moderate turn
            
            # Check if we've reached the goal despite not finding a path
            return twist, goal_reached
        
        # For debugging: log the best trajectory details
        if len(best_trajectory) > 1:
            end_pos = best_trajectory[-1]
            self.logger.debug(f"Best trajectory: v={best_u[0]:.2f}, w={best_u[1]:.2f}, " +
                            f"ends at ({end_pos[0]:.2f}, {end_pos[1]:.2f})")
        
        # Store for later use
        self.last_trajectory = best_trajectory
        self.last_score = best_score
        
        # Update previous commands (used for velocity continuity)
        self.prev_velocity = best_u[0]
        self.prev_yaw_rate = best_u[1]

        # If we're using the small mask, scale down the final velocity
        if is_mask_status_fresh and self.water_mask_status == "small":
            # Scale down velocity to 30% for small masks
            twist = Twist()
            twist.linear.x = best_u[0] * 0.3  # 30% of normal speed
            twist.angular.z = best_u[1]       # Keep original turning rate
            
            # Log the reduced velocity
            self.logger.debug(f"Small mask: Reduced speed to {twist.linear.x:.2f}m/s (30% of {best_u[0]:.2f})")
            return twist, goal_reached
        
        # Create twist message from best controls
        twist = Twist()
        twist.linear.x = best_u[0]  # Will be non-negative due to our constraints
        twist.angular.z = best_u[1]
        
        # Log the final velocity commands for debugging
        self.logger.debug(f"Linear velocity: {twist.linear.x:.2f}m/s, Angular velocity: {twist.angular.z:.2f}rad/s")
        
        # Return final commands and goal reached status
        return twist, goal_reached

    def get_emergency_maneuver(self, state, goal, is_water_emergency=False):
        """Emergency maneuver with consistent turn direction."""
        twist = Twist()
        twist.linear.x = 0.0  # Always zero linear velocity in emergency
        
        # Initialize turn direction if not set
        if self.emergency_turn_direction is None:
            if goal and len(goal) >= 2:
                # Try to turn toward goal
                goal_angle = math.atan2(goal[1] - state[1], goal[0] - state[0])
                angle_diff = self.normalize_angle(goal_angle - state[2])
                self.emergency_turn_direction = 1.0 if angle_diff > 0 else -1.0
            else:
                # Default turn direction based on current state
                self.emergency_turn_direction = -1.0  # Default to right turn
            
            self.logger.info(f"Emergency: Setting {'LEFT' if self.emergency_turn_direction > 0 else 'RIGHT'} turn")
        
        # Use stored turn direction
        turning_rate = 0.8 if is_water_emergency else 0.6
        twist.angular.z = self.emergency_turn_direction * turning_rate
        
        self.logger.info(f"Emergency: Maintaining turn at {twist.angular.z:.2f}rad/s")
        return twist

    def calculate_dynamic_window(self, current_vel, current_yaw_rate, front_depth_emergency=False):
        """
        Calculate dynamic window with obstacle-based constraints.
        If front_depth_emergency is True, force zero linear velocity.
        """
        if front_depth_emergency:
            # Force zero linear velocity for water boundary emergency
            vs = [0.0, 0.0, -self.config['max_yaw_rate'], self.config['max_yaw_rate']]
            vd = [0.0, 0.0, 
                  max(-self.config['max_yaw_rate'], current_yaw_rate - self.config['max_yaw_accel'] * self.config['dt']),
                  min(self.config['max_yaw_rate'], current_yaw_rate + self.config['max_yaw_accel'] * self.config['dt'])]
            self.logger.debug("Front depth emergency: Forcing zero linear velocity in dynamic window")
            return [max(vs[0], vd[0]), min(vs[1], vd[1]), max(vs[2], vd[2]), min(vs[3], vd[3])]

        # Normal dynamic window calculation
        vs = [0.0, self.config['max_speed'], 
              -self.config['max_yaw_rate'], self.config['max_yaw_rate']]
        vd = [
            max(0.0, current_vel - self.config['max_accel'] * self.config['dt']),
            min(self.config['max_speed'], current_vel + self.config['max_accel'] * self.config['dt']),
            max(-self.config['max_yaw_rate'], current_yaw_rate - self.config['max_yaw_accel'] * self.config['dt']),
            min(self.config['max_yaw_rate'], current_yaw_rate + self.config['max_yaw_accel'] * self.config['dt'])
        ]

        min_obstacle_distance = self.get_min_obstacle_distance()
        front_depth_distance = self.get_front_depth_value()
        combined_min_distance = min(min_obstacle_distance, front_depth_distance)
        
        if combined_min_distance < self.config['obstacle_range']:
            if front_depth_distance <= 0.1 and front_depth_distance < min_obstacle_distance:
                max_allowed_speed = 0.05
                self.logger.warn(f"Invalid water mask detected - forcing near-zero speed")
            else:
                normalized_dist = min(1.0, combined_min_distance / self.config['obstacle_range'])
                sigmoid_factor = 1.0 / (1.0 + math.exp(-5.0 * (normalized_dist - 0.5)))
                max_allowed_speed = self.config['max_speed'] * sigmoid_factor
                
                if front_depth_distance < min_obstacle_distance and front_depth_distance < self.config['front_depth_threshold']:
                    front_depth_factor = front_depth_distance / self.config['front_depth_threshold']
                    front_depth_max_speed = self.config['max_speed'] * front_depth_factor * 0.8
                    max_allowed_speed = min(max_allowed_speed, front_depth_max_speed)
                    self.logger.debug(f"Water boundary at {front_depth_distance:.2f}m limits speed to {front_depth_max_speed:.2f}m/s")
            
            max_allowed_speed = max(0.01, max_allowed_speed)
            vs[1] = min(vs[1], max_allowed_speed)
            vd[1] = min(vd[1], max_allowed_speed)
            self.logger.debug(f"Obstacle at {combined_min_distance:.2f}m, max speed reduced to {max_allowed_speed:.2f}m/s")
        
        dw = [
            max(vs[0], vd[0]), min(vs[1], vd[1]),
            max(vs[2], vd[2]), min(vs[3], vd[3])
        ]
        
        return dw

    def predict_trajectory(self, x, v, w):
        """Predict trajectory with proper collision checking for DWA navigation."""
        # Enforce non-negative velocity
        if v < 0.0:
            return None
            
        trajectory = []
        time = 0
        
        # Start with current state
        trajectory.append(x[:3].copy())
        
        # Predict trajectory over time steps
        while time <= self.config['predict_time']:
            # Apply motion model to update state
            x[0] += v * math.cos(x[2]) * self.config['dt']
            x[1] += v * math.sin(x[2]) * self.config['dt']
            x[2] += w * self.config['dt']
            x[2] = self.normalize_angle(x[2])
            
            # Add new point to trajectory
            trajectory.append(x[:3].copy())
            
            # Update simulation time
            time += self.config['dt']
            
            # Check for collision along trajectory
            if self.check_collision(x):
                # Return trajectory up to collision point (for partial scoring)
                # This is important for DWA to evaluate partially valid trajectories
                return trajectory
        
        return trajectory

    def check_collision(self, state):
        """Check if state collides with obstacles using sonar and vision data."""
        # Check sonar-based collisions
        if self.check_sonar_collision(state):
            return True
        
        # Check front depth (water boundary) collision
        if self.check_front_depth_collision(state):
            return True
            
        # Check vision-based obstacles
        if self.obstacle_positions and len(self.obstacle_positions) > 0:
            # Only use fresh obstacle data (within 1 second)
            if self.obstacle_timestamp and (self.get_current_time() - self.obstacle_timestamp) < 1.0:
                for obstacle in self.obstacle_positions:
                    if len(obstacle) >= 2:
                        # Calculate distance to obstacle
                        dx = state[0] - obstacle[0]
                        dy = state[1] - obstacle[1]
                        distance = math.sqrt(dx*dx + dy*dy)
                        
                        # Check if collision
                        if distance < self.config['safe_distance']:
                            return True
                    
        return False
    
    def check_front_depth_collision(self, state):
        """Check if state collides with water boundary using front depth data."""
        if self.front_depth_value is None:
            return False
            
        # Only use fresh front depth data (within 2 seconds)
        current_time = self.get_current_time()
        if self.front_depth_timestamp and (current_time - self.front_depth_timestamp) < 2.0:
            # Get robot's forward vector in world frame
            robot_x, robot_y, robot_theta = state[0], state[1], state[2]
            forward_x = math.cos(robot_theta)
            forward_y = math.sin(robot_theta)
            
            # Calculate position at front depth distance
            boundary_x = robot_x + self.front_depth_value * forward_x
            boundary_y = robot_y + self.front_depth_value * forward_y
            
            # Check if this point is within the safe distance
            # We're using 90% of safe_distance to make water boundary avoidance more conservative
            water_safe_distance = self.config['safe_distance'] * 0.9
            if self.front_depth_value < water_safe_distance:
                return True
                
        return False

    def check_sonar_collision(self, state):
        """Check for collisions using sonar data."""
        if not self.sonar_data:
            return False
            
        # Get robot position and orientation from state
        robot_x, robot_y, robot_theta = state[0], state[1], state[2]
        
        # Check each sonar reading
        for name, distance in self.sonar_data.items():
            # Skip invalid readings
            if math.isnan(distance):
                continue
                
            # Get sonar position and orientation in robot frame
            sonar_pos, sonar_orient = self.get_sonar_position(name, robot_theta)
            
            # Calculate sonar position in world frame
            sonar_world_x = robot_x + sonar_pos[0] * math.cos(robot_theta) - sonar_pos[1] * math.sin(robot_theta)
            sonar_world_y = robot_y + sonar_pos[0] * math.sin(robot_theta) + sonar_pos[1] * math.cos(robot_theta)
            
            # Calculate sonar direction in world frame
            sonar_dir = robot_theta + sonar_orient
            
            # Calculate obstacle position based on sonar reading
            obstacle_x = sonar_world_x + distance * math.cos(sonar_dir)
            obstacle_y = sonar_world_y + distance * math.sin(sonar_dir)
            
            # Calculate distance from robot center to obstacle
            dx = obstacle_x - robot_x
            dy = obstacle_y - robot_y
            center_dist = math.sqrt(dx*dx + dy*dy)
            
            # Check if obstacle is within safety distance
            if center_dist < self.config['safe_distance'] + 0.2:  # Add small buffer
                return True
                
        return False

    def get_min_sonar_by_side(self, side):
        """Get minimum valid sonar reading from specified side."""
        min_distance = float('inf')
        
        for name, distance in self.sonar_data.items():
            if side.lower() in name.lower() and not math.isnan(distance):
                min_distance = min(min_distance, distance)
                
        return min_distance if min_distance < float('inf') else 3.0

    def get_min_obstacle_distance(self):
        """Get minimum distance to any obstacle from sonar or vision."""
        min_distance = float('inf')
        
        # Check sonar data
        for name, distance in self.sonar_data.items():
            if not math.isnan(distance):
                min_distance = min(min_distance, distance)
        
        # Check vision-based obstacles if available and recent
        if (self.obstacle_positions and self.obstacle_timestamp and 
            (self.get_current_time() - self.obstacle_timestamp) < 1.0):
            
            for obstacle in self.obstacle_positions:
                if len(obstacle) >= 2:
                    # We would need current robot position to calculate distance
                    # Since we don't have it here, we'll skip this for now
                    pass
        
        return min_distance if min_distance < float('inf') else self.config['obstacle_range']
    
    def get_front_depth_value(self):
        """
        Get front depth (water boundary) value if available and fresh.
        Returns near-zero for invalid water masks.
        """
        if self.front_depth_value is None:
            return float('inf')
            
        current_time = self.get_current_time()
        if self.front_depth_timestamp and (current_time - self.front_depth_timestamp) < 2.0:
            # If front depth is very small (indicates invalid mask), treat as an emergency
            if self.front_depth_value <= 0.01:
                return 0.01  # Return a very small value, but not zero to avoid div by zero
            return self.front_depth_value
            
        return float('inf')
        
    def get_min_sonar_distance(self):
        """Get minimum distance from sonar data only."""
        min_distance = float('inf')
        
        # Check sonar data
        for name, distance in self.sonar_data.items():
            if not isinstance(distance, (int, float)):
                # Handle dict format
                if isinstance(distance, dict) and 'range' in distance:
                    distance = distance['range']
                else:
                    continue
                    
            if not math.isnan(distance):
                min_distance = min(min_distance, distance)
        
        return min_distance if min_distance < float('inf') else self.config['obstacle_range']

    def evaluate_trajectory(self, trajectory, goal, path_point=None):
        """
        Evaluate trajectory with comprehensive scoring.
        This is the core of DWA - balancing goal-directedness with obstacle avoidance.
        """
        # Safety check for empty trajectory
        if not trajectory or len(trajectory) < 2:
            return float('-inf'), {
                'heading': 0.0,
                'distance': 0.0,
                'velocity': 0.0,
                'clearance': 0.0,
                'path': 0.0,
                'front_depth': 0.0
            }
            
        # Calculate core metrics
        heading_score = self.heading_score(trajectory, goal)
        distance_score = self.distance_score(trajectory, goal)
        velocity_score = self.velocity_score(trajectory)
        clearance_score = self.clearance_score(trajectory)
        
        # Front depth (water boundary) score
        front_depth_score = self.front_depth_score(trajectory)
        
        # Path following score (if path point available)
        path_score = 0.0
        if path_point is not None:
            path_score = self.path_score(trajectory, path_point)
        
        # Calculate weighted combination - these weights are critical for proper navigation
        total_score = (
            self.config['heading_weight'] * heading_score +      # Importance of heading toward goal
            self.config['distance_weight'] * distance_score +    # Importance of getting closer to goal
            self.config['obstacle_weight'] * clearance_score +   # Importance of avoiding obstacles
            0.2 * velocity_score                                # Importance of maintaining speed
        )
        
        # Add front depth score with higher weight (water boundary avoidance is more important)
        if front_depth_score > 0:
            total_score += self.config['front_depth_weight'] * front_depth_score
        
        # Add path bias if following a specific path
        if path_point is not None:
            total_score += self.config['path_bias'] * path_score
        
        # Track subscores for debugging
        subscores = {
            'heading': heading_score,
            'distance': distance_score,
            'velocity': velocity_score,
            'clearance': clearance_score,
            'path': path_score,
            'front_depth': front_depth_score
        }
        
        return total_score, subscores

    def front_depth_score(self, trajectory):
        """
        Score based on maintaining safe distance from water boundary.
        Higher score means safer (further from water boundary).
        """
        if self.front_depth_value is None:
            return 0.0
            
        # Only use fresh front depth data (within 2 seconds)
        current_time = self.get_current_time()
        if not (self.front_depth_timestamp and (current_time - self.front_depth_timestamp) < 2.0):
            return 0.0
            
        # Get final position in trajectory
        final_x, final_y, final_theta = trajectory[-1]
        
        # Calculate forward vector from robot's final pose
        forward_x = math.cos(final_theta)
        forward_y = math.sin(final_theta)
        
        # Simple approach - score based on front depth value
        # Higher score for trajectories that keep us away from the water boundary
        threshold = self.config['front_depth_threshold']  # Now using 1.8m
        
        # Special handling for small mask - less restrictive
        if self.water_mask_status == "small":
            # Allow to get closer with small mask (80% of normal threshold)
            adjusted_threshold = threshold * 0.8
            # Use adjusted threshold for scoring
            if self.front_depth_value >= adjusted_threshold:
                # Beyond threshold, maximum score
                return 1.0
            else:
                # Scale score linearly based on front depth
                score = self.front_depth_value / adjusted_threshold
                score = 1.0 / (1.0 + math.exp(-5.0 * (score - 0.5)))
                return score
        
        # Special handling for invalid water mask (zero/near-zero value)
        if self.front_depth_value <= 0.1:
            # Invalid mask - very low score
            return 0.01  # Very small but positive to avoid multiplication issues
            
        if self.front_depth_value >= threshold:
            # Beyond threshold, maximum score
            return 1.0
        else:
            # Scale score linearly based on front depth
            # 0.0 at 0 distance, 1.0 at threshold distance
            score = self.front_depth_value / threshold
            
            # Apply non-linear scaling to emphasize staying away from boundary
            score = 1.0 / (1.0 + math.exp(-5.0 * (score - 0.5)))
            
            return score

    def heading_score(self, trajectory, goal):
        """
        Score based on final heading relative to goal.
        This evaluates how well the robot will be aligned with the goal after executing the trajectory.
        """
        # Get final pose in trajectory
        final_x, final_y, final_theta = trajectory[-1]
        
        # Calculate desired heading to goal
        dx = goal[0] - final_x
        dy = goal[1] - final_y
        goal_theta = math.atan2(dy, dx)
        
        # Calculate heading error (normalized to [-pi, pi])
        heading_error = abs(self.normalize_angle(goal_theta - final_theta))
        
        # Convert to score (1.0 is perfect alignment, 0.0 is worst)
        score = (math.pi - heading_error) / math.pi
        
        # Apply non-linear scaling for more emphasis on good alignment
        # This ensures the robot prefers trajectories that align well with the goal
        score = 1.0 / (1.0 + math.exp(-5.0 * (score - 0.5)))
        
        return score

    def distance_score(self, trajectory, goal):
        """
        Score based on final distance to goal.
        This evaluates how close the robot will get to the goal after executing the trajectory.
        """
        # Get final position
        final_x, final_y, _ = trajectory[-1]
        
        # Calculate distance to goal
        dx = goal[0] - final_x
        dy = goal[1] - final_y
        dist = math.sqrt(dx*dx + dy*dy)
        
        # Convert to score (1.0 is closest, 0.0 is farthest)
        # Use exponential decay for smoother scaling
        max_dist = 5.0  # Maximum relevant distance
        score = math.exp(-dist / (max_dist / 3.0))
        
        return score

    def velocity_score(self, trajectory):
        """
        Score based on maintaining good velocity.
        This encourages the robot to maintain a reasonable speed.
        """
        # Calculate average velocity over trajectory
        if len(trajectory) < 2:
            return 0.0
            
        total_dist = 0.0
        for i in range(1, len(trajectory)):
            x1, y1, _ = trajectory[i-1]
            x2, y2, _ = trajectory[i]
            dx = x2 - x1
            dy = y2 - y1
            total_dist += math.sqrt(dx*dx + dy*dy)
            
        avg_speed = total_dist / (self.config['dt'] * (len(trajectory) - 1))
        
        # Score is higher for speeds closer to optimal speed (approx. 70% of max)
        optimal_speed = 0.7 * self.config['max_speed']
        speed_diff = abs(avg_speed - optimal_speed)
        speed_score = math.exp(-(speed_diff / optimal_speed)**2)
        
        # Boost score slightly for forward motion to encourage progress
        if avg_speed > 0.1:
            speed_score *= 1.1
        
        return speed_score

    def clearance_score(self, trajectory):
        """
        Score based on clearance from obstacles.
        This evaluates how safely the robot can execute the trajectory without collision.
        """
        # Find minimum clearance along trajectory
        min_clearance = float('inf')
        
        for point in trajectory:
            x, y, _ = point
            
            # Calculate clearance at this point
            clearance = self.calculate_clearance_at_point(x, y)
            min_clearance = min(min_clearance, clearance)
        
        # Convert to score (1.0 is maximum clearance, 0.0 is collision)
        if min_clearance <= 0.0:
            return 0.0
            
        # Calculate normalized score based on safe distance
        score = min(1.0, min_clearance / self.config['safe_distance'] / 2.0)
        
        # Apply non-linear scaling to emphasize safety
        score = 1.0 / (1.0 + math.exp(-5.0 * (score - 0.5)))
        
        return score

    def path_score(self, trajectory, path_point):
        """
        Score based on adherence to provided path.
        This evaluates how well the trajectory follows a predefined path.
        """
        if path_point is None:
            return 0.0
            
        # Calculate average cross-track error along trajectory
        total_error = 0.0
        for point in trajectory:
            x, y, _ = point
            
            # Calculate distance to path
            dx = x - path_point[0]
            dy = y - path_point[1]
            error = math.sqrt(dx*dx + dy*dy)
            
            total_error += error
            
        avg_error = total_error / len(trajectory)
        
        # Convert to score (1.0 is perfect path following, 0.0 is far from path)
        max_error = 2.0  # Maximum error we care about
        score = max(0.0, 1.0 - avg_error / max_error)
        
        # Apply exponential scaling for more emphasis on staying on path
        score = math.exp(-2.0 * (1.0 - score))
        
        return score

    def calculate_clearance_at_point(self, x, y):
        """Calculate clearance (distance to obstacles) at a specific point."""
        min_distance = float('inf')
        
        # Check vision-based obstacles
        if (self.obstacle_positions and self.obstacle_timestamp and 
            (self.get_current_time() - self.obstacle_timestamp) < 1.0):
            
            for obstacle in self.obstacle_positions:
                if len(obstacle) >= 2:
                    ox, oy = obstacle[0], obstacle[1]
                    dist = math.sqrt((x - ox)**2 + (y - oy)**2)
                    min_distance = min(min_distance, dist)
        
        # Check front depth for water boundary
        if self.front_depth_value is not None:
            current_time = self.get_current_time()
            if self.front_depth_timestamp and (current_time - self.front_depth_timestamp) < 2.0:
                # This is a simplification - we're treating the water boundary as 
                # being directly in front of the robot at the measured distance
                # A more accurate approach would require robot pose
                front_depth_dist = self.front_depth_value
                if front_depth_dist < min_distance:
                    min_distance = front_depth_dist
        
        if min_distance == float('inf'):
            # No obstacles detected
            return self.config['obstacle_range']
        else:
            return min_distance

    def find_closest_path_point(self, state, path_points):
        """Find the closest point on the path for guidance."""
        x, y = state[0], state[1]
        
        min_dist = float('inf')
        closest_point = None
        
        for point in path_points:
            if len(point) >= 2:
                dx = point[0] - x
                dy = point[1] - y
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < min_dist:
                    min_dist = dist
                    closest_point = point
        
        return closest_point

    def get_sonar_position(self, name, robot_theta):
        """
        Get the position and orientation of a sonar based on its name.
        Returns position [x,y] and orientation (radians) in robot frame.
        """
        # These values are approximate and should be calibrated
        # for the actual robot configuration
        
        # Default position and orientation (forward)
        position = [0.0, 0.0]  # [x, y] in robot frame
        orientation = 0.0  # radians relative to robot heading
        
        # Adjust based on sonar name
        if 'left_front' in name:
            position = [0.2, 0.3]  # Forward and left
            orientation = math.pi/4  # 45 degrees left
        elif 'left_rear' in name:
            position = [-0.2, 0.3]  # Rear and left
            orientation = 3*math.pi/4  # 135 degrees left
        elif 'right_front' in name:
            position = [0.2, -0.3]  # Forward and right
            orientation = -math.pi/4  # 45 degrees right
        elif 'right_rear' in name:
            position = [-0.2, -0.3]  # Rear and right
            orientation = -3*math.pi/4  # 135 degrees right
        elif 'front' in name:
            position = [0.3, 0.0]  # Forward center
            orientation = 0.0  # Straight ahead
        elif 'rear' in name:
            position = [-0.3, 0.0]  # Rear center
            orientation = math.pi  # Straight behind
        
        return position, orientation

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
    def get_yaw(pose):
        """Extract yaw from quaternion."""
        siny_cosp = 2.0 * (pose.orientation.w * pose.orientation.z +
                          pose.orientation.x * pose.orientation.y)
        cosy_cosp = 1.0 - 2.0 * (pose.orientation.y**2 + pose.orientation.z**2)
        return math.atan2(siny_cosp, cosy_cosp)