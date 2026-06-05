#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, PoseArray, Pose, PoseWithCovarianceStamped, Point
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Range
from visualization_msgs.msg import Marker, MarkerArray
import numpy as np
import math
from scipy.spatial.transform import Rotation
from typing import List, Dict, Tuple, Optional
import tf2_ros
import random
import time
from dataclasses import dataclass

@dataclass
class SonarConfig:
    """Configuration for sonar sensor positioning and orientation."""
    position: Tuple[float, float, float]  # x, y, z in meters
    orientation: Tuple[float, float, float]  # roll, pitch, yaw in degrees
    
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation

class Particle:
    """Represents a single particle in the particle filter."""
    def __init__(self, x, y, theta, weight=1.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight
    
    def to_pose(self):
        """Convert particle to a ROS Pose message."""
        pose = Pose()
        pose.position.x = self.x
        pose.position.y = self.y
        pose.position.z = -0.03  # 1 unit below the ground plane
        
        # Convert theta to quaternion (rotation around z-axis)
        pose.orientation.w = math.cos(self.theta / 2.0)
        pose.orientation.x = 0.0
        pose.orientation.y = 0.0
        pose.orientation.z = math.sin(self.theta / 2.0)
        
        return pose

class ParticleFilter:
    """
    Particle filter for robot localization using obstacles and sonar data.
    """
    def __init__(self, 
                 num_particles=500, 
                 motion_noise=(0.05, 0.05, 0.05),
                 measurement_noise=(0.1, 0.1)):
        # Number of particles
        self.num_particles = num_particles
        
        # Noise parameters
        self.motion_noise = motion_noise  # (x_noise, y_noise, theta_noise)
        self.measurement_noise = measurement_noise  # (range_noise, bearing_noise)
        
        # Particles
        self.particles = []
        
        # Known static obstacles (map frame)
        self.known_obstacles = {}
        
        # Known goals (map frame)
        self.known_goals = {}
        
        # Sonar configurations
        self.sonar_configs = {
            'right_front': SonarConfig((3.81, 0.554, 0.91), (0, 90, 0)),
            'right_rear': SonarConfig((3.81, 0.554, -3.797), (0, 90, 0)),
            'left_front': SonarConfig((-3.81, 0.554, 0.91), (0, -90, 0)),
            'left_rear': SonarConfig((-3.81, 0.554, -3.796), (0, -90, 0))
        }
        
        # Latest sonar readings
        self.sonar_readings = {
            'left_front': None,
            'left_rear': None,
            'right_front': None,
            'right_rear': None
        }
        
        # Effective particle count threshold for resampling
        self.resample_threshold = self.num_particles / 2.0
        
        # Track when resampling was last performed
        self.last_resample_time = time.time()
        self.min_resample_interval = 1.0  # Minimum time between resampling in seconds
        
        # Overall localization quality estimate
        self.localization_quality = 0.5  # Scale from 0 to 1
        
        # Initialization flag
        self.initialized = False
        
        print(f"Particle filter initialized with {num_particles} particles")
    
    def initialize(self, initial_pose, position_uncertainty=(0.5, 0.5), orientation_uncertainty=0.2):
        """
        Initialize particles around an initial pose estimate.
        
        Args:
            initial_pose: Initial pose as (x, y, theta) or Pose message
            position_uncertainty: Uncertainties in x, y position (meters)
            orientation_uncertainty: Uncertainty in orientation (radians)
        """
        # Extract initial pose
        if isinstance(initial_pose, Pose):
            x = initial_pose.position.x
            y = initial_pose.position.y
            
            # Extract theta from quaternion
            qx = initial_pose.orientation.x
            qy = initial_pose.orientation.y
            qz = initial_pose.orientation.z
            qw = initial_pose.orientation.w
            
            # Calculate yaw from quaternion
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            theta = math.atan2(siny_cosp, cosy_cosp)
        else:
            x, y, theta = initial_pose
        
        # Create particles with Gaussian distribution around initial pose
        self.particles = []
        for _ in range(self.num_particles):
            particle_x = random.gauss(x, position_uncertainty[0])
            particle_y = random.gauss(y, position_uncertainty[1])
            particle_theta = random.gauss(theta, orientation_uncertainty)
            
            # Normalize theta to [-pi, pi]
            particle_theta = math.atan2(math.sin(particle_theta), math.cos(particle_theta))
            
            # Create and add particle
            particle = Particle(particle_x, particle_y, particle_theta)
            self.particles.append(particle)
        
        self.initialized = True
        self.update_pose_estimate()
        
        print(f"Particle filter initialized around pose ({x:.2f}, {y:.2f}, {math.degrees(theta):.1f}°)")
    
    def predict(self, odometry_delta):
        """
        Update particles based on odometry motion model.
        
        Args:
            odometry_delta: Change in pose (dx, dy, dtheta) since last update
        """
        if not self.initialized or not self.particles:
            return
        
        dx, dy, dtheta = odometry_delta
        
        for particle in self.particles:
            # Add noise to motion
            noisy_dx = dx + random.gauss(0, self.motion_noise[0])
            noisy_dy = dy + random.gauss(0, self.motion_noise[1])
            noisy_dtheta = dtheta + random.gauss(0, self.motion_noise[2])
            
            # Apply motion in global coordinates
            # First rotate the translation to global frame based on particle's orientation
            global_dx = noisy_dx * math.cos(particle.theta) - noisy_dy * math.sin(particle.theta)
            global_dy = noisy_dx * math.sin(particle.theta) + noisy_dy * math.cos(particle.theta)
            
            # Update particle
            particle.x += global_dx
            particle.y += global_dy
            particle.theta += noisy_dtheta
            
            # Normalize theta to [-pi, pi]
            particle.theta = math.atan2(math.sin(particle.theta), math.cos(particle.theta))
    
    def register_sonar_reading(self, sonar_name, range_value):
        """Register a sonar reading for use in the update step."""
        if sonar_name in self.sonar_readings:
            self.sonar_readings[sonar_name] = range_value
    
    def register_static_obstacle(self, obstacle_id, position):
        """Register a known static obstacle position in the map frame."""
        self.known_obstacles[obstacle_id] = np.array(position)
    
    def register_goal(self, goal_id, position):
        """Register a known goal position in the map frame."""
        self.known_goals[goal_id] = np.array(position)
    
    def _get_expected_obstacle_positions(self, particle, obstacle_positions):
        """
        Convert known obstacle positions from map frame to the particle's local frame.
        
        Args:
            particle: The particle whose frame obstacles will be transformed to
            obstacle_positions: Dictionary of obstacle positions in map frame
            
        Returns:
            Dictionary of expected obstacle positions in particle's local frame
        """
        expected_positions = {}
        
        # Create rotation matrix from particle orientation
        c = math.cos(particle.theta)
        s = math.sin(particle.theta)
        R = np.array([[c, s], [-s, c]])  # 2D rotation matrix
        
        # Particle position as numpy array
        particle_pos = np.array([particle.x, particle.y])
        
        for obstacle_id, position in obstacle_positions.items():
            # Extract 2D position
            obstacle_pos = position[:2]
            
            # Vector from particle to obstacle in map frame
            map_vector = obstacle_pos - particle_pos
            
            # Rotate vector to particle frame
            local_vector = np.dot(R, map_vector)
            
            # Store expected position
            expected_positions[obstacle_id] = local_vector
        
        return expected_positions
    
    def _get_expected_sonar_readings(self, particle):
        """
        Calculate expected sonar readings for a particle based on known obstacles.
        
        Args:
            particle: Particle to calculate readings for
            
        Returns:
            Dictionary of expected readings for each sonar
        """
        expected_readings = {}
        
        # Transform known obstacles to particle frame
        local_obstacles = self._get_expected_obstacle_positions(particle, self.known_obstacles)
        
        for sonar_name, config in self.sonar_configs.items():
            # Get sonar position and orientation in robot frame
            x_sonar, y_sonar, _ = config.position
            _, _, yaw_deg = config.orientation
            yaw_rad = math.radians(yaw_deg)
            
            # Sonar direction vector in robot frame
            sonar_dir = np.array([math.cos(yaw_rad), math.sin(yaw_rad)])
            
            # Find minimum distance to any obstacle in sonar's field of view
            min_range = float('inf')
            
            for obstacle_pos in local_obstacles.values():
                # Vector from sonar to obstacle in robot frame
                to_obstacle = obstacle_pos - np.array([x_sonar, y_sonar])
                
                # Distance to obstacle
                distance = np.linalg.norm(to_obstacle)
                
                # Direction to obstacle (normalized)
                if distance > 0:
                    direction = to_obstacle / distance
                else:
                    continue
                
                # Check if obstacle is in sonar's field of view
                # Dot product gives cosine of angle between vectors
                angle_cos = np.dot(sonar_dir, direction)
                
                # Field of view is typically 30 degrees for sonars, so cos(15 deg) = 0.9659
                if angle_cos > 0.9659:  # Within 15 degrees of sonar direction
                    if distance < min_range:
                        min_range = distance
            
            # If we found an obstacle in range, store the expected reading
            if min_range < float('inf'):
                expected_readings[sonar_name] = min_range
        
        return expected_readings
    
    def update_weights(self, detected_obstacles, use_sonar=True):
        """
        Update particle weights based on detected obstacles and sonar readings.
        
        Args:
            detected_obstacles: List of detected obstacle positions in robot frame
            use_sonar: Whether to include sonar readings in update
        """
        if not self.initialized or not self.particles:
            return
        
        # Skip update if we don't have known obstacles or detections
        if not self.known_obstacles and not detected_obstacles and not any(self.sonar_readings.values()):
            return
        
        total_weight = 0.0
        
        for particle in self.particles:
            # Initialize particle weight
            particle_weight = 1.0
            
            # Part 1: Update based on detected obstacles
            if self.known_obstacles and detected_obstacles:
                # Get expected obstacle positions in particle's frame
                expected_positions = self._get_expected_obstacle_positions(particle, self.known_obstacles)
                
                # Match detected obstacles to expected positions
                for detection in detected_obstacles.poses:
                    # Get detection position in robot frame
                    detection_pos = np.array([detection.position.x, detection.position.y])
                    
                    # Find closest expected obstacle
                    min_dist = float('inf')
                    
                    for expected_pos in expected_positions.values():
                        # Calculate distance between expected and detected positions
                        dist = np.linalg.norm(detection_pos - expected_pos)
                        min_dist = min(min_dist, dist)
                    
                    # Update particle weight based on distance error
                    # Use Gaussian probability density: p(z|x) ∝ exp(-error²/(2*σ²))
                    if min_dist < float('inf'):
                        error_prob = math.exp(-0.5 * (min_dist / self.measurement_noise[0])**2)
                        particle_weight *= error_prob
            
            # Part 2: Update based on sonar readings
            if use_sonar and any(self.sonar_readings.values()):
                # Get expected sonar readings for this particle
                expected_readings = self._get_expected_sonar_readings(particle)
                
                # Compare with actual readings
                for sonar_name, actual_reading in self.sonar_readings.items():
                    if actual_reading is None:
                        continue
                        
                    expected_reading = expected_readings.get(sonar_name)
                    if expected_reading is None:
                        continue
                    
                    # Calculate error
                    error = abs(expected_reading - actual_reading)
                    
                    # Update weight based on error
                    error_prob = math.exp(-0.5 * (error / self.measurement_noise[0])**2)
                    particle_weight *= error_prob
            
            # Ensure weight doesn't underflow to zero
            particle_weight = max(particle_weight, 1e-10)
            
            # Update particle weight
            particle.weight = particle_weight
            total_weight += particle_weight
        
        # Normalize weights
        if total_weight > 0:
            for particle in self.particles:
                particle.weight /= total_weight
        
        # Update pose estimate and check if resampling is needed
        self.update_pose_estimate()
        
        # Calculate effective sample size
        n_eff = 1.0 / sum(p.weight**2 for p in self.particles)
        
        # Check if resampling should be performed
        current_time = time.time()
        time_since_resample = current_time - self.last_resample_time
        
        if n_eff < self.resample_threshold and time_since_resample >= self.min_resample_interval:
            self.resample()
            self.last_resample_time = current_time
    
    def resample(self):
        """Resample particles based on their weights."""
        if not self.particles:
            return
            
        # Extract weights and particles
        weights = np.array([p.weight for p in self.particles])
        
        # Low variance resampling
        new_particles = []
        n = len(self.particles)
        r = random.uniform(0, 1.0/n)
        c = weights[0]
        i = 0
        
        for m in range(n):
            u = r + m/n
            while u > c:
                i += 1
                if i >= n:
                    i = n - 1
                c += weights[i]
            
            # Create new particle based on selected particle
            selected = self.particles[i]
            new_particle = Particle(
                selected.x, 
                selected.y, 
                selected.theta,
                1.0/n  # Reset weight
            )
            
            # Add small random noise to avoid particle depletion
            new_particle.x += random.gauss(0, self.motion_noise[0] * 0.1)
            new_particle.y += random.gauss(0, self.motion_noise[1] * 0.1)
            new_particle.theta += random.gauss(0, self.motion_noise[2] * 0.1)
            new_particle.theta = math.atan2(math.sin(new_particle.theta), math.cos(new_particle.theta))
            
            new_particles.append(new_particle)
        
        # Replace old particles with new ones
        self.particles = new_particles
        
        # Update pose estimate
        self.update_pose_estimate()
        
        print(f"Resampled {n} particles")
    
    def update_pose_estimate(self):
        """Update the best pose estimate and covariance from particles."""
        if not self.particles:
            return
        
        # Methods for pose estimation:
        # 1. Weighted average (can be unstable with multi-modal distributions)
        # 2. Highest weight particle (can be noisy)
        # 3. Weighted average of cluster around highest weight particle (more robust)
        
        # For this implementation, we'll use the weighted average approach
        total_x = 0.0
        total_y = 0.0
        total_cos_theta = 0.0
        total_sin_theta = 0.0
        
        for particle in self.particles:
            total_x += particle.weight * particle.x
            total_y += particle.weight * particle.y
            total_cos_theta += particle.weight * math.cos(particle.theta)
            total_sin_theta += particle.weight * math.sin(particle.theta)
        
        # Calculate average pose
        avg_x = total_x
        avg_y = total_y
        avg_theta = math.atan2(total_sin_theta, total_cos_theta)
        
        # Calculate covariance matrix
        covariance = np.zeros((3, 3))
        for particle in self.particles:
            dx = particle.x - avg_x
            dy = particle.y - avg_y
            dtheta = math.atan2(math.sin(particle.theta - avg_theta), math.cos(particle.theta - avg_theta))
            
            diff = np.array([dx, dy, dtheta])
            covariance += particle.weight * np.outer(diff, diff)
        
        # Update best pose and covariance
        self.best_pose = (avg_x, avg_y, avg_theta)
        self.pose_covariance = covariance
        
        # Update localization quality
        # Based on particle dispersion - lower dispersion = higher quality
        position_variance = covariance[0, 0] + covariance[1, 1]
        orientation_variance = covariance[2, 2]
        
        # Normalize variances to a quality metric between 0 and 1
        position_quality = math.exp(-position_variance / 1.0)  # Scale factor 1.0 determines sensitivity
        orientation_quality = math.exp(-orientation_variance / 0.5)  # Scale factor 0.5 determines sensitivity
        
        # Combine position and orientation quality
        self.localization_quality = 0.7 * position_quality + 0.3 * orientation_quality
    
    def get_best_pose(self):
        """
        Get the best pose estimate from the particle distribution.
        
        Returns:
            (pose, covariance, quality): 
                pose as (position, orientation_quaternion)
                covariance matrix (3x3)
                localization quality from 0 to 1
        """
        if not self.best_pose:
            return None, None, 0.0
        
        x, y, theta = self.best_pose
        
        # Convert to position and quaternion format
        position = np.array([x, y, 0.0])
        qz = math.sin(theta / 2.0)
        qw = math.cos(theta / 2.0)
        orientation = np.array([qw, 0.0, 0.0, qz])  # [w, x, y, z]
        
        return (position, orientation), self.pose_covariance, self.localization_quality
    
    def get_particles_as_poses(self):
        """
        Convert particles to a list of Pose messages for visualization.
        
        Returns:
            List of Pose messages
        """
        return [p.to_pose() for p in self.particles]

class RobotPosePublisher(Node):
    def __init__(self):
        super().__init__('robot_pose_publisher')
        
        # Initialize EKF
        self.ekf = ExtendedKalmanFilter()
        
        # Initialize Particle Filter
        self.particle_filter = ParticleFilter(
            num_particles=500,
            motion_noise=(0.05, 0.05, 0.05),
            measurement_noise=(0.1, 0.1)
        )
        
        # Create publisher for robot pose
        self.publisher = self.create_publisher(PoseStamped, 'robot_pose', 10)
        
        # Create publisher for pose with covariance
        self.pose_with_cov_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'robot_pose_with_covariance', 10)
        
        # Create publisher for particles (for visualization)
        self.particle_pub = self.create_publisher(
            PoseArray, 'localization_particles', 10)
            
        # Create publisher for visualization markers
        self.marker_pub = self.create_publisher(
            MarkerArray, 'localization_markers', 10)
        
        # Subscribe to VIO odometry
        self.vio_subscription = self.create_subscription(
            Odometry,
            '/oak/vio/odometry',
            self.vio_callback,
            10
        )
        
        # Subscribe to IMU data
        self.imu_subscription = self.create_subscription(
            Imu,
            '/oak/imu/data',
            self.imu_callback,
            10
        )
        
        # Subscribe to obstacle detections
        self.obstacle_subscription = self.create_subscription(
            PoseArray,
            '/obstacle_poses',
            self.obstacle_callback,
            10
        )
        
        # Subscribe to goal location
        self.goal_subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_callback,
            10
        )
        
        # Subscribe to sonar sensors with corrected topic paths
        self.sonar_subscriptions = {}
        sonar_topics = {
            'left_front': '/sonar/left/front',
            'left_rear': '/sonar/left/rear',
            'right_front': '/sonar/right/front',  # Note the extra slash in path
            'right_rear': '/sonar/right/rear'     # Note the extra slash in path
        }
        
        for sonar_name, topic in sonar_topics.items():
            self.sonar_subscriptions[sonar_name] = self.create_subscription(
                Range,
                topic,
                lambda msg, name=sonar_name: self.sonar_callback(msg, name),
                10
            )
        
        # Initialize pose
        self.current_pose = PoseStamped()
        self.current_pose.header.frame_id = 'map'
        self.current_pose.pose.position.x = 0.0
        self.current_pose.pose.position.y = 0.0
        self.current_pose.pose.position.z = 0.0
        self.current_pose.pose.orientation.w = 1.0
        
        # Store detected obstacles
        self.detected_obstacles = PoseArray()
        self.detected_obstacles.header.frame_id = 'robot'
        
        # Store previous pose for odometry calculation
        self.previous_pose = None
        self.last_time = None
        
        # Particle filter initialization status
        self.pf_initialized = False
        
        # Create timer for regular publishing
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10Hz
        
        # Create timer for logging localization stats
        self.stats_timer = self.create_timer(5.0, self.stats_callback)  # Every 5 seconds
        
        # Set up TF buffer for coordinate transformations
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.get_logger().info("RobotPosePublisher initialized")

    def sonar_callback(self, msg: Range, sonar_name: str):
        """Handle sonar range messages."""
        try:
            # Skip NaN values
            if math.isnan(msg.range):
                self.get_logger().debug(f"Received NaN reading from {sonar_name} sonar")
                return
                
            # Skip values outside range limits
            if msg.range < msg.min_range or msg.range > msg.max_range:
                self.get_logger().debug(f"Received out-of-range reading from {sonar_name}: {msg.range:.2f}m")
                return
                
            # Register the sonar reading with the particle filter
            self.particle_filter.register_sonar_reading(sonar_name, msg.range)
            self.get_logger().debug(f"Registered valid sonar reading from {sonar_name}: {msg.range:.2f}m")
            
        except Exception as e:
            self.get_logger().error(f"Error processing sonar data from {sonar_name}: {str(e)}")

    def vio_callback(self, msg):
        """
        Update pose based on VIO odometry data
        """
        try:
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            # Initial pose estimate from EKF
            if self.last_time is not None:
                # Time update
                dt = current_time - self.last_time
                self.ekf.predict(dt)
            
            # Measurement update
            measurement = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
                msg.pose.pose.orientation.w,
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z
            ]
            self.ekf.update(measurement)
            
            # Get filtered pose from EKF
            position, orientation = self.ekf.get_pose()
            estimated_pose = (np.array(position), np.array(orientation))
            
            # Initialize particle filter if needed
            if not self.pf_initialized:
                self.particle_filter.initialize(
                    Pose(
                        position=Point(
                            x=float(position[0]),
                            y=float(position[1]),
                            z=float(position[2])
                        ),
                        orientation=msg.pose.pose.orientation
                    ),
                    position_uncertainty=(0.5, 0.5),
                    orientation_uncertainty=0.2
                )
                self.pf_initialized = True
                self.previous_pose = estimated_pose
            else:
                # Calculate odometry change since last update
                if self.previous_pose is not None:
                    prev_position, prev_orientation = self.previous_pose
                    
                    # Position delta in global frame
                    dx = position[0] - prev_position[0]
                    dy = position[1] - prev_position[1]
                    
                    # Extract yaw angles from quaternions
                    r_prev = Rotation.from_quat([prev_orientation[1], prev_orientation[2], 
                                               prev_orientation[3], prev_orientation[0]])
                    prev_euler = r_prev.as_euler('xyz')
                    prev_yaw = prev_euler[2]
                    
                    r_curr = Rotation.from_quat([orientation[1], orientation[2], 
                                               orientation[3], orientation[0]])
                    curr_euler = r_curr.as_euler('xyz')
                    curr_yaw = curr_euler[2]
                    
                    # Calculate yaw delta
                    dyaw = curr_yaw - prev_yaw
                    # Normalize to [-pi, pi]
                    dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
                    
                    # Convert global frame delta to robot frame delta
                    # Rotate back by the inverse of the previous yaw
                    cos_prev = math.cos(-prev_yaw)
                    sin_prev = math.sin(-prev_yaw)
                    
                    dx_robot = dx * cos_prev - dy * sin_prev
                    dy_robot = dx * sin_prev + dy * cos_prev
                    
                    # Update particle filter with odometry delta
                    self.particle_filter.predict((dx_robot, dy_robot, dyaw))
                
                # Update particle filter weights with detected obstacles
                self.particle_filter.update_weights(self.detected_obstacles)
                
                # Update pose estimate from particle filter
                pf_pose, pf_covariance, quality = self.particle_filter.get_best_pose()
                
                if pf_pose is not None:
                    # Blend EKF and particle filter estimates based on quality
                    pf_position, pf_orientation = pf_pose
                    blended_position = position * (1 - quality) + pf_position * quality
                    
                    # For orientation, use SLERP (spherical linear interpolation)
                    # But for simplicity, just use the particle filter orientation if quality is high
                    blended_orientation = orientation
                    if quality > 0.7:
                        blended_orientation = pf_orientation
                    
                    # Update current pose
                    position = blended_position
                    orientation = blended_orientation
                
                # Save pose for next iteration
                self.previous_pose = (position, orientation)
            
            # Update current pose
            self.current_pose.pose.position.x = float(position[0])
            self.current_pose.pose.position.y = float(position[1])
            self.current_pose.pose.position.z = float(position[2])
            
            self.current_pose.pose.orientation.w = float(orientation[0])
            self.current_pose.pose.orientation.x = float(orientation[1])
            self.current_pose.pose.orientation.y = float(orientation[2])
            self.current_pose.pose.orientation.z = float(orientation[3])
            
            self.last_time = current_time
            
            # Publish particles for visualization
            self.publish_particles()
            
        except Exception as e:
            self.get_logger().error(f'Error processing VIO data: {str(e)}')

    def imu_callback(self, msg):
        """
        Update pose based on IMU data
        """
        try:
            current_time = self.get_clock().now().nanoseconds / 1e9
            
            if self.last_time is not None:
                # Time update
                dt = current_time - self.last_time
                self.ekf.predict(dt)
            
            # The rest of the IMU processing logic from previous implementation
            # This will only update the EKF portion
            
        except Exception as e:
            self.get_logger().error(f'Error processing IMU data: {str(e)}')

    def obstacle_callback(self, msg):
        """
        Process detected obstacles for localization
        """
        try:
            # Store latest obstacle detections
            self.detected_obstacles = msg
            
            # Log number of obstacles detected
            self.get_logger().debug(f'Received {len(msg.poses)} obstacle detections')
            
        except Exception as e:
            self.get_logger().error(f'Error processing obstacle data: {str(e)}')

    def goal_callback(self, msg):
        """
        Process goal location
        """
        try:
            # Register the goal in the particle filter
            goal_position = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z
            ])
            
            # Register with ID 0 (assuming only one goal at a time)
            self.particle_filter.register_goal(0, goal_position)
            
            self.get_logger().info(f'Registered new goal at x={goal_position[0]:.2f}, '
                                 f'y={goal_position[1]:.2f}')
            
        except Exception as e:
            self.get_logger().error(f'Error processing goal data: {str(e)}')

    def timer_callback(self):
        """
        Publish the current pose at regular intervals
        """
        try:
            # Update timestamp
            self.current_pose.header.stamp = self.get_clock().now().to_msg()
            
            # Publish pose
            self.publisher.publish(self.current_pose)
            
            # Publish pose with covariance
            if self.pf_initialized:
                _, covariance, quality = self.particle_filter.get_best_pose()
                if covariance is not None:
                    pose_with_cov = PoseWithCovarianceStamped()
                    pose_with_cov.header = self.current_pose.header
                    pose_with_cov.pose.pose = self.current_pose.pose
                    
                    # Convert 3x3 covariance to 6x6 (x, y, z, roll, pitch, yaw)
                    full_covariance = np.zeros((6, 6))
                    full_covariance[0:2, 0:2] = covariance[0:2, 0:2]  # Position (x, y)
                    full_covariance[5, 5] = covariance[2, 2]          # Orientation (yaw)
                    
                    # Flatten row-major order
                    pose_with_cov.pose.covariance = full_covariance.flatten().tolist()
                    
                    self.pose_with_cov_pub.publish(pose_with_cov)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing pose: {str(e)}')

    def publish_particles(self):
        """Publish particles and visualization markers for RViz"""
        if not self.pf_initialized:
            return
            
        try:
            # 1. Publish particles as PoseArray
            particle_msg = PoseArray()
            particle_msg.header.frame_id = 'map'
            particle_msg.header.stamp = self.get_clock().now().to_msg()
            
            # Convert particles to poses
            particle_msg.poses = self.particle_filter.get_particles_as_poses()
            self.particle_pub.publish(particle_msg)
            
            # 2. Create marker array for additional visualizations
            marker_array = MarkerArray()
            
            # Get current time stamp
            current_time = self.get_clock().now().to_msg()
            
            # 3. Visualize covariance ellipse
            pose, covariance, quality = self.particle_filter.get_best_pose()
            if pose is not None and covariance is not None:
                position, orientation = pose
                
                # Add covariance ellipse
                ellipse_marker = self.create_covariance_marker(
                    position, covariance, current_time, quality)
                marker_array.markers.append(ellipse_marker)
                
                # Add text marker showing localization quality
                quality_marker = self.create_text_marker(
                    position, f"Quality: {quality:.2f}", 
                    current_time, id=1, z_offset=0.5, 
                    r=0.0, g=0.8, b=1.0)
                marker_array.markers.append(quality_marker)
                
            # 4. Visualize high-weight particles
            high_weight_markers = self.create_high_weight_particle_markers(
                self.particle_filter.particles, current_time)
            marker_array.markers.extend(high_weight_markers)
            
            # 5. Visualize known obstacles in the map
            obstacle_markers = self.create_obstacle_markers(
                self.particle_filter.known_obstacles, current_time)
            marker_array.markers.extend(obstacle_markers)
            
            # 6. Visualize sonar readings
            sonar_markers = self.create_sonar_markers(current_time)
            marker_array.markers.extend(sonar_markers)
            
            # Publish all markers
            self.marker_pub.publish(marker_array)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing visualization: {str(e)}')
            
    def create_covariance_marker(self, position, covariance, timestamp, quality=1.0):
        """Create a marker representing the covariance ellipse"""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = timestamp
        marker.ns = "covariance"
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        
        # Set position
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = -0.99  # Just below ground plane for visibility
        
        # Set orientation (identity quaternion for 2D)
        marker.pose.orientation.w = 1.0
        
        # Calculate ellipse size from covariance
        if covariance.shape == (3, 3):
            # Extract position covariance (2x2)
            pos_cov = covariance[:2, :2]
            
            # Calculate eigenvalues and eigenvectors
            try:
                eigenvalues, eigenvectors = np.linalg.eig(pos_cov)
                
                # Sort by eigenvalue
                idx = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[idx]
                eigenvectors = eigenvectors[:, idx]
                
                # Calculate ellipse size (3-sigma for 99% confidence)
                marker.scale.x = max(0.2, math.sqrt(eigenvalues[0]) * 3.0)
                marker.scale.y = max(0.2, math.sqrt(eigenvalues[1]) * 3.0)
                marker.scale.z = 0.01  # Thin disc
                
                # Calculate orientation from eigenvectors
                angle = math.atan2(eigenvectors[1, 0], eigenvectors[0, 0])
                
                # Convert to quaternion
                marker.pose.orientation.w = math.cos(angle / 2.0)
                marker.pose.orientation.z = math.sin(angle / 2.0)
            except np.linalg.LinAlgError:
                # Fallback if eigenvalue calculation fails
                marker.scale.x = 0.5
                marker.scale.y = 0.5
                marker.scale.z = 0.01
                
        else:
            # Default size if covariance format is unexpected
            marker.scale.x = 0.5
            marker.scale.y = 0.5
            marker.scale.z = 0.01
        
        # Color based on quality (green for high quality, red for low)
        marker.color.r = 1.0 - quality
        marker.color.g = quality
        marker.color.b = 0.0
        marker.color.a = 0.5  # Semi-transparent
        
        return marker
    
    def create_text_marker(self, position, text, timestamp, id=0, z_offset=0.0, r=1.0, g=1.0, b=1.0):
        """Create a text marker at the specified position"""
        marker = Marker()
        marker.header.frame_id = 'map'
        marker.header.stamp = timestamp
        marker.ns = "text"
        marker.id = id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2]) + z_offset
        marker.pose.orientation.w = 1.0
        
        marker.text = text
        marker.scale.z = 0.2  # Text size
        
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = 1.0
        
        return marker
    
    def create_high_weight_particle_markers(self, particles, timestamp):
        """Create markers for the highest weight particles"""
        markers = []
        
        # Skip if no particles
        if not particles:
            return markers
        
        # Sort particles by weight
        sorted_particles = sorted(particles, key=lambda p: p.weight, reverse=True)
        
        # Take top 5% or at least 5 particles
        num_to_visualize = max(5, int(len(particles) * 0.05))
        top_particles = sorted_particles[:num_to_visualize]
        
        for i, particle in enumerate(top_particles):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = timestamp
            marker.ns = "high_weight_particles"
            marker.id = i
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            
            # Position
            marker.pose.position.x = particle.x
            marker.pose.position.y = particle.y
            marker.pose.position.z = -0.98  # Just below ground
            
            # Orientation
            marker.pose.orientation.w = math.cos(particle.theta / 2.0)
            marker.pose.orientation.z = math.sin(particle.theta / 2.0)
            
            # Size - scale by weight
            weight_scale = 0.5 + (particle.weight * 10.0)  # Boost small weights for visibility
            marker.scale.x = 0.3 * weight_scale  # Arrow length
            marker.scale.y = 0.05 * weight_scale  # Arrow width
            marker.scale.z = 0.05 * weight_scale  # Arrow height
            
            # Color - yellow to red gradient by weight
            marker.color.r = 1.0
            marker.color.g = particle.weight * 5.0  # Scale to make small differences visible
            marker.color.b = 0.0
            marker.color.a = 0.8
            
            markers.append(marker)
        
        return markers
    
    def create_obstacle_markers(self, known_obstacles, timestamp):
        """Create markers for known obstacles in the map"""
        markers = []
        
        for i, (obs_id, position) in enumerate(known_obstacles.items()):
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = timestamp
            marker.ns = "known_obstacles"
            marker.id = i
            marker.type = Marker.CYLINDER
            marker.action = Marker.ADD
            
            marker.pose.position.x = float(position[0])
            marker.pose.position.y = float(position[1])
            marker.pose.position.z = -0.97  # Just below ground
            marker.pose.orientation.w = 1.0
            
            marker.scale.x = 0.3  # Diameter
            marker.scale.y = 0.3  # Diameter
            marker.scale.z = 0.02  # Height
            
            # Obstacles are blue
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 0.8
            marker.color.a = 0.7
            
            markers.append(marker)
        
        return markers
    
    def create_sonar_markers(self, timestamp):
        """Create markers visualizing sonar readings"""
        markers = []
        
        # Get robot position and orientation
        if not hasattr(self, 'current_pose') or not self.current_pose:
            return markers
            
        robot_x = self.current_pose.pose.position.x
        robot_y = self.current_pose.pose.position.y
        
        # Extract yaw from quaternion
        qx = self.current_pose.pose.orientation.x
        qy = self.current_pose.pose.orientation.y
        qz = self.current_pose.pose.orientation.z
        qw = self.current_pose.pose.orientation.w
        
        # Calculate robot yaw from quaternion
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Create markers for each sonar
        for i, (sonar_name, range_value) in enumerate(self.particle_filter.sonar_readings.items()):
            if range_value is None or range_value > 5.0:  # Skip invalid or very long readings
                continue
                
            config = self.particle_filter.sonar_configs.get(sonar_name)
            if config is None:
                continue
                
            # Get sonar position and orientation in robot frame
            x_sonar, y_sonar, z_sonar = config.position
            _, _, yaw_deg = config.orientation
            yaw_rad = math.radians(yaw_deg)
            
            # Calculate sonar position in world frame
            # First rotate by robot's yaw
            x_rotated = x_sonar * math.cos(robot_yaw) - y_sonar * math.sin(robot_yaw)
            y_rotated = x_sonar * math.sin(robot_yaw) + y_sonar * math.cos(robot_yaw)
            
            # Then translate by robot's position
            sonar_world_x = robot_x + x_rotated
            sonar_world_y = robot_y + y_rotated
            
            # Calculate sonar's absolute orientation
            sonar_yaw = robot_yaw + yaw_rad
            
            # Create marker for sonar cone
            cone = Marker()
            cone.header.frame_id = 'map'
            cone.header.stamp = timestamp
            cone.ns = "sonar_cones"
            cone.id = i
            cone.type = Marker.LINE_STRIP
            cone.action = Marker.ADD
            
            # Sonar cone angle (typically around 30 degrees)
            cone_angle = math.radians(15.0)  # Half the cone angle
            
            # Starting point (sonar position)
            start_point = Point()
            start_point.x = sonar_world_x
            start_point.y = sonar_world_y
            start_point.z = -0.95  # Just below ground
            cone.points.append(start_point)
            
            # Left edge point
            left_angle = sonar_yaw + cone_angle
            left_point = Point()
            left_point.x = sonar_world_x + range_value * math.cos(left_angle)
            left_point.y = sonar_world_y + range_value * math.sin(left_angle)
            left_point.z = -0.95
            cone.points.append(left_point)
            
            # Arc points
            num_arc_points = 10
            for j in range(1, num_arc_points):
                arc_angle = sonar_yaw + cone_angle - (2.0 * cone_angle * j / num_arc_points)
                arc_point = Point()
                arc_point.x = sonar_world_x + range_value * math.cos(arc_angle)
                arc_point.y = sonar_world_y + range_value * math.sin(arc_angle)
                arc_point.z = -0.95
                cone.points.append(arc_point)
            
            # Right edge point
            right_angle = sonar_yaw - cone_angle
            right_point = Point()
            right_point.x = sonar_world_x + range_value * math.cos(right_angle)
            right_point.y = sonar_world_y + range_value * math.sin(right_angle)
            right_point.z = -0.95
            cone.points.append(right_point)
            
            # Back to start to close the cone
            cone.points.append(start_point)
            
            # Line properties
            cone.scale.x = 0.02  # Line width
            cone.color.r = 0.0
            cone.color.g = 0.8
            cone.color.b = 0.8
            cone.color.a = 0.5
            
            markers.append(cone)
            
            # Add endpoint marker
            endpoint = Marker()
            endpoint.header.frame_id = 'map'
            endpoint.header.stamp = timestamp
            endpoint.ns = "sonar_endpoints"
            endpoint.id = i
            endpoint.type = Marker.SPHERE
            endpoint.action = Marker.ADD
            
            # Endpoint position
            center_angle = sonar_yaw
            endpoint.pose.position.x = sonar_world_x + range_value * math.cos(center_angle)
            endpoint.pose.position.y = sonar_world_y + range_value * math.sin(center_angle)
            endpoint.pose.position.z = -0.95
            endpoint.pose.orientation.w = 1.0
            
            # Size
            endpoint.scale.x = 0.1
            endpoint.scale.y = 0.1
            endpoint.scale.z = 0.1
            
            # Color - cyan endpoint
            endpoint.color.r = 0.0
            endpoint.color.g = 1.0
            endpoint.color.b = 1.0
            endpoint.color.a = 0.9
            
            markers.append(endpoint)
            
            # Add text label with sonar name and range
            text = Marker()
            text.header.frame_id = 'map'
            text.header.stamp = timestamp
            text.ns = "sonar_labels"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            
            # Position text above endpoint
            text.pose.position.x = endpoint.pose.position.x
            text.pose.position.y = endpoint.pose.position.y
            text.pose.position.z = endpoint.pose.position.z + 0.2
            text.pose.orientation.w = 1.0
            
            # Text content and appearance
            text.text = f"{sonar_name}: {range_value:.2f}m"
            text.scale.z = 0.15  # Text size
            text.color.r = 0.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            
            markers.append(text)
            
        return markers

    def stats_callback(self):
        """
        Log localization statistics periodically
        """
        if not self.pf_initialized:
            return
            
        try:
            _, _, quality = self.particle_filter.get_best_pose()
            self.get_logger().info(f"Localization quality: {quality:.2f}")
            
            # Log sonar readings
            valid_sonar_readings = 0
            for sonar_name, reading in self.particle_filter.sonar_readings.items():
                if reading is not None:
                    valid_sonar_readings += 1
                    self.get_logger().info(f"Sonar {sonar_name}: {reading:.2f}m")
            
            if valid_sonar_readings == 0:
                self.get_logger().info("No valid sonar readings available")
            
        except Exception as e:
            self.get_logger().error(f'Error logging localization stats: {str(e)}')

class ExtendedKalmanFilter:
    def __init__(self):
        # State: [x, y, z, vx, vy, vz, roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate]
        self.state = np.zeros(12)
        self.state[11] = 1.0  # Initial orientation (w component of quaternion)
        
        # Initial State Covariance (P) - represents initial uncertainty
        self.P = np.zeros((12, 12))
        # Position uncertainty (meters^2)
        self.P[0:3, 0:3] = np.eye(3) * 0.01  # Start with 10cm position uncertainty
        # Velocity uncertainty (meters/sec^2)
        self.P[3:6, 3:6] = np.eye(3) * 0.04  # Start with 0.2 m/s velocity uncertainty
        # Orientation uncertainty (radians^2)
        self.P[6:9, 6:9] = np.eye(3) * (np.pi/180)**2  # Start with 1 degree uncertainty
        # Angular velocity uncertainty (radians/sec^2)
        self.P[9:12, 9:12] = np.eye(3) * (np.pi/180)**2  # Start with 1 deg/s uncertainty
        
        # Process Noise Covariance (Q) - represents system dynamics uncertainty
        self.Q = np.zeros((12, 12))
        # Position process noise
        self.Q[0:3, 0:3] = np.eye(3) * 0.01  # 10cm position uncertainty per second
        # Velocity process noise
        self.Q[3:6, 3:6] = np.eye(3) * 0.25  # 0.5 m/s velocity uncertainty per second
        # Orientation process noise
        self.Q[6:9, 6:9] = np.eye(3) * (np.pi/180)**2  # 1 degree orientation uncertainty per second
        # Angular velocity process noise
        self.Q[9:12, 9:12] = np.eye(3) * (np.pi/90)**2  # 2 degrees/s angular velocity uncertainty per second
        
        # Measurement Noise Covariance (R) - represents sensor noise
        self.R = np.zeros((7, 7))
        # VIO position measurement noise (meters^2)
        self.R[0:3, 0:3] = np.eye(3) * 0.0025  # 5cm standard deviation for position
        # VIO orientation measurement noise (radians^2)
        self.R[3:7, 3:7] = np.eye(4) * (np.pi/180)**2  # 1 degree standard deviation for orientation
        
        self.last_time = None

    def predict(self, dt):
        """
        Predict step of EKF
        """
        if dt <= 0:
            return

        # State transition matrix
        F = np.eye(12)
        F[0:3, 3:6] = np.eye(3) * dt  # Position update from velocity
        
        # For orientation, we'll only update if there's significant angular velocity
        angular_vel_magnitude = np.linalg.norm(self.state[9:12])
        if angular_vel_magnitude > 0.01:  # Only update orientation if angular velocity is significant
            F[6:9, 9:12] = np.eye(3) * dt  # Orientation update from angular velocity

        # Predict state
        self.state = np.dot(F, self.state)
        
        # Normalize orientation angles and wrap to [-pi, pi]
        for i in range(6, 9):
            self.state[i] = np.arctan2(np.sin(self.state[i]), np.cos(self.state[i]))

        # Zero out very small angular velocities to prevent drift
        mask = np.abs(self.state[9:12]) < 0.01
        self.state[9:12][mask] = 0.0

        # Predict covariance
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q * dt

    def update(self, measurement):
        """
        Update step of EKF
        measurement: [x, y, z, qw, qx, qy, qz]
        """
        try:
            # Measurement matrix (linear for position, nonlinear for orientation)
            H = np.zeros((7, 12))
            H[0:3, 0:3] = np.eye(3)  # Position measurements
            H[3:6, 6:9] = np.eye(3)  # Orientation measurements
            
            # Convert quaternion to euler for measurement
            r = Rotation.from_quat([measurement[4], measurement[5], measurement[6], measurement[3]])
            euler = r.as_euler('xyz', degrees=False)
            
            # Get current state euler angles
            current_euler = self.state[6:9]
            
            # Measurement vector
            z = np.zeros(7)
            z[0:3] = measurement[0:3]  # Position
            z[3:6] = euler  # Orientation as euler angles
            
            # Calculate angle differences properly
            angle_diff = np.zeros(3)
            for i in range(3):
                diff = euler[i] - current_euler[i]
                angle_diff[i] = np.arctan2(np.sin(diff), np.cos(diff))
            
            # Innovation
            y = np.zeros(7)
            y[0:3] = z[0:3] - self.state[0:3]  # Position innovation
            y[3:6] = angle_diff  # Orientation innovation
            
            # Innovation covariance
            S = np.dot(np.dot(H, self.P), H.T) + self.R
            
            # Kalman gain
            K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
            
            # Update state
            state_update = np.dot(K, y)
            self.state += state_update
            
            # Normalize angles
            for i in range(6, 9):
                self.state[i] = np.arctan2(np.sin(self.state[i]), np.cos(self.state[i]))
            
            # Update covariance
            self.P = self.P - np.dot(np.dot(K, H), self.P)
            
            # Ensure covariance stays positive definite
            self.P = (self.P + self.P.T) / 2
            
        except Exception as e:
            print(f"Error in EKF update: {str(e)}")
            return

    def get_pose(self):
        """
        Get current pose as (position, orientation_quaternion)
        """
        position = self.state[0:3]
        
        # Convert euler angles to quaternion
        r = Rotation.from_euler('xyz', self.state[6:9])
        quat = r.as_quat()  # Returns [x, y, z, w]
        
        return position, [quat[3], quat[0], quat[1], quat[2]]  # Reorder to [w, x, y, z]

def main(args=None):
    rclpy.init(args=args)
    node = RobotPosePublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down RobotPosePublisher.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()