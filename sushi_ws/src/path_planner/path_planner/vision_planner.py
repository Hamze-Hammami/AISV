#!/usr/bin/env python3

import numpy as np
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import ColorRGBA
import rclpy
import math
import time
import random
import hashlib
from queue import Queue, PriorityQueue
from scipy.ndimage import distance_transform_edt

class VisionGuidedPlanner:
    def __init__(self,
                 k_att: float = 1.0,        # Attractive potential gain
                 k_rep: float = 100.0,      # Repulsive potential gain
                 rho_0: float = 2.0,        # Influence radius of obstacles
                 d_star: float = 2.0,       # Desired distance to goal
                 resolution: float = 0.05,    # Grid resolution
                 max_linear_speed: float = 0.5,  # Maximum linear speed (m/s)
                 max_angular_speed: float = 1.0,  # Maximum angular speed (rad/s)
                 robot_length: float = 0.9,  # Robot length in meters
                 robot_width: float = 0.7,   # Robot width in meters
                 safety_margin: float = 0.05,  # Additional safety margin in meters
                 visualization_interval: float = 0.5):  # Visualization update interval
                 
        # Planning parameters
        self.k_att = k_att
        self.k_rep = k_rep
        self.rho_0 = rho_0
        self.d_star = d_star
        self.resolution = resolution
        self.max_linear_speed = max_linear_speed
        self.max_angular_speed = max_angular_speed
        
        # Robot physical parameters
        self.robot_length = robot_length
        self.robot_width = robot_width
        self.safety_margin = safety_margin
        self.robot_radius = math.sqrt(robot_length**2 + robot_width**2) / 2.0
        
        # Visualization parameters
        self.visualization_interval = visualization_interval
        self.visualization_level = 2  # 0: minimal, 1: normal, 2: detailed
        self.force_visualization = False
        self.last_visualization_time = 0
        self.last_visualization_markers = None
        self.last_grid_hash = None
        
        # Map properties
        self.map_width = 20.0  # 20x20 meter map
        self.map_height = 20.0
        self.grid_width = int(self.map_width / resolution)
        self.grid_height = int(self.map_height / resolution)
        
        # Grid data
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width))
        self.wavefront_grid = None
        self.potential_field = None
        self.flow_field = None
        self.distance_field = None
        self.clearance_field = None
        self.passage_visualization_grid = None
        
        # Last processed robot position for grid centering
        self.last_robot_pos = None
        
        # The 8 neighboring directions for grid traversal
        self.directions = [(1,0), (-1,0), (0,1), (0,-1), 
                           (1,1), (-1,1), (1,-1), (-1,-1)]
        
        # For visualization
        self.wavefront_grid_publisher = None
        self.debug_publisher = None
        
        # Enhanced parameters for local minima avoidance
        self.local_minima_detection_threshold = 5  # Steps without improvement before triggering escape
        self.noise_amplitude = 0.1  # Maximum amplitude of random noise for escaping local minima
        self.momentum_factor = 0.3  # Weight of previous steps (momentum) for escaping local minima
        self.goal_bias_factor = 0.2  # Bias toward goal when stuck
        self.max_random_attempts = 3  # Maximum number of random escape attempts
        
        # Wavefront-specific parameters
        self.wavefront_influence_base = 0.1  # Base influence of wavefront on potential field
        self.wavefront_influence_max = 0.3   # Maximum influence of wavefront
        
        # Adaptive parameters
        self.adaptive_parameters = True  # Whether to use adaptive parameters
        self.adaptive_k_att_range = (0.5 * k_att, 2.0 * k_att)  # Range for adaptive k_att
        self.adaptive_k_rep_range = (0.5 * k_rep, 2.0 * k_rep)  # Range for adaptive k_rep
        
        # Multi-resolution planning parameters
        self.use_multi_resolution = True
        self.coarse_factor = 4  # 4x coarser grid for initial planning
        
        # Local minima optimization parameters
        self.enable_local_minima_optimization = True
        self.clearance_fitness_threshold = 1.2  # Minimum clearance ratio
        self.repulsive_reduction_factor = 0.5  # How much to reduce repulsive forces in passages
        self.repulsive_increase_factor = 1.5  # How much to increase repulsive forces to go around
        self.local_minima_regions = []  # Store detected local minima regions
        
        # State tracking
        self.prev_position = None
        self.prev_force = None
        self.oscillation_count = 0
        self.progress_stagnation_count = 0
        self.escape_attempt_count = 0
        self.random_walk_active = False
        self.current_escapestrategy = 0
        
        # Planning metrics
        self.last_planning_time = None
        self.last_planning_success = False
        self.successful_paths_count = 0
        
        self.last_planning_time = None
        self.last_wavefront_time = None
        self.last_path_computation_time = None

        # Successful parameters memory
        self.last_successful_params = {
            'k_att': k_att,
            'k_rep': k_rep,
            'rho_0': rho_0
        }
        
    def set_visualization_publisher(self, publisher):
        """Set the publisher for visualizing the wavefront grid."""
        self.wavefront_grid_publisher = publisher
        
        # Initialize the grid for visualization if needed
        if self.occupancy_grid is None:
            self.occupancy_grid = np.zeros((self.grid_height, self.grid_width))
            
        # Force visualization update on next call
        self.force_visualization = True
    
    def set_debug_publisher(self, publisher):
        """Set publisher for debug visualization."""
        self.debug_publisher = publisher
        
    def world_to_grid(self, world_point):
        """Convert world coordinates to grid coordinates."""
        if self.last_robot_pos is None:
            # If no robot position yet, assume grid is centered at origin
            grid_x = int((world_point[0] + self.map_width/2) / self.resolution)
            grid_y = int((world_point[1] + self.map_height/2) / self.resolution)
        else:
            # Grid is centered on robot
            center_x = self.last_robot_pos[0]
            center_y = self.last_robot_pos[1]
            grid_x = int((world_point[0] - center_x + self.map_width/2) / self.resolution)
            grid_y = int((world_point[1] - center_y + self.map_height/2) / self.resolution)
        
        return grid_x, grid_y
    
    def grid_to_world(self, grid_x, grid_y):
        """Convert grid coordinates to world coordinates."""
        if self.last_robot_pos is None:
            # If no robot position yet, assume grid is centered at origin
            world_x = grid_x * self.resolution - self.map_width/2
            world_y = grid_y * self.resolution - self.map_height/2
        else:
            # Grid is centered on robot
            center_x = self.last_robot_pos[0]
            center_y = self.last_robot_pos[1]
            world_x = grid_x * self.resolution - self.map_width/2 + center_x
            world_y = grid_y * self.resolution - self.map_height/2 + center_y
        
        return world_x, world_y
    
    def update_grid_from_obstacles(self, robot_position, object_positions, obstacle_data, sonar_data=None):
        """Update occupancy grid with proper robot-sized inflation - FIXED VERSION."""
        # Reset grid
        self.occupancy_grid = np.zeros((self.grid_height, self.grid_width))
        
        # Create a separate grid for exact obstacle positions (no inflation)
        raw_obstacle_grid = np.zeros((self.grid_height, self.grid_width))
        
        # Center grid on robot position
        self.last_robot_pos = robot_position
        center_x = robot_position[0]
        center_y = robot_position[1]
        
        # Add obstacles to raw grid with increased size for safety
        for obstacle in obstacle_data:
            if not isinstance(obstacle, dict):
                continue
            
            position = obstacle.get("position", None)
            size = obstacle.get("size", 0.3)
            
            # Increase effective size by 20% for safety
            size = size * 1.0
            
            if position is None or len(position) < 2:
                continue
            
            world_x, world_y = position[0], position[1]
            grid_x = int((world_x - center_x + self.map_width/2) / self.resolution)
            grid_y = int((world_y - center_y + self.map_height/2) / self.resolution)
            
            if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height:
                # Use a larger radius for initial obstacle marking
                obstacle_radius_cells = int((size / 2.0) / self.resolution)
                
                for dx in range(-obstacle_radius_cells, obstacle_radius_cells + 1):
                    for dy in range(-obstacle_radius_cells, obstacle_radius_cells + 1):
                        nx, ny = grid_x + dx, grid_y + dy
                        if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                            # Use circular inflation
                            if dx*dx + dy*dy <= obstacle_radius_cells*obstacle_radius_cells:
                                raw_obstacle_grid[ny, nx] = 1
        
        # Calculate distance transform for proper inflation
        inverted_grid = 1.0 - raw_obstacle_grid
        self.distance_field = distance_transform_edt(inverted_grid) * self.resolution
        
        # FIXED: Use less conservative clearance requirements
        required_clearance = self.robot_radius * 0.6 + self.safety_margin * 0.05  # Much less conservative
        
        # Mark cells as occupied if too close to obstacles - less aggressive
        self.occupancy_grid = np.where(self.distance_field < required_clearance, 1, 0)
        
        # IMPORTANT: Always keep the robot's current position and start positions free
        if robot_position is not None:
            robot_grid_x, robot_grid_y = self.world_to_grid(robot_position[:2])
            if (0 <= robot_grid_x < self.grid_width and 0 <= robot_grid_y < self.grid_height):
                # Force robot's current position to be free in a small radius
                clear_radius = 2  # 5x5 area around robot
                for dy in range(-clear_radius, clear_radius + 1):
                    for dx in range(-clear_radius, clear_radius + 1):
                        nx, ny = robot_grid_x + dx, robot_grid_y + dy
                        if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                            # Only clear if it's not a real obstacle (distance > robot_radius/2)
                            if self.distance_field[ny, nx] > self.robot_radius * 0.3:
                                self.occupancy_grid[ny, nx] = 0
        
        # Ensure proper corridor widths - less aggressive
        self._ensure_proper_corridors()
        
        return self.occupancy_grid

    def _ensure_proper_corridors(self):
        """Ensure corridors are wide enough for the robot with LESS aggressive margins."""
        
        # MUCH less conservative calculation
        min_passage_cells = int((self.robot_radius * 1.2 + self.safety_margin * 0.6) / self.resolution)  # Reduced multipliers
        
        # Increase the minimum threshold - don't close unless resolution is very coarse
        if min_passage_cells <= 1:  # Changed from 2 to 1
            return
            
        temp_grid = self.occupancy_grid.copy()
        
        # Process horizontal corridors with LESS strict criteria
        for y in range(self.grid_height):
            corridor_start = -1
            for x in range(self.grid_width):
                if corridor_start == -1 and temp_grid[y, x] == 0:
                    corridor_start = x
                elif corridor_start != -1 and temp_grid[y, x] == 1:
                    corridor_width = x - corridor_start
                    
                    # ONLY close if corridor is VERY narrow (less than 60% of minimum)
                    if 0 < corridor_width < max(1, int(min_passage_cells * 0.6)):  # Much more permissive
                        temp_grid[y, corridor_start:x] = 1
                    corridor_start = -1
        
        # Process vertical corridors with LESS strict criteria  
        for x in range(self.grid_width):
            corridor_start = -1
            for y in range(self.grid_height):
                if corridor_start == -1 and temp_grid[y, x] == 0:
                    corridor_start = y
                elif corridor_start != -1 and temp_grid[y, x] == 1:
                    corridor_height = y - corridor_start
                    
                    # ONLY close if corridor is VERY narrow (less than 60% of minimum)
                    if 0 < corridor_height < max(1, int(min_passage_cells * 0.6)):  # Much more permissive
                        temp_grid[corridor_start:y, x] = 1
                    corridor_start = -1
        
        self.occupancy_grid = temp_grid

    def find_safe_point_nearby(self, grid_x, grid_y, max_radius=10):
        """Find a safe point near the given grid coordinates."""
        if 0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height and self.occupancy_grid[grid_y, grid_x] == 0:
            return grid_x, grid_y
            
        # Search in expanding rings
        for radius in range(1, max_radius + 1):
            # Get points in the ring
            candidates = []
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only points on the ring
                        nx, ny = grid_x + dx, grid_y + dy
                        if (0 <= nx < self.grid_width and 
                            0 <= ny < self.grid_height and 
                            self.occupancy_grid[ny, nx] == 0):
                            # Calculate distance (for sorting)
                            dist = math.sqrt(dx*dx + dy*dy)
                            candidates.append((nx, ny, dist))
            
            # If we found candidates, return the closest one
            if candidates:
                candidates.sort(key=lambda x: x[2])  # Sort by distance
                return candidates[0][0], candidates[0][1]
        
        # If no safe point found within max_radius, return None
        return None

    def wavefront_expansion(self, start_point, goal_point):
        """
        Enhanced wavefront expansion from goal to start.
        Uses a priority queue for distance-based expansion and handles obstacles better.
        Creates both wavefront distance grid and flow field for navigation.
        """
        start_time = time.time()
        
        # Convert points to grid coordinates
        start_grid_x, start_grid_y = self.world_to_grid(start_point)
        goal_grid_x, goal_grid_y = self.world_to_grid(goal_point)
        
        # Check if points are within grid
        if (start_grid_x < 0 or start_grid_x >= self.grid_width or 
            start_grid_y < 0 or start_grid_y >= self.grid_height or
            goal_grid_x < 0 or goal_grid_x >= self.grid_width or 
            goal_grid_y < 0 or goal_grid_y >= self.grid_height):
            return None
            
        # Find safe points if start or goal are in obstacles
        if self.occupancy_grid[start_grid_y, start_grid_x] == 1:
            safe_start = self.find_safe_point_nearby(start_grid_x, start_grid_y)
            if safe_start is None:
                return None
            start_grid_x, start_grid_y = safe_start
            
        if self.occupancy_grid[goal_grid_y, goal_grid_x] == 1:
            safe_goal = self.find_safe_point_nearby(goal_grid_x, goal_grid_y)
            if safe_goal is None:
                return None
            goal_grid_x, goal_grid_y = safe_goal
        
        # Initialize wavefront grid and flow field
        self.wavefront_grid = np.full((self.grid_height, self.grid_width), -1, dtype=float)
        self.flow_field = np.zeros((self.grid_height, self.grid_width, 2), dtype=float)
        
        self.wavefront_grid[goal_grid_y, goal_grid_x] = 0
        
        # Use priority queue for better path expansion (based on distance)
        # This produces more natural paths than a simple queue
        queue = PriorityQueue()
        queue.put((0, goal_grid_x, goal_grid_y))  # (priority, x, y)
        
        # Perform wavefront expansion
        while not queue.empty():
            _, current_x, current_y = queue.get()
            current_dist = self.wavefront_grid[current_y, current_x]
            
            # Check if we've reached the start
            if current_x == start_grid_x and current_y == start_grid_y:
                break
            
            # Expand in all directions
            for dx, dy in self.directions:
                new_x = current_x + dx
                new_y = current_y + dy
                
                # Check if valid position
                if (0 <= new_x < self.grid_width and 
                    0 <= new_y < self.grid_height and
                    self.wavefront_grid[new_y, new_x] == -1 and
                    self.occupancy_grid[new_y, new_x] == 0):
                    
                    # Calculate distance for diagonals vs cardinal directions
                    if abs(dx) == 1 and abs(dy) == 1:
                        # Diagonal move (cost = sqrt(2))
                        new_dist = current_dist + 1.414
                    else:
                        # Cardinal move (cost = 1)
                        new_dist = current_dist + 1
                    
                    # Update wavefront grid
                    self.wavefront_grid[new_y, new_x] = new_dist
                    
                    # Update flow field (direction to go from this cell)
                    # Points toward the direction of the wavefront expansion (gradient)
                    self.flow_field[new_y, new_x] = [-dx, -dy]
                    
                    # Add to queue with priority based on distance to goal
                    # This makes the wavefront expand more evenly
                    heuristic = math.sqrt((new_x - start_grid_x)**2 + (new_y - start_grid_y)**2)
                    priority = new_dist + heuristic * 0.1  # Slight A* behavior
                    queue.put((priority, new_x, new_y))
        
        # Check if we found a path to start
        if self.wavefront_grid[start_grid_y, start_grid_x] == -1:
            self.last_planning_success = False
            return None
        
        # Smooth the flow field to produce better paths
        self._smooth_flow_field()
        
        # Extract path
        path = self.extract_path(start_grid_x, start_grid_y, goal_grid_x, goal_grid_y)
        
        # Record planning time and success
        self.last_planning_time = time.time() - start_time
        self.last_planning_success = (path is not None)
        
        return path
        
    def _smooth_flow_field(self):
        """Smooth the flow field to produce better paths."""
        if self.flow_field is None:
            return
            
        # Create a copy of the flow field
        smooth_field = self.flow_field.copy()
        
        # Smooth the flow field by averaging neighboring vectors
        for y in range(1, self.grid_height-1):
            for x in range(1, self.grid_width-1):
                # Skip obstacles and unprocessed cells
                if self.occupancy_grid[y, x] == 1 or self.wavefront_grid[y, x] == -1:
                    continue
                
                # Get neighboring flow vectors
                neighbors = []
                for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and
                        self.wavefront_grid[ny, nx] != -1 and self.occupancy_grid[ny, nx] == 0):
                        neighbors.append(self.flow_field[ny, nx])
                
                if neighbors:
                    # Average the vectors
                    avg_vector = np.mean(neighbors, axis=0)
                    # Combine with original vector (80% original, 20% average)
                    smooth_field[y, x] = 0.8 * self.flow_field[y, x] + 0.2 * avg_vector
                    
                    # Normalize
                    length = np.linalg.norm(smooth_field[y, x])
                    if length > 0:
                        smooth_field[y, x] /= length
        
        # Update flow field with smoothed version
        self.flow_field = smooth_field
    
    def _calculate_wavefront_gradient(self, x, y):
        """
        Calculate the gradient of the wavefront at a specific grid cell.
        Enhanced to handle boundaries better and provide more stable gradients.
        """
        if self.wavefront_grid is None:
            return np.zeros(2)
            
        gradient = np.zeros(2)
        
        # Check bounds with larger margin to avoid edge artifacts
        if x < 2 or y < 2 or x >= self.grid_width-2 or y >= self.grid_height-2:
            # Near boundary, use simpler calculation
            
            # Calculate x gradient using available cells
            if x > 0 and x < self.grid_width-1:
                if self.wavefront_grid[y, x+1] != -1 and self.wavefront_grid[y, x-1] != -1:
                    gradient[0] = (self.wavefront_grid[y, x+1] - self.wavefront_grid[y, x-1]) / 2.0
                elif self.wavefront_grid[y, x+1] != -1 and self.wavefront_grid[y, x] != -1:
                    gradient[0] = self.wavefront_grid[y, x+1] - self.wavefront_grid[y, x]
                elif self.wavefront_grid[y, x] != -1 and self.wavefront_grid[y, x-1] != -1:
                    gradient[0] = self.wavefront_grid[y, x] - self.wavefront_grid[y, x-1]
            
            # Calculate y gradient using available cells
            if y > 0 and y < self.grid_height-1:
                if self.wavefront_grid[y+1, x] != -1 and self.wavefront_grid[y-1, x] != -1:
                    gradient[1] = (self.wavefront_grid[y+1, x] - self.wavefront_grid[y-1, x]) / 2.0
                elif self.wavefront_grid[y+1, x] != -1 and self.wavefront_grid[y, x] != -1:
                    gradient[1] = self.wavefront_grid[y+1, x] - self.wavefront_grid[y, x]
                elif self.wavefront_grid[y, x] != -1 and self.wavefront_grid[y-1, x] != -1:
                    gradient[1] = self.wavefront_grid[y, x] - self.wavefront_grid[y-1, x]
        else:
            # For interior points, use Sobel operator for better gradient estimation
            x_weights = np.array([
                [1, 0, -1],
                [2, 0, -2],
                [1, 0, -1]
            ])
            
            y_weights = np.array([
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ])
            
            # Extract 3x3 window around the point
            window = np.zeros((3, 3))
            valid_points = 0
            
            for dy in range(3):
                for dx in range(3):
                    ny, nx = y + dy - 1, x + dx - 1
                    if (0 <= nx < self.grid_width and 0 <= ny < self.grid_height and 
                        self.wavefront_grid[ny, nx] != -1):
                        window[dy, dx] = self.wavefront_grid[ny, nx]
                        valid_points += 1
                    # Else keep as 0
            
            # Only use Sobel if enough valid points
            if valid_points >= 5:  # Need majority of points
                gradient[0] = np.sum(window * x_weights) / 8.0
                gradient[1] = np.sum(window * y_weights) / 8.0
            else:
                # Fall back to simpler calculation
                if self.wavefront_grid[y, x+1] != -1 and self.wavefront_grid[y, x-1] != -1:
                    gradient[0] = (self.wavefront_grid[y, x+1] - self.wavefront_grid[y, x-1]) / 2.0
                
                if self.wavefront_grid[y+1, x] != -1 and self.wavefront_grid[y-1, x] != -1:
                    gradient[1] = (self.wavefront_grid[y+1, x] - self.wavefront_grid[y-1, x]) / 2.0
        
        # Normalize for consistent magnitudes
        gradient_magnitude = np.linalg.norm(gradient)
        if gradient_magnitude > 0:
            gradient = gradient / gradient_magnitude
            
        return gradient
        
    def extract_path(self, start_grid_x, start_grid_y, goal_grid_x, goal_grid_y):
        """
        Extract path from wavefront grid and flow field.
        This creates a smoother and more natural path than just following gradient descent.
        """
        if self.wavefront_grid is None or self.flow_field is None:
            return None
            
        path = []
        current_x, current_y = start_grid_x, start_grid_y
        
        # Convert start point to world coordinates and add to path
        start_world_x, start_world_y = self.grid_to_world(start_grid_x, start_grid_y)
        path.append((start_world_x, start_world_y))
        
        # Maximum steps to prevent infinite loops
        max_steps = min(1000, self.grid_width * self.grid_height // 10)
        
        # Track visited points to detect loops
        visited = set([(current_x, current_y)])
        
        # Minimum distance improvement to detect progress
        min_dist_improvement = 0.5  # Must improve distance by this amount every 5 steps
        last_best_dist = self.wavefront_grid[current_y, current_x]
        steps_since_improvement = 0
        
        for step in range(max_steps):
            # Check if we've reached the goal vicinity
            if math.sqrt((current_x - goal_grid_x)**2 + (current_y - goal_grid_y)**2) <= 2:
                # Add goal point and exit
                goal_world_x, goal_world_y = self.grid_to_world(goal_grid_x, goal_grid_y)
                path.append((goal_world_x, goal_world_y))
                break
                
            # Get flow direction at current point
            flow_dir = self.flow_field[current_y, current_x]
            
            # If flow direction is zero, use wavefront gradient
            if np.linalg.norm(flow_dir) < 0.1:
                gradient = self._calculate_wavefront_gradient(current_x, current_y)
                if np.linalg.norm(gradient) > 0.1:
                    flow_dir = -gradient  # Negative because we're going toward lower values
            
            # If still no clear direction, use wavefront grid directly
            if np.linalg.norm(flow_dir) < 0.1:
                # Find neighbor with lowest wavefront value
                min_val = float('inf')
                next_x, next_y = current_x, current_y
                
                for dx, dy in self.directions:
                    new_x, new_y = current_x + dx, current_y + dy
                    
                    if (0 <= new_x < self.grid_width and 
                        0 <= new_y < self.grid_height and
                        self.wavefront_grid[new_y, new_x] != -1 and
                        self.occupancy_grid[new_y, new_x] == 0 and
                        (new_x, new_y) not in visited):
                        
                        if self.wavefront_grid[new_y, new_x] < min_val:
                            min_val = self.wavefront_grid[new_y, new_x]
                            next_x, next_y = new_x, new_y
                
                # If stuck, break path planning
                if next_x == current_x and next_y == current_y:
                    # Try with a larger search radius before giving up
                    found_path = self._handle_stuck_point(current_x, current_y, visited, goal_grid_x, goal_grid_y)
                    if found_path:
                        path.extend(found_path[1:])  # Skip the first point as it's already in the path
                    break
                
                current_x, current_y = next_x, next_y
            else:
                # Normalize flow direction
                flow_dir = flow_dir / np.linalg.norm(flow_dir)
                
                # Find the neighbor that best matches flow direction
                best_match = None
                best_alignment = -1.0
                
                for dx, dy in self.directions:
                    new_x, new_y = current_x + dx, current_y + dy
                    
                    if (0 <= new_x < self.grid_width and 
                        0 <= new_y < self.grid_height and
                        self.wavefront_grid[new_y, new_x] != -1 and
                        self.occupancy_grid[new_y, new_x] == 0 and
                        (new_x, new_y) not in visited):
                        
                        # Calculate alignment between flow direction and this neighbor
                        neighbor_dir = np.array([dx, dy])
                        neighbor_dir = neighbor_dir / np.linalg.norm(neighbor_dir)
                        alignment = np.dot(flow_dir, neighbor_dir)
                        
                        # Weight alignment by wavefront improvement
                        wavefront_improvement = max(0, self.wavefront_grid[current_y, current_x] - self.wavefront_grid[new_y, new_x])
                        score = alignment + wavefront_improvement * 0.1
                        
                        if score > best_alignment:
                            best_alignment = score
                            best_match = (new_x, new_y)
                
                # If stuck, handle it
                if best_match is None:
                    # Try with a larger search radius before giving up
                    found_path = self._handle_stuck_point(current_x, current_y, visited, goal_grid_x, goal_grid_y)
                    if found_path:
                        path.extend(found_path[1:])  # Skip the first point as it's already in the path
                    break
                    
                current_x, current_y = best_match
            
            # Track visited points
            visited.add((current_x, current_y))
            
            # Check if we're making progress
            current_dist = self.wavefront_grid[current_y, current_x]
            if current_dist < last_best_dist - min_dist_improvement:
                last_best_dist = current_dist
                steps_since_improvement = 0
            else:
                steps_since_improvement += 1
                
            # If we're not making progress for a while, try to escape
            if steps_since_improvement > 5:
                # Try random exploration
                escape_successful = self._try_escape_local_minimum(current_x, current_y, visited)
                if not escape_successful:
                    # If random exploration fails, try more extreme measures
                    found_path = self._handle_stuck_point(current_x, current_y, visited, goal_grid_x, goal_grid_y)
                    if found_path:
                        path.extend(found_path[1:])  # Skip the first point as it's already in the path
                        break
                
                # Reset progress tracking
                steps_since_improvement = 0
            
            # Convert current point to world coordinates and add to path
            world_x, world_y = self.grid_to_world(current_x, current_y)
            path.append((world_x, world_y))
        
        # If we exited without reaching the goal, add the goal point
        if path and (path[-1][0] != goal_grid_x or path[-1][1] != goal_grid_y):
            world_x, world_y = self.grid_to_world(goal_grid_x, goal_grid_y)
            path.append((world_x, world_y))
        
        # Smooth the resulting path
        return self.smooth_path(path)
        
    def _try_escape_local_minimum(self, x, y, visited, max_attempts=3):
        """Try to escape a local minimum by random exploration."""
        current_x, current_y = x, y
        
        for _ in range(max_attempts):
            # Find unvisited neighbors
            unvisited_neighbors = []
            for dx, dy in self.directions:
                new_x, new_y = current_x + dx, current_y + dy
                
                if (0 <= new_x < self.grid_width and 
                    0 <= new_y < self.grid_height and
                    self.wavefront_grid[new_y, new_x] != -1 and
                    self.occupancy_grid[new_y, new_x] == 0 and
                    (new_x, new_y) not in visited):
                    
                    unvisited_neighbors.append((new_x, new_y))
            
            if not unvisited_neighbors:
                return False
                
            # Choose a random neighbor
            next_x, next_y = random.choice(unvisited_neighbors)
            visited.add((next_x, next_y))
            
            # Check if this point has any unvisited neighbors with lower wavefront value
            for dx, dy in self.directions:
                nx, ny = next_x + dx, next_y + dy
                
                if (0 <= nx < self.grid_width and 
                    0 <= ny < self.grid_height and
                    self.wavefront_grid[ny, nx] != -1 and
                    self.occupancy_grid[ny, nx] == 0 and
                    (nx, ny) not in visited and
                    self.wavefront_grid[ny, nx] < self.wavefront_grid[current_y, current_x]):
                    
                    # Found a promising direction
                    return True
            
            # Move to the random neighbor and try again
            current_x, current_y = next_x, next_y
        
        return False
        
    def _handle_stuck_point(self, x, y, visited, goal_x, goal_y):
        """
        Handle a stuck point by trying various strategies:
        1. Look further in neighborhood
        2. Try A* search from current point
        3. Try to find any valid path to goal
        """
        # Strategy 1: Look in a larger neighborhood
        for search_radius in [3, 5, 7]:
            # Find any point with lower wavefront value in the larger neighborhood
            min_val = self.wavefront_grid[y, x]
            best_x, best_y = None, None
            
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    nx, ny = x + dx, y + dy
                    
                    if (0 <= nx < self.grid_width and 
                        0 <= ny < self.grid_height and
                        self.wavefront_grid[ny, nx] != -1 and
                        self.occupancy_grid[ny, nx] == 0 and
                        (nx, ny) not in visited):
                        
                        if self.wavefront_grid[ny, nx] < min_val:
                            min_val = self.wavefront_grid[ny, nx]
                            best_x, best_y = nx, ny
            
            if best_x is not None:
                # Found a better point, create a direct path to it
                world_x, world_y = self.grid_to_world(best_x, best_y)
                
                # Now compute path from this point to goal
                current_world_x, current_world_y = self.grid_to_world(best_x, best_y)
                goal_world_x, goal_world_y = self.grid_to_world(goal_x, goal_y)
                
                # Convert back to path (single point for now)
                return [(current_world_x, current_world_y)]
        
        # If we get here, we couldn't find a good escape path
        return None
        
    def _heuristic(self, x1, y1, x2, y2):
        """
        Calculate heuristic distance between two grid points.
        Uses Euclidean distance.
        """
        return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
    def compute_attractive_force(self, current_pos, goal_pos):
        """
        Compute attractive force towards goal.
        Enhanced to provide better convergence characteristics.
        """
        dx = goal_pos[0] - current_pos[0]
        dy = goal_pos[1] - current_pos[1]
        dist = np.sqrt(dx*dx + dy*dy)
        
        # Attractive force
        if dist <= self.resolution:
            return np.zeros(2)
            
        # Use quadratic attractive potential within d_star
        if dist <= self.d_star:
            force_magnitude = self.k_att * dist
        else:
            # Use conic attractive potential beyond d_star
            force_magnitude = self.k_att * self.d_star * dist/abs(dist)
            
        # Normalize direction vector
        direction = np.array([dx/dist, dy/dist])
            
        # Apply goal bias if we're having trouble making progress
        if self.progress_stagnation_count > self.local_minima_detection_threshold:
            # Gradually increase goal bias with stagnation
            goal_bias = self.goal_bias_factor * (1.0 + self.progress_stagnation_count / 10.0)
            force_magnitude = max(force_magnitude, self.k_att * dist * goal_bias)
            
        return force_magnitude * direction
    
    def compute_repulsive_force(self, current_pos, obstacles, obstacle_sizes=None):
        """
        Compute repulsive force from obstacles, considering their sizes.
        Enhanced to use distance fields for better clearance detection.
        
        Parameters:
        -----------
        current_pos : np.array
            Current position [x, y]
        obstacles : list of np.array
            List of obstacle positions [[x1, y1], [x2, y2], ...]
        obstacle_sizes : list of float
            List of obstacle widths in meters
            
        Returns:
        --------
        np.array
            Repulsive force vector [fx, fy]
        """
        if not obstacles:
            return np.zeros(2)
            
        total_force = np.zeros(2)
        
        # Debug information
        forces = []
        
        # Calculate robot "radius" (half the diagonal)
        # Use a smaller effective radius to reduce repulsion extent
        robot_radius = math.sqrt(self.robot_length**2 + self.robot_width**2) / 2.5  # Reduced from 2.0
        
        # Convert current position to grid coordinates for wavefront planner
        grid_x, grid_y = self.world_to_grid(current_pos)
        
        # Check if we're in a passage - use clearance field if available
        in_passage = False
        passage_clearance = 0.0
        passage_direction = np.zeros(2)
        
        if (0 <= grid_x < self.grid_width and 
            0 <= grid_y < self.grid_height and 
            self.distance_field is not None):
            
            # Get distance to nearest obstacle
            clearance = self.distance_field[grid_y, grid_x]
            min_required_clearance = min(self.robot_width, self.robot_length) / 2.0 + self.safety_margin
            
            if clearance < min_required_clearance * 2.0:  # Within 2x of minimum clearance
                # We might be in a passage
                # Check if we have a flow field to get passage direction
                if (self.flow_field is not None and
                    self.wavefront_grid is not None and
                    self.wavefront_grid[grid_y, grid_x] != -1):
                    
                    # Get flow direction at current position
                    flow_dir = self.flow_field[grid_y, grid_x]
                    if np.linalg.norm(flow_dir) > 0.1:
                        in_passage = True
                        passage_clearance = clearance
                        passage_direction = flow_dir / np.linalg.norm(flow_dir)
        
        for i, obstacle in enumerate(obstacles):
            # Get obstacle size
            obstacle_size = 0.3  # Default size
            if obstacle_sizes is not None and i < len(obstacle_sizes):
                obstacle_size = obstacle_sizes[i]
                
            # Half obstacle size (effectively its "radius") - with scaling
            obstacle_radius = obstacle_size / 2.5  # Reduced to decrease repulsion radius
            
            # Calculate center-to-center vector
            dx = current_pos[0] - obstacle[0]
            dy = current_pos[1] - obstacle[1]
            center_dist = np.sqrt(dx*dx + dy*dy)
            
            # Calculate the actual distance between robot and obstacle edges
            # by subtracting both "radii" from the center distance - with reduced effect
            edge_dist = center_dist - (robot_radius + obstacle_radius)
            
            # Ensure we don't have negative distances
            edge_dist = max(0.01, edge_dist)  # Small positive value to avoid division by zero
            
            # Apply repulsive force if within influence radius
            if edge_dist < self.rho_0:
                # Normalized direction vector
                if center_dist > 0:
                    nx = dx / center_dist
                    ny = dy / center_dist
                else:
                    # If centers are exactly at same point (should be rare), 
                    # push in a random direction
                    nx, ny = 1.0, 0.0
                
                # Calculate repulsive force with modified strength based on distance
                # The closer the obstacle, the stronger the repulsion - but with softer falloff
                force_magnitude = self.k_rep * ((1/edge_dist) - (1/self.rho_0))**2 / (edge_dist*edge_dist)
                
                # Apply stronger repulsion for very close obstacles - but less aggressively
                if edge_dist < 0.3:  # Reduced from 0.5
                    force_magnitude *= 1.5  # Reduced multiplier from 2.0 to 1.5
                
                # Scale force based on obstacle size - but with reduced scaling factor
                size_factor = 1.0 + (obstacle_size - 0.3) / 0.6  # 1.0 for 0.3m, 1.5 for 0.6m
                force_magnitude *= max(1.0, size_factor)
                
                # If we're in a passage, adjust force direction to align with passage
                if in_passage:
                    # Calculate how much the repulsive force aligns with passage direction
                    # Higher alignment = less modification needed
                    obstacle_direction = np.array([nx, ny])
                    alignment = np.dot(obstacle_direction, passage_direction)
                    
                    # If force is perpendicular to passage, reduce its magnitude based on clearance
                    if abs(alignment) < 0.3:  # Nearly perpendicular
                        # Calculate clearance fitness (1.0 = very wide passage, 0.1 = narrow passage)
                        min_required = min(self.robot_width, self.robot_length) / 2.0 + self.safety_margin
                        clearance_fitness = min(1.0, max(0.1, (passage_clearance - min_required) / min_required))
                        
                        # Reduce force magnitude based on clearance
                        force_magnitude *= max(0.2, 1.0 - clearance_fitness)
                
                # Add angle variation if we've detected oscillation
                if self.oscillation_count > 2:
                    # Add a slight rotation to the repulsive force to break symmetry
                    angle_variation = 0.1 * self.oscillation_count * math.pi / 180.0  # Convert to radians
                    cos_angle = math.cos(angle_variation)
                    sin_angle = math.sin(angle_variation)
                    nx_rotated = nx * cos_angle - ny * sin_angle
                    ny_rotated = nx * sin_angle + ny * cos_angle
                    nx, ny = nx_rotated, ny_rotated
                
                force = force_magnitude * np.array([nx, ny])
                total_force += force
                
                # Store for debugging
                forces.append((obstacle, edge_dist, force))
        
        # If we have a previous force, check for oscillation
        if self.prev_force is not None:
            # Oscillation detected if current force is in opposite direction to previous
            dot_product = np.dot(total_force, self.prev_force)
            force_magnitude = np.linalg.norm(total_force) * np.linalg.norm(self.prev_force)
            
            if force_magnitude > 0:
                normalized_dot = dot_product / force_magnitude
                
                # If forces are pointing in opposite directions (angle > 90 degrees)
                if normalized_dot < 0:
                    self.oscillation_count += 1
                else:
                    # Reset oscillation count gradually
                    self.oscillation_count = max(0, self.oscillation_count - 0.5)
        
        # Store current force for next iteration
        self.prev_force = total_force.copy()
                
        # Publish debug visualization if publisher is available
        if self.debug_publisher is not None and forces:
            self.publish_force_visualization(current_pos, forces)
                
        return total_force
    
    def apply_wavefront_guidance(self, current_pos, goal_pos, total_force=None):
        """
        Enhanced function to use wavefront grid to guide force, especially when stuck in local minima.
        Uses distance field for clearance-based guidance weight.
        
        Parameters:
        -----------
        current_pos : np.array
            Current position [x, y]
        goal_pos : np.array
            Goal position [x, y]
        total_force : np.array, optional
            Current total force, if available
            
        Returns:
        --------
        np.array
            Modified force incorporating wavefront guidance
        """
        # If no wavefront grid or total force, return zero force
        if self.wavefront_grid is None:
            return np.zeros(2) if total_force is None else total_force
            
        # Convert current position to grid coordinates
        grid_x, grid_y = self.world_to_grid(current_pos)
        
        # Check if position is within grid and has a valid wavefront value
        if (0 <= grid_x < self.grid_width and 
            0 <= grid_y < self.grid_height and 
            self.wavefront_grid[grid_y, grid_x] != -1):
            
            # Calculate clearance-based wavefront influence
            # Narrow passages get more wavefront influence, open areas get more APF influence
            clearance_factor = 0.5  # Default factor
            
            if self.distance_field is not None:
                clearance = self.distance_field[grid_y, grid_x]
                # Higher clearance = lower wavefront influence
                clearance_factor = max(0.1, min(1.0, 1.5 - clearance))
            
            # Get flow direction at current position
            if self.flow_field is not None:
                flow_dir = self.flow_field[grid_y, grid_x]
                
                # If flow direction is significant, use it
                if np.linalg.norm(flow_dir) > 0.1:
                    # Normalize flow direction
                    flow_dir = flow_dir / np.linalg.norm(flow_dir)
                    
                    # Create guidance force based on flow direction
                    # Scale with k_att and distance to goal
                    dist_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(current_pos))
                    guidance_force = flow_dir * self.k_att * min(dist_to_goal, self.d_star)
                    
                    # Weight guidance based on 3 factors:
                    # 1. Progress stagnation (more stuck = more guidance)
                    # 2. Clearance (less clearance = more guidance)
                    # 3. Distance to goal (closer to goal = less guidance)
                    progress_weight = min(5.0, self.progress_stagnation_count / 2.0) / 10.0
                    goal_proximity_factor = max(0.3, min(1.0, dist_to_goal / (2.0 * self.d_star)))
                    
                    guidance_weight = (0.1 + progress_weight) * clearance_factor * goal_proximity_factor
                    
                    # Combine with existing force if provided
                    if total_force is not None:
                        # Calculate how much the existing force conflicts with the guidance
                        force_mag = np.linalg.norm(total_force)
                        if force_mag > 0:
                            force_dir = total_force / force_mag
                            alignment = np.dot(force_dir, flow_dir)
                            
                            # If forces are very opposed, give more weight to guidance
                            if alignment < -0.5:  # More than 120 degrees apart
                                guidance_weight = min(0.9, guidance_weight * 2.0)  # Boost guidance but cap at 0.9
                        
                        return (1.0 - guidance_weight) * total_force + guidance_weight * guidance_force
                    else:
                        return guidance_force
                    
            # If no flow field, try to use wavefront gradient directly
            gradient = self._calculate_wavefront_gradient(grid_x, grid_y)
            if np.linalg.norm(gradient) > 0.1:
                # Gradient points uphill, we want downhill
                guidance_dir = -gradient
                dist_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(current_pos))
                guidance_force = guidance_dir * self.k_att * min(dist_to_goal, self.d_star)
                
                # Apply similar weighting as above
                progress_weight = min(5.0, self.progress_stagnation_count / 2.0) / 10.0
                guidance_weight = (0.1 + progress_weight) * clearance_factor
                
                if total_force is not None:
                    return (1.0 - guidance_weight) * total_force + guidance_weight * guidance_force
                else:
                    return guidance_force
        
        # If we couldn't get wavefront guidance, return original force
        return np.zeros(2) if total_force is None else total_force
    
    def handle_local_minima(self, current_pos, goal_pos, step_size, obstacles, obstacle_sizes):
        """
        Apply special strategies to escape local minima.
        Enhanced to use flow field as a primary escape mechanism and respect grid constraints.
        
        Parameters:
        -----------
        current_pos : np.array
            Current position [x, y]
        goal_pos : np.array
            Goal position [x, y]
        step_size : float
            Normal step size for movement
        obstacles : list
            List of obstacle positions
        obstacle_sizes : list
            List of obstacle sizes
            
        Returns:
        --------
        np.array
            New position after applying escape strategy
        """
        # First check if current position is valid on grid
        current_grid_x, current_grid_y = self.world_to_grid(current_pos)
        if not (0 <= current_grid_x < self.grid_width and 
                0 <= current_grid_y < self.grid_height):
            # Current position is outside grid - try to find a valid position
            return current_pos
        
        if self.occupancy_grid[current_grid_y, current_grid_x] == 1:
            # Current position is in an obstacle - find safe point
            safe_point = self.find_safe_point_nearby(current_grid_x, current_grid_y)
            if safe_point:
                wx, wy = self.grid_to_world(*safe_point)
                return np.array([wx, wy])
            # If no safe point found, stay at current position
            return current_pos
        
        # Strategy 1: Use flow field from wavefront to guide escape
        if self.flow_field is not None and self.wavefront_grid is not None:
            # Get flow direction at current position
            flow_dir = self.flow_field[current_grid_y, current_grid_x]
            
            # If flow direction is significant, use it
            if np.linalg.norm(flow_dir) > 0.1:
                # Normalize flow direction
                flow_dir = flow_dir / np.linalg.norm(flow_dir)
                
                # Use flow direction to calculate next position
                next_pos = current_pos + flow_dir * step_size
                next_grid_x, next_grid_y = self.world_to_grid(next_pos)
                
                # Check if next position is valid on grid
                if (0 <= next_grid_x < self.grid_width and 
                    0 <= next_grid_y < self.grid_height and
                    self.occupancy_grid[next_grid_y, next_grid_x] == 0):
                    # Also check for collision
                    if not self.check_collision(next_pos, obstacles, obstacle_sizes):
                        self.current_escapestrategy = (self.current_escapestrategy + 1) % 4
                        return next_pos
            
            # If flow direction isn't useful, try to follow wavefront gradient
            gradient = self._calculate_wavefront_gradient(current_grid_x, current_grid_y)
            if np.linalg.norm(gradient) > 0.1:
                # Gradient points uphill, we want downhill
                guidance_dir = -gradient / np.linalg.norm(gradient)
                next_pos = current_pos + guidance_dir * step_size
                next_grid_x, next_grid_y = self.world_to_grid(next_pos)
                
                # Check if next position is valid on grid
                if (0 <= next_grid_x < self.grid_width and 
                    0 <= next_grid_y < self.grid_height and
                    self.occupancy_grid[next_grid_y, next_grid_x] == 0):
                    # Check for collision
                    if not self.check_collision(next_pos, obstacles, obstacle_sizes):
                        self.current_escapestrategy = (self.current_escapestrategy + 1) % 4
                        return next_pos
            
            # Look for a passage or open area nearby
            if self.local_minima_regions:
                # Find nearest passage point
                nearest_passage = None
                min_dist = float('inf')
                
                for passage in self.local_minima_regions:
                    passage_pos = passage['position']
                    dist = np.linalg.norm(np.array(passage_pos) - current_pos)
                    
                    if dist < min_dist and dist < self.rho_0 * 2.0:  # Within range
                        # Make sure passage point is in free grid cell
                        pass_grid_x, pass_grid_y = self.world_to_grid(passage_pos)
                        if (0 <= pass_grid_x < self.grid_width and 
                            0 <= pass_grid_y < self.grid_height and
                            self.occupancy_grid[pass_grid_y, pass_grid_x] == 0):
                            min_dist = dist
                            nearest_passage = passage_pos
                
                if nearest_passage:
                    # Move toward passage
                    direction = np.array(nearest_passage) - current_pos
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction) * step_size
                        next_pos = current_pos + direction
                        next_grid_x, next_grid_y = self.world_to_grid(next_pos)
                        
                        # Check if next position is valid on grid
                        if (0 <= next_grid_x < self.grid_width and 
                            0 <= next_grid_y < self.grid_height and
                            self.occupancy_grid[next_grid_y, next_grid_x] == 0):
                            # Check for collision
                            if not self.check_collision(next_pos, obstacles, obstacle_sizes):
                                self.current_escapestrategy = (self.current_escapestrategy + 1) % 4
                                return next_pos
        
        # Try other strategies in sequence based on current escape attempt
        self.escape_attempt_count += 1
        strategy = (self.current_escapestrategy) % 4
        
        # Strategy 2: Random perturbation
        if strategy == 0:
            for _ in range(5):  # Try a few random directions
                # Add random noise to current position
                noise_x = (random.random() * 2 - 1) * self.noise_amplitude
                noise_y = (random.random() * 2 - 1) * self.noise_amplitude
                noise = np.array([noise_x, noise_y])
                
                # Scale noise with distance to goal to avoid overshooting
                dist_to_goal = np.linalg.norm(goal_pos - current_pos)
                noise_scale = min(1.0, dist_to_goal / (2 * step_size))
                
                # Apply noise as displacement
                next_pos = current_pos + noise * noise_scale
                next_grid_x, next_grid_y = self.world_to_grid(next_pos)
                
                # Check if next position is valid on grid
                if (0 <= next_grid_x < self.grid_width and 
                    0 <= next_grid_y < self.grid_height and
                    self.occupancy_grid[next_grid_y, next_grid_x] == 0):
                    # Check if next position collides with obstacles
                    if not self.check_collision(next_pos, obstacles, obstacle_sizes):
                        self.current_escapestrategy = (self.current_escapestrategy + 1) % 4
                        return next_pos
        
        # Strategy 3: Direct path to goal attempt
        elif strategy == 1:
            # Try to move directly toward goal
            disp = goal_pos - current_pos
            dist = np.linalg.norm(disp)
            
            if dist > step_size:
                disp = disp / dist * step_size
            
            next_pos = current_pos + disp
            next_grid_x, next_grid_y = self.world_to_grid(next_pos)
            
            # Check if next position is valid on grid
            if (0 <= next_grid_x < self.grid_width and 
                0 <= next_grid_y < self.grid_height and
                self.occupancy_grid[next_grid_y, next_grid_x] == 0):
                # Check if this direct path is collision-free
                if not self.check_collision(next_pos, obstacles, obstacle_sizes):
                    self.current_escapestrategy = (self.current_escapestrategy + 1) % 4
                    return next_pos
        
        # Strategy 4: Wall following
        elif strategy == 2:
            # Find grid cells with obstacles nearby
            for search_radius in [1, 2, 3]:
                for dy in range(-search_radius, search_radius + 1):
                    for dx in range(-search_radius, search_radius + 1):
                        if dx == 0 and dy == 0:
                            continue
                        
                        nx, ny = current_grid_x + dx, current_grid_y + dy
                        if (0 <= nx < self.grid_width and 
                            0 <= ny < self.grid_height and
                            self.occupancy_grid[ny, nx] == 1):
                            
                            # Found obstacle - follow its edge
                            # Get vector from obstacle to current position
                            to_robot_x = current_grid_x - nx
                            to_robot_y = current_grid_y - ny
                            
                            # Calculate perpendicular direction (wall following)
                            if self.current_escapestrategy % 2 == 0:  # Clockwise
                                perp_x, perp_y = to_robot_y, -to_robot_x
                            else:  # Counter-clockwise
                                perp_x, perp_y = -to_robot_y, to_robot_x
                                
                            # Normalize to unit step
                            mag = math.sqrt(perp_x*perp_x + perp_y*perp_y)
                            if mag > 0:
                                perp_x, perp_y = perp_x/mag, perp_y/mag
                            
                            # Calculate target grid cell
                            target_grid_x = current_grid_x + int(round(perp_x))
                            target_grid_y = current_grid_y + int(round(perp_y))
                            
                            # Check if target is valid
                            if (0 <= target_grid_x < self.grid_width and 
                                0 <= target_grid_y < self.grid_height and
                                self.occupancy_grid[target_grid_y, target_grid_x] == 0):
                                
                                # Convert to world coordinates
                                world_x, world_y = self.grid_to_world(target_grid_x, target_grid_y)
                                next_pos = np.array([world_x, world_y])
                                
                                # Check for collision
                                if not self.check_collision(next_pos, obstacles, obstacle_sizes):
                                    self.current_escapestrategy = (self.current_escapestrategy + 1) % 4
                                    return next_pos
                            
                            # If first attempt failed, try different directions
                            for test_dx, test_dy in [(1,0), (0,1), (-1,0), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
                                test_x = current_grid_x + test_dx
                                test_y = current_grid_y + test_dy
                                
                                if (0 <= test_x < self.grid_width and 
                                    0 <= test_y < self.grid_height and
                                    self.occupancy_grid[test_y, test_x] == 0):
                                    
                                    # Convert to world coordinates
                                    world_x, world_y = self.grid_to_world(test_x, test_y)
                                    next_pos = np.array([world_x, world_y])
                                    
                                    # Check for collision
                                    if not self.check_collision(next_pos, obstacles, obstacle_sizes):
                                        self.current_escapestrategy = (self.current_escapestrategy + 1) % 4
                                        return next_pos
        
        # Strategy 5: Random walk with goal bias
        elif strategy == 3:
            self.random_walk_active = True
            
            # Try all 8 directions in the grid
            directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (-1, 1), (1, -1), (-1, -1)
            ]
            
            # Calculate direction to goal for biasing
            goal_grid_x, goal_grid_y = self.world_to_grid(goal_pos)
            goal_dx = goal_grid_x - current_grid_x
            goal_dy = goal_grid_y - current_grid_y
            goal_dist = max(1, math.sqrt(goal_dx*goal_dx + goal_dy*goal_dy))
            goal_dir_x, goal_dir_y = goal_dx/goal_dist, goal_dy/goal_dist
            
            # Weight directions by alignment with goal
            weighted_dirs = []
            for dx, dy in directions:
                # Calculate alignment with goal direction
                dir_mag = math.sqrt(dx*dx + dy*dy)
                if dir_mag > 0:
                    alignment = (dx*goal_dir_x + dy*goal_dir_y) / dir_mag
                    weight = max(0, alignment) + 0.5  # Ensure all directions have chance
                else:
                    weight = 0.5
                    
                # Check if the direction leads to a valid cell
                nx, ny = current_grid_x + dx, current_grid_y + dy
                if (0 <= nx < self.grid_width and 
                    0 <= ny < self.grid_height and
                    self.occupancy_grid[ny, nx] == 0):
                    weighted_dirs.append((dx, dy, weight))
            
            # Sort by weight in descending order
            weighted_dirs.sort(key=lambda x: x[2], reverse=True)
            
            # Try directions in order of preference
            for dx, dy, _ in weighted_dirs:
                nx, ny = current_grid_x + dx, current_grid_y + dy
                world_x, world_y = self.grid_to_world(nx, ny)
                next_pos = np.array([world_x, world_y])
                
                # Check for collision
                if not self.check_collision(next_pos, obstacles, obstacle_sizes):
                    self.current_escapestrategy = (self.current_escapestrategy + 1) % 4
                    return next_pos
        
        # If all else fails, try to find any valid neighboring cell
        for dx, dy in [(0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]:
            nx, ny = current_grid_x + dx, current_grid_y + dy
            if (0 <= nx < self.grid_width and 
                0 <= ny < self.grid_height and
                self.occupancy_grid[ny, nx] == 0):
                world_x, world_y = self.grid_to_world(nx, ny)
                return np.array([world_x, world_y])
        
        # If absolutely nothing works, return the original position
        self.current_escapestrategy = (self.current_escapestrategy + 1) % 4
        return current_pos

    def check_collision(self, position, obstacles, obstacle_sizes):
        """
        Check if position collides with any obstacle.
        
        Parameters:
        -----------
        position : np.array
            Position to check [x, y]
        obstacles : list
            List of obstacle positions
        obstacle_sizes : list
            List of obstacle sizes
            
        Returns:
        --------
        bool
            True if collision detected, False otherwise
        """
        # Early return if no obstacles
        if obstacles is None or len(obstacles) == 0:
            # Still check occupancy grid
            if self.occupancy_grid is not None:
                grid_x, grid_y = self.world_to_grid(position)
                
                if (0 <= grid_x < self.grid_width and 
                    0 <= grid_y < self.grid_height and
                    self.occupancy_grid[grid_y, grid_x] == 1):
                    return True
            return False
        
        # Calculate robot radius
        robot_radius = math.sqrt(self.robot_length**2 + self.robot_width**2) / 2.0
        
        # Check each obstacle
        for i, obs in enumerate(obstacles):
            # Get obstacle position (only x, y coordinates)
            if isinstance(obs, (list, tuple, np.ndarray)):
                if len(obs) >= 2:
                    obs_pos = np.array(obs[:2])
                else:
                    continue  # Invalid obstacle format
            else:
                # Try to handle other formats
                try:
                    obs_pos = np.array([obs.x, obs.y])
                except:
                    continue  # Can't extract position
            
            # Calculate distance between robot and obstacle centers
            try:
                dist = np.linalg.norm(position - obs_pos)
            except:
                continue  # Can't calculate distance
            
            # Get obstacle size
            obs_size = 0.3  # Default
            if obstacle_sizes is not None and i < len(obstacle_sizes):
                obs_size = obstacle_sizes[i]
            
            # Calculate minimum safe distance
            min_dist = robot_radius + obs_size/2 + self.safety_margin
            
            # Check for collision
            if dist < min_dist:
                return True
                
        # Handle occupancy grid collision check if available
        if self.occupancy_grid is not None:
            grid_x, grid_y = self.world_to_grid(position)
            
            if (0 <= grid_x < self.grid_width and 
                0 <= grid_y < self.grid_height and
                self.occupancy_grid[grid_y, grid_x] == 1):
                return True
                
        return False
        
    def publish_force_visualization(self, current_pos, forces):
        """Visualize repulsive forces for debugging."""
        if self.debug_publisher is None:
            return
            
        # Throttle visualization to avoid overwhelming ROS
        current_time = time.time()
        if current_time - self.last_visualization_time < self.visualization_interval:
            return
            
        self.last_visualization_time = current_time
            
        marker_array = MarkerArray()
        
        # Show current position
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rclpy.time.Time().to_msg()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.id = 0
        marker.pose.position.x = current_pos[0]
        marker.pose.position.y = current_pos[1]
        marker.pose.position.z = 0.0
        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.2
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker_array.markers.append(marker)
        
        # Show forces
        for i, (obstacle, dist, force) in enumerate(forces):
            # Force vector
            arrow = Marker()
            arrow.header.frame_id = "map"
            arrow.header.stamp = rclpy.time.Time().to_msg()
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.id = i + 1
            
            # Start point
            arrow.points.append(Point(x=current_pos[0], y=current_pos[1], z=0.0))
            
            # Scale the arrow for visibility
            scale = min(1.0, max(0.1, np.linalg.norm(force) / 10.0))
            
            # End point
            arrow.points.append(Point(
                x=current_pos[0] + force[0] * scale,
                y=current_pos[1] + force[1] * scale,
                z=0.0
            ))
            
            arrow.scale.x = 0.05  # Shaft diameter
            arrow.scale.y = 0.1   # Head diameter
            arrow.scale.z = 0.1   # Head length
            
            # Color based on distance (red for close, blue for far)
            norm_dist = min(1.0, dist / self.rho_0)
            arrow.color.a = 1.0
            arrow.color.r = 1.0 - norm_dist
            arrow.color.g = 0.0
            arrow.color.b = norm_dist
            
            marker_array.markers.append(arrow)
            
        # Add status text
        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = rclpy.time.Time().to_msg()
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.id = len(forces) + 1
        text_marker.pose.position.x = current_pos[0]
        text_marker.pose.position.y = current_pos[1]
        text_marker.pose.position.z = 1.0
        
        # Status text
        status = f"Osc: {self.oscillation_count}, Stag: {self.progress_stagnation_count}"
        if self.random_walk_active:
            status += " (Random Walk)"
            
        text_marker.text = status
        text_marker.scale.z = 0.2
        text_marker.color.a = 1.0
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        
        marker_array.markers.append(text_marker)
        
        # Add wavefront info if available
        if self.wavefront_grid is not None:
            wave_marker = Marker()
            wave_marker.header.frame_id = "map"
            wave_marker.header.stamp = rclpy.time.Time().to_msg()
            wave_marker.type = Marker.TEXT_VIEW_FACING
            wave_marker.action = Marker.ADD
            wave_marker.id = len(forces) + 2
            wave_marker.pose.position.x = current_pos[0]
            wave_marker.pose.position.y = current_pos[1] - 0.5  # Below the status text
            wave_marker.pose.position.z = 1.0
            
            # Check if position is in wavefront grid
            grid_x, grid_y = self.world_to_grid(current_pos)
            if (0 <= grid_x < self.grid_width and 
                0 <= grid_y < self.grid_height and
                self.wavefront_grid[grid_y, grid_x] != -1):
                
                wave_val = self.wavefront_grid[grid_y, grid_x]
                wave_marker.text = f"Wavefront: {wave_val:.1f}"
            else:
                wave_marker.text = "Wavefront: N/A"
                
            wave_marker.scale.z = 0.2
            wave_marker.color.a = 1.0
            wave_marker.color.r = 0.2
            wave_marker.color.g = 0.7
            wave_marker.color.b = 1.0
            
            marker_array.markers.append(wave_marker)
        
        self.debug_publisher.publish(marker_array)
    
    def _is_passage_between_obstacles(self, x, y, distance_to_obstacles):
        """
        Improved passage detection between obstacles.
        Ensures connected obstacles are properly identified.
        Returns (is_passage, clearance, passage_direction) where:
        - is_passage: True if point is in a passage between obstacles
        - clearance: The clearance width at this point
        - passage_direction: Direction vector of the passage (parallel to walls)
        """
        if not self.enable_local_minima_optimization:
            return False, 0.0, np.zeros(2)
            
        # Calculate minimum required clearance for robot to pass
        min_required_clearance = min(self.robot_width, self.robot_length) + self.safety_margin
            
        # Get distance to the nearest obstacle at this point
        current_dist = distance_to_obstacles[y, x]
        
        # If too close to an obstacle, this can't be a passage
        if current_dist < min_required_clearance / 2:
            return False, 0.0, np.zeros(2)
            
        # Search in multiple directions to detect parallel obstacles
        # Check more directions to better detect passages at various angles
        directions = [(1, 0), (0, 1), (1, 1), (1, -1), 
                    (2, 1), (1, 2), (2, -1), (-1, 2)]  # Added more diagonal directions
        
        best_passage = None
        best_clearance = 0.0
        best_direction = np.zeros(2)
        
        for dx, dy in directions:
            # Normalize direction
            dir_len = math.sqrt(dx*dx + dy*dy)
            dx_norm, dy_norm = dx/dir_len, dy/dir_len
            
            # Search perpendicular to this direction
            perp_dx, perp_dy = -dy_norm, dx_norm
            
            # Find obstacles in opposite directions perpendicular to current direction
            obstacle_found = [False, False]  # Left and right
            obstacle_dist = [float('inf'), float('inf')]  # Distance to left and right obstacles
            
            # Search up to a maximum distance in both perpendicular directions
            max_search_dist = min(int(self.rho_0 / self.resolution * 3), 30)  # Increased search distance
            
            # Flag to track if we found continuous obstacles (walls)
            continuous_obstacles = [True, True]
            obstacle_gaps = [0, 0]  # Count gaps in obstacles
            
            for d in range(1, max_search_dist):
                # Check left side
                left_x = int(x + perp_dx * d)
                left_y = int(y + perp_dy * d)
                
                if (0 <= left_x < self.grid_width and 0 <= left_y < self.grid_height):
                    if self.occupancy_grid[left_y, left_x] == 1:  # Found obstacle
                        if not obstacle_found[0]:
                            obstacle_found[0] = True
                            obstacle_dist[0] = d * self.resolution
                        continuous_obstacles[0] = True
                    else:
                        # If we already found an obstacle but now found a gap, 
                        # this may be discontinuous obstacles
                        if obstacle_found[0]:
                            continuous_obstacles[0] = False
                            obstacle_gaps[0] += 1
                else:
                    break  # Out of bounds
            
            for d in range(1, max_search_dist):
                # Check right side
                right_x = int(x - perp_dx * d)
                right_y = int(y - perp_dy * d)
                
                if (0 <= right_x < self.grid_width and 0 <= right_y < self.grid_height):
                    if self.occupancy_grid[right_y, right_x] == 1:  # Found obstacle
                        if not obstacle_found[1]:
                            obstacle_found[1] = True
                            obstacle_dist[1] = d * self.resolution
                        continuous_obstacles[1] = True
                    else:
                        # If we already found an obstacle but now found a gap, 
                        # this may be discontinuous obstacles
                        if obstacle_found[1]:
                            continuous_obstacles[1] = False
                            obstacle_gaps[1] += 1
                else:
                    break  # Out of bounds
            
            # If obstacles found on both sides and they are relatively continuous,
            # we're in a passage
            if obstacle_found[0] and obstacle_found[1]:
                # Calculate total clearance
                clearance = obstacle_dist[0] + obstacle_dist[1]
                
                # Evaluate if this is the best passage direction
                # Prefer more continuous walls (fewer gaps) and wider clearance
                passage_quality = clearance - (obstacle_gaps[0] + obstacle_gaps[1]) * 0.2
                
                if best_passage is None or passage_quality > best_passage:
                    best_passage = passage_quality
                    best_clearance = clearance
                    # Store the passage direction (parallel to walls)
                    best_direction = np.array([dx_norm, dy_norm])
        
        if best_passage is not None:
            return True, best_clearance, best_direction
        
        return False, 0.0, np.zeros(2)

    def _calculate_clearance_fitness(self, clearance):
        """
        Enhanced fitness score for passages based on clearance.
        Returns a value between 0.0 and 1.0:
        - 0.0: No passage or not enough clearance
        - 1.0: Wide clearance, easy to pass
        """
        # Calculate minimum required clearance for robot to pass
        min_required_clearance = min(self.robot_width, self.robot_length) + self.safety_margin
        
        # Add a small buffer to avoid borderline cases
        min_required_clearance *= 1.05  # Add 5% buffer
        
        # Normalize clearance to a fitness score
        if clearance < min_required_clearance:
            return 0.0  # Not enough clearance
        
        # Calculate fitness based on clearance - more gradual scaling
        # Fitness increases smoothly from 0.1 to 1.0 as clearance increases
        # with more emphasis on having sufficient clearance
        fitness = min(1.0, 0.1 + 0.9 * (clearance - min_required_clearance) / (min_required_clearance * 2))
        
        return fitness
        
    def _add_repulsive_potential(self):
        """
        Add repulsive potential from obstacles to the potential field with local minima optimization.
        Enhanced to better handle connected obstacles and detect passages.
        """
        if self.potential_field is None:
            return
            
        # Use distance field directly if available
        if self.distance_field is not None:
            distance_to_obstacles = self.distance_field
        else:
            # Create distance transform of occupancy grid if not already available
            # Invert grid (1=obstacle becomes 0, 0=free becomes 1)
            inverted_grid = 1 - self.occupancy_grid
            
            # Calculate distance transform (distance to nearest obstacle)
            distance_to_obstacles = distance_transform_edt(inverted_grid) * self.resolution
        
        # Clear local minima regions list
        self.local_minima_regions = []
        
        # Create passage visualization grid
        self.passage_visualization_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
        
        # Add repulsive potential based on distance transform
        rho_0 = self.rho_0  # Influence radius
        
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Skip obstacles and points with infinite potential
                if self.occupancy_grid[y, x] == 1 or self.potential_field[y, x] == float('inf'):
                    continue
                
                # Get distance to nearest obstacle
                dist = distance_to_obstacles[y, x]
                
                # Only apply repulsion within influence radius
                if dist <= rho_0:
                    # Calculate effective distance considering robot size
                    # Use smaller robot radius to reduce inflation
                    effective_dist = max(0.05, dist - (self.robot_radius * 0.7))
                    
                    # Check if this point is in a passage between obstacles
                    is_passage, clearance, passage_direction = self._is_passage_between_obstacles(x, y, distance_to_obstacles)
                    
                    # Store passage information for visualization
                    if is_passage:
                        fitness = self._calculate_clearance_fitness(clearance)
                        # Scale passage visualization value by fitness
                        self.passage_visualization_grid[y, x] = fitness
                    
                    # Calculate repulsive potential with softer falloff
                    if effective_dist <= rho_0:
                        # Use a smoother quadratic function with less steep falloff near obstacles
                        rep = 0.5 * self.k_rep * ((1/effective_dist) - (1/rho_0))**2
                        
                        # Add extra repulsion only for very close obstacles - reduced effect
                        if effective_dist < 0.3:  # Reduced from 0.5
                            rep *= 1.5  # Reduced multiplier from 2.0 to 1.5
                        
                        # If point is in a passage, adjust repulsive potential based on clearance
                        if is_passage:
                            fitness = self._calculate_clearance_fitness(clearance)
                            
                            if fitness > 0.1:  # If passage is wide enough, reduce repulsion
                                # Record this as a potential path point through a passage
                                world_x, world_y = self.grid_to_world(x, y)
                                self.local_minima_regions.append({
                                    'position': (world_x, world_y),
                                    'clearance': clearance,
                                    'fitness': fitness,
                                    'grid_pos': (x, y),  # Store grid position for flow field usage
                                    'direction': passage_direction  # Store passage direction
                                })
                                
                                # Reduce repulsive potential based on fitness - more reduction for viable passages
                                rep *= max(0.05, 1.0 - fitness * self.repulsive_reduction_factor * 1.5)
                            else:
                                # Not enough clearance, increase repulsion to discourage passage
                                rep *= self.repulsive_increase_factor * 1.2
                                
                        self.potential_field[y, x] += rep
    
    def _get_heat_map_color(self, value, min_val, max_val):
        """
        Get a color from blue (min) to red (max) for the given value.
        Enhanced to produce a more visually distinctive gradient.
        """
        # Normalize value to 0-1
        if max_val == min_val:
            normalized = 0.5
        else:
            normalized = float((value - min_val) / (max_val - min_val))
        
        # Blue to red gradient with improved visual separation
        if normalized < 0.25:
            # Blue to cyan
            r = 0.0
            g = float(normalized * 4.0)
            b = 1.0
        elif normalized < 0.5:
            # Cyan to green
            r = 0.0
            g = 1.0
            b = float(1.0 - (normalized - 0.25) * 4.0)
        elif normalized < 0.75:
            # Green to yellow
            r = float((normalized - 0.5) * 4.0)
            g = 1.0
            b = 0.0
        else:
            # Yellow to red
            r = 1.0
            g = float(1.0 - (normalized - 0.75) * 4.0)
            b = 0.0
        
        return [float(r), float(g), float(b)]
    
    def compute_potential_field(self, start_point, goal_point):
        """
        Enhanced potential field computation with better wavefront integration.
        Combines traditional artificial potential field with wavefront guidance.
        Uses distance fields for better clearance detection.
        """
        # First do wavefront expansion
        wavefront_path = self.wavefront_expansion(start_point, goal_point)
        
        if wavefront_path is None:
            return None
        
        # Create potential field grid
        self.potential_field = np.zeros((self.grid_height, self.grid_width))
        
        # Convert points to grid coordinates
        start_grid_x, start_grid_y = self.world_to_grid(start_point)
        goal_grid_x, goal_grid_y = self.world_to_grid(goal_point)
        
        # Add attractive potential from goal
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Skip obstacles
                if self.occupancy_grid[y, x] == 1:
                    self.potential_field[y, x] = float('inf')
                    continue
                
                # Attractive potential
                dist = math.sqrt((x - goal_grid_x)**2 + (y - goal_grid_y)**2)
                
                # Check if beyond d_star
                if dist <= self.d_star / self.resolution:
                    # quadratic potential
                    attr = 0.5 * self.k_att * dist**2
                else:
                    # conic potential
                    attr = self.d_star / self.resolution * self.k_att * dist - 0.5 * self.k_att * (self.d_star / self.resolution)**2
                
                # Add wavefront influence to avoid local minima - enhanced version
                if self.wavefront_grid is not None and self.wavefront_grid[y, x] != -1:
                    # Calculate gradient of wavefront at this point
                    gradient = self._calculate_wavefront_gradient(x, y)
                    gradient_magnitude = np.linalg.norm(gradient)
                    
                    # Get wavefront distance
                    wave_dist = self.wavefront_grid[y, x]
                    
                    # Calculate clearance-based influence weight
                    # Higher clearance = lower wavefront influence (use APF)
                    # Lower clearance = higher wavefront influence (follow wavefront)
                    if self.distance_field is not None:
                        clearance = self.distance_field[y, x]
                        # Lower clearance gives higher wavefront influence
                        clearance_factor = max(0.1, min(1.0, 1.5 - clearance))
                    else:
                        clearance_factor = 0.5  # Default if no clearance field
                    
                    # Adaptive influence - stronger where gradient is weak (potential local minima)
                    # and in narrow passages (low clearance)
                    if gradient_magnitude < 0.5:
                        # Low gradient may indicate local minimum - increase influence
                        wave_influence = -self.wavefront_influence_max * clearance_factor * self.k_att * wave_dist
                    else:
                        # Normal areas - standard influence
                        wave_influence = -self.wavefront_influence_base * clearance_factor * self.k_att * wave_dist
                        
                    attr += wave_influence
                
                # Set potential
                self.potential_field[y, x] = attr
        
        # Add repulsive potential from obstacles
        self._add_repulsive_potential()
        
        # Extract path from potential field
        path = self.extract_path_from_potential(start_grid_x, start_grid_y, goal_grid_x, goal_grid_y)
        
        if path:
            return path
        
        # If potential field path extraction fails, fall back to wavefront path
        return wavefront_path
        
    def extract_path_from_potential(self, start_grid_x, start_grid_y, goal_grid_x, goal_grid_y):
        """
        Enhanced path extraction from potential field with better handling of connected obstacles.
        Uses flow field and passage detection for more robust navigation.
        """
        if self.potential_field is None:
            return None
            
        path = []
        current_x, current_y = start_grid_x, start_grid_y
        
        # Convert start point to world coordinates and add to path
        start_world_x, start_world_y = self.grid_to_world(start_grid_x, start_grid_y)
        path.append((start_world_x, start_world_y))
        
        # Maximum steps to prevent infinite loops
        max_steps = min(1000, self.grid_width * self.grid_height // 10)
        
        # Track visited points to detect loops
        visited = set([(current_x, current_y)])
        
        # Track movement history to detect oscillations
        position_history = [(current_x, current_y)]
        
        # Local minima detection and escape
        stuck_counter = 0
        local_minima_counter = 0
        backtrack_counter = 0
        random_walk_counter = 0
        
        # Track when we're trying to navigate through a passage
        in_passage_mode = False
        passage_target = None
        
        # Track when we're using flow field as escape
        using_flow_field = False
        flow_field_steps = 0
        max_flow_field_steps = 20  # Maximum steps to follow flow field before reverting to APF
        
        # For circumventing connected obstacles
        circumventing_obstacles = False
        circumvention_direction = 1  # 1 for clockwise, -1 for counter-clockwise
        wall_following_steps = 0
        max_wall_following_steps = 50  # Maximum steps for wall following
        
        for step in range(max_steps):
            # Check if reached goal vicinity
            if math.sqrt((current_x - goal_grid_x)**2 + (current_y - goal_grid_y)**2) <= 2:
                # Add goal to path and break
                goal_world_x, goal_world_y = self.grid_to_world(goal_grid_x, goal_grid_y)
                if path[-1] != (goal_world_x, goal_world_y):
                    path.append((goal_world_x, goal_world_y))
                break
            
            # If we've been using flow field escape for too long, revert to APF
            if using_flow_field:
                flow_field_steps += 1
                if flow_field_steps >= max_flow_field_steps:
                    using_flow_field = False
                    flow_field_steps = 0
            
            # If we've been following walls for too long, try another approach
            if circumventing_obstacles:
                wall_following_steps += 1
                if wall_following_steps >= max_wall_following_steps:
                    circumventing_obstacles = False
                    wall_following_steps = 0
                    # Switch to flow field escape after wall following
                    using_flow_field = True
                    flow_field_steps = 0
            
            # Determine next position based on current mode
            if circumventing_obstacles:
                # Use wall following to get around connected obstacles
                next_x, next_y = self._follow_wall(current_x, current_y, visited, circumvention_direction)
            elif using_flow_field and self.flow_field is not None:
                # Use flow field for escape
                next_x, next_y = self._follow_flow_field(current_x, current_y, visited)
                
                # If flow field escape didn't work, try circumventing obstacles
                if next_x == current_x and next_y == current_y:
                    using_flow_field = False
                    circumventing_obstacles = True
                    wall_following_steps = 0
                    next_x, next_y = self._follow_wall(current_x, current_y, visited, circumvention_direction)
            else:
                # Find neighbor with lowest potential (standard APF)
                min_potential = float('inf')
                next_x, next_y = current_x, current_y
                
                for dx, dy in self.directions:
                    new_x = current_x + dx
                    new_y = current_y + dy
                    
                    if (0 <= new_x < self.grid_width and 
                        0 <= new_y < self.grid_height and
                        self.potential_field[new_y, new_x] < min_potential and
                        self.potential_field[new_y, new_x] != float('inf') and
                        (new_x, new_y) not in visited):
                        
                        min_potential = self.potential_field[new_y, new_x]
                        next_x, next_y = new_x, new_y
            
            # Check if stuck (no movement)
            if next_x == current_x and next_y == current_y:
                stuck_counter += 1
                local_minima_counter += 1
                
                # Check for oscillation
                if len(position_history) >= 4:
                    recent_positions = position_history[-4:]
                    unique_positions = set(recent_positions)
                    if len(unique_positions) <= 2:  # Oscillating between 1-2 positions
                        # Try random movement to escape oscillation
                        random_result = self._try_random_move(current_x, current_y, visited, path)
                        if random_result:
                            continue  # Successfully moved randomly
                        
                        # If random movement failed, start circumventing obstacles
                        circumventing_obstacles = True
                        wall_following_steps = 0
                        
                        # Alternate direction each time
                        circumvention_direction *= -1
                        
                        next_x, next_y = self._follow_wall(current_x, current_y, visited, circumvention_direction)
                        if next_x != current_x or next_y != current_y:
                            # Successfully started wall following
                            pass
                        else:
                            # If wall following failed, try the other direction
                            circumvention_direction *= -1
                            next_x, next_y = self._follow_wall(current_x, current_y, visited, circumvention_direction)
                
                # If we're possibly in a local minimum, try flow field escape
                if local_minima_counter >= self.local_minima_detection_threshold and not using_flow_field and not circumventing_obstacles:
                    # Switch to flow field mode
                    using_flow_field = True
                    flow_field_steps = 0
                    
                    # Try flow field for one step
                    if self.flow_field is not None:
                        next_x, next_y = self._follow_flow_field(current_x, current_y, visited)
                    
                    # If still stuck, try to find a passage
                    if next_x == current_x and next_y == current_y:
                        # Try to find a passage that can help escape the local minimum
                        passage_target = self._find_nearest_passage(current_x, current_y, goal_grid_x, goal_grid_y)
                        
                        if passage_target:
                            # Found a promising passage, target it directly
                            in_passage_mode = True
                            next_x, next_y = passage_target
                            
                            # Calculate direct path to passage point
                            path_to_passage = self._direct_path_to_target(
                                current_x, current_y, 
                                next_x, next_y, 
                                visited
                            )
                            
                            if path_to_passage:
                                # Add path to passage to our main path
                                for point in path_to_passage[1:]:  # Skip the first point
                                    world_x, world_y = self.grid_to_world(point[0], point[1])
                                    path.append((world_x, world_y))
                                    visited.add((point[0], point[1]))
                                    position_history.append((point[0], point[1]))
                                
                                # Update current position to last point in path
                                current_x, current_y = path_to_passage[-1]
                                stuck_counter = 0
                                local_minima_counter = 0
                                continue
                
                # If still stuck after trying everything, try wavefront
                if (stuck_counter >= 5 and not using_flow_field and not circumventing_obstacles 
                    and not in_passage_mode):
                    # Attempt to follow wavefront
                    if self.wavefront_grid is not None and self.wavefront_grid[current_y, current_x] != -1:
                        # Find neighbor with lowest wavefront value in larger neighborhood
                        min_wave = float('inf')
                        
                        # Look in a larger neighborhood (3-cell radius)
                        for dy in range(-3, 4):
                            for dx in range(-3, 4):
                                new_x = current_x + dx
                                new_y = current_y + dy
                                
                                if (0 <= new_x < self.grid_width and 
                                    0 <= new_y < self.grid_height and
                                    self.wavefront_grid[new_y, new_x] != -1 and
                                    self.occupancy_grid[new_y, new_x] == 0 and
                                    (new_x, new_y) not in visited):
                                    
                                    if self.wavefront_grid[new_y, new_x] < min_wave:
                                        min_wave = self.wavefront_grid[new_y, new_x]
                                        next_x, next_y = new_x, new_y
                        
                        if next_x != current_x or next_y != current_y:
                            # Successfully found a point from wavefront
                            world_x, world_y = self.grid_to_world(next_x, next_y)
                            path.append((world_x, world_y))
                            current_x, current_y = next_x, next_y
                            visited.add((current_x, current_y))
                            position_history.append((current_x, current_y))
                            stuck_counter = 0
                            continue
                    
                    # If we're still stuck, try a new fallback - direct path to goal with obstacle check
                    direct_path = self._direct_path_with_obstacle_avoidance(
                        current_x, current_y, goal_grid_x, goal_grid_y)
                        
                    if direct_path:
                        # Add this path to our main path
                        for point in direct_path[1:]:  # Skip the first point
                            world_x, world_y = self.grid_to_world(point[0], point[1])
                            path.append((world_x, world_y))
                            visited.add((point[0], point[1]))
                            position_history.append((point[0], point[1]))
                        
                        # Update current position to last point in path
                        current_x, current_y = direct_path[-1]
                        stuck_counter = 0
                        continue
                
                # If still stuck, try other strategies
                if next_x == current_x and next_y == current_y:
                    # Try backtracking
                    if len(path) > 2:
                        # Remove the current position and go back
                        path.pop()
                        world_x, world_y = path[-1]
                        grid_x, grid_y = self.world_to_grid((world_x, world_y))
                        current_x, current_y = grid_x, grid_y
                        position_history.append((current_x, current_y))
                        continue
                    
                    # If backtracking fails, try random walk
                    success = self._try_random_walk(current_x, current_y, visited, path, goal_grid_x, goal_grid_y)
                    if success:
                        current_x, current_y = success
                        visited.add((current_x, current_y))
                        position_history.append((current_x, current_y))
                        world_x, world_y = self.grid_to_world(current_x, current_y)
                        path.append((world_x, world_y))
                        stuck_counter = 0
                        continue
                    
                    # If all else fails, fall back to wavefront path
                    world_x, world_y = self.grid_to_world(current_x, current_y)
                    goal_world_x, goal_world_y = self.grid_to_world(goal_grid_x, goal_grid_y)
                    wavefront_path = self.wavefront_expansion(
                        (world_x, world_y),
                        (goal_world_x, goal_world_y)
                    )
                    if wavefront_path:
                        path.extend(wavefront_path[1:])  # Skip first point as it's already in path
                    break
            
            # Move to next position
            current_x, current_y = next_x, next_y
            visited.add((current_x, current_y))
            position_history.append((current_x, current_y))
            
            # Reset counters since we moved
            if not using_flow_field and not circumventing_obstacles:
                stuck_counter = 0
                backtrack_counter = 0
            
            # Add to path
            world_x, world_y = self.grid_to_world(current_x, current_y)
            path.append((world_x, world_y))
        
        # Add goal to path if not already close
        goal_world_x, goal_world_y = self.grid_to_world(goal_grid_x, goal_grid_y)
        if path:
            last_x, last_y = path[-1]
            if math.sqrt((last_x - goal_world_x)**2 + (last_y - goal_world_y)**2) > 0.2:
                path.append((goal_world_x, goal_world_y))
        
        return self.smooth_path(path)
    
    def _follow_flow_field(self, current_x, current_y, visited):
        """
        Follow the flow field to escape from local minima.
        Returns the next grid coordinates.
        """
        if self.flow_field is None:
            return current_x, current_y
        
        # Get flow direction at current point
        if (0 <= current_x < self.grid_width and 
            0 <= current_y < self.grid_height and 
            self.wavefront_grid is not None and 
            self.wavefront_grid[current_y, current_x] != -1):
            
            flow_dir = self.flow_field[current_y, current_x]
            
            # If flow direction is zero, use wavefront gradient
            if np.linalg.norm(flow_dir) < 0.1:
                gradient = self._calculate_wavefront_gradient(current_x, current_y)
                if np.linalg.norm(gradient) > 0.1:
                    flow_dir = -gradient  # Negative because we're going toward lower values
            
            # If still no clear direction, find the neighbor with lowest wavefront value
            if np.linalg.norm(flow_dir) < 0.1:
                min_val = float('inf')
                next_x, next_y = current_x, current_y
                
                for dx, dy in self.directions:
                    new_x = current_x + dx
                    new_y = current_y + dy
                    
                    if (0 <= new_x < self.grid_width and 
                        0 <= new_y < self.grid_height and
                        self.wavefront_grid[new_y, new_x] != -1 and
                        self.occupancy_grid[new_y, new_x] == 0 and
                        (new_x, new_y) not in visited):
                        
                        if self.wavefront_grid[new_y, new_x] < min_val:
                            min_val = self.wavefront_grid[new_y, new_x]
                            next_x, next_y = new_x, new_y
                
                if next_x != current_x or next_y != current_y:
                    return next_x, next_y
            
            # Use flow direction to find the best next step
            if np.linalg.norm(flow_dir) > 0.1:
                flow_dir = flow_dir / np.linalg.norm(flow_dir)
                
                # Find the neighbor that best aligns with flow direction
                best_match = None
                best_alignment = -1.0
                
                for dx, dy in self.directions:
                    new_x = current_x + dx
                    new_y = current_y + dy
                    
                    if (0 <= new_x < self.grid_width and 
                        0 <= new_y < self.grid_height and
                        self.occupancy_grid[new_y, new_x] == 0 and
                        (new_x, new_y) not in visited):
                        
                        # Calculate alignment with flow direction
                        direction = np.array([float(dx), float(dy)])
                        direction = direction / np.linalg.norm(direction)
                        alignment = np.dot(flow_dir, direction)
                        
                        # Also consider wavefront value if available
                        if self.wavefront_grid is not None and self.wavefront_grid[new_y, new_x] != -1:
                            # Better if wavefront value is lower
                            wave_factor = 1.0
                            if self.wavefront_grid[new_y, new_x] < self.wavefront_grid[current_y, current_x]:
                                wave_factor = 1.2  # Small bonus for moving toward goal
                            alignment *= wave_factor
                        
                        if alignment > best_alignment:
                            best_alignment = alignment
                            best_match = (new_x, new_y)
                
                if best_match is not None:
                    return best_match
        
        return current_x, current_y
        
    def _follow_wall(self, current_x, current_y, visited, direction=1):
        """
        Follow a wall to circumvent connected obstacles.
        Modified to strictly respect the occupancy grid.
        
        Parameters:
        -----------
        current_x, current_y : int
            Current grid position
        visited : set
            Set of already visited grid positions
        direction : int
            1 for clockwise wall following, -1 for counter-clockwise
            
        Returns:
        --------
        (next_x, next_y) : tuple
            Next grid position along the wall
        """
        # First, find the nearest obstacle in the grid
        found_obstacle = False
        nearest_obs_x, nearest_obs_y = None, None
        min_dist = float('inf')
        
        # Search in a small neighborhood
        search_radius = 3
        
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                    
                nx, ny = current_x + dx, current_y + dy
                
                if (0 <= nx < self.grid_width and 
                    0 <= ny < self.grid_height and
                    self.occupancy_grid[ny, nx] == 1):
                    
                    dist = dx*dx + dy*dy  # Squared distance is fine for comparison
                    if dist < min_dist:
                        min_dist = dist
                        nearest_obs_x, nearest_obs_y = nx, ny
                        found_obstacle = True
        
        if not found_obstacle:
            # No obstacle found nearby, try a larger search radius
            search_radius = 5
            
            for dy in range(-search_radius, search_radius + 1):
                for dx in range(-search_radius, search_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                        
                    nx, ny = current_x + dx, current_y + dy
                    
                    if (0 <= nx < self.grid_width and 
                        0 <= ny < self.grid_height and
                        self.occupancy_grid[ny, nx] == 1):
                        
                        dist = dx*dx + dy*dy
                        if dist < min_dist:
                            min_dist = dist
                            nearest_obs_x, nearest_obs_y = nx, ny
                            found_obstacle = True
        
        if not found_obstacle:
            # Still no obstacle found, check all 8 neighboring cells for any valid move
            for dx, dy in [(1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]:
                nx, ny = current_x + dx, current_y + dy
                
                if (0 <= nx < self.grid_width and 
                    0 <= ny < self.grid_height and
                    self.occupancy_grid[ny, nx] == 0 and
                    (nx, ny) not in visited):
                    return nx, ny
                    
            # If absolutely no valid move, return current position
            return current_x, current_y
        
        # Calculate direction from current position to obstacle
        to_obs_x = nearest_obs_x - current_x
        to_obs_y = nearest_obs_y - current_y
        
        # Get the perpendicular direction (rotate 90 degrees)
        # For clockwise: (x, y) -> (y, -x)
        # For counter-clockwise: (x, y) -> (-y, x)
        if direction == 1:  # Clockwise
            perp_x = to_obs_y
            perp_y = -to_obs_x
        else:  # Counter-clockwise
            perp_x = -to_obs_y
            perp_y = to_obs_x
        
        # Normalize to unit steps
        if abs(perp_x) > 1 or abs(perp_y) > 1:
            # Get the sign
            perp_x = 1 if perp_x > 0 else (-1 if perp_x < 0 else 0)
            perp_y = 1 if perp_y > 0 else (-1 if perp_y < 0 else 0)
        
        # Calculate next position
        next_x = current_x + perp_x
        next_y = current_y + perp_y
        
        # Check if valid and not visited and not in obstacle
        if (0 <= next_x < self.grid_width and 
            0 <= next_y < self.grid_height and
            self.occupancy_grid[next_y, next_x] == 0 and
            (next_x, next_y) not in visited):
            return next_x, next_y
        
        # If invalid or visited, try all 8 neighboring directions
        for dx, dy in [(1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (1,-1), (-1,-1)]:
            nx, ny = current_x + dx, current_y + dy
            
            if (0 <= nx < self.grid_width and 
                0 <= ny < self.grid_height and
                self.occupancy_grid[ny, nx] == 0 and
                (nx, ny) not in visited):
                return nx, ny
        
        # If all else fails, return current position
        return current_x, current_y

    def _find_nearest_passage(self, current_x, current_y, goal_x, goal_y):
        """
        Find the nearest detected passage that might help escape local minima.
        Returns (x, y) grid coordinates of the passage point, or None if no suitable passage found.
        """
        if not self.local_minima_regions:
            return None
            
        # Get world coordinates for current position
        current_world_x, current_world_y = self.grid_to_world(current_x, current_y)
        goal_world_x, goal_world_y = self.grid_to_world(goal_x, goal_y)
        
        # Calculate vector from current to goal
        goal_vector = np.array([goal_world_x - current_world_x, goal_world_y - current_world_y])
        goal_dist = np.linalg.norm(goal_vector)
        if goal_dist > 0:
            goal_vector = goal_vector / goal_dist  # Normalize
        
        # Find passages that are:
        # 1. Within reasonable distance
        # 2. Have good clearance (high fitness)
        # 3. Are somewhat in the direction of the goal
        max_passage_dist = self.rho_0 * 3.0  # Maximum distance to consider a passage
        best_passage = None
        best_score = -float('inf')
        
        for passage in self.local_minima_regions:
            pos = passage['position']
            passage_x, passage_y = pos
            
            # Calculate distance to passage
            passage_dist = math.sqrt((passage_x - current_world_x)**2 + (passage_y - current_world_y)**2)
            
            # Skip if too far
            if passage_dist > max_passage_dist:
                continue
                
            # Calculate direction vector to passage
            passage_vector = np.array([passage_x - current_world_x, passage_y - current_world_y])
            if passage_dist > 0:
                passage_vector = passage_vector / passage_dist  # Normalize
                
            # Calculate alignment with goal direction (dot product)
            direction_alignment = np.dot(goal_vector, passage_vector)
            
            # Calculate score based on distance, fitness, and alignment
            # Higher fitness, lower distance, and better alignment = higher score
            score = passage['fitness'] * 2.0  # Fitness is most important
            score -= passage_dist / max_passage_dist  # Closer is better
            score += max(0, direction_alignment)  # Alignment with goal direction
            
            if score > best_score:
                best_score = score
                best_passage = passage
        
        if best_passage and best_score > 0.5:  # Only use passage if score is good enough
            # Convert back to grid coordinates
            grid_x, grid_y = self.world_to_grid(best_passage['position'])
            return grid_x, grid_y
            
        return None
    
    def _try_random_move(self, x, y, visited, path):
        """Try a random move to escape local minimum."""
        valid_moves = []
        
        for dx, dy in self.directions:
            new_x = x + dx
            new_y = y + dy
            
            if (0 <= new_x < self.grid_width and 
                0 <= new_y < self.grid_height and
                self.occupancy_grid[new_y, new_x] == 0 and
                (new_x, new_y) not in visited):
                
                valid_moves.append((new_x, new_y))
        
        if valid_moves:
            # Choose a random valid move
            next_x, next_y = random.choice(valid_moves)
            visited.add((next_x, next_y))
            
            # Add to path
            world_x, world_y = self.grid_to_world(next_x, next_y)
            path.append((world_x, world_y))
            
            return True
        
        return False
    
    def _direct_path_to_target(self, start_x, start_y, target_x, target_y, visited):
        """
        Calculate a direct path from start to target using a modified A* algorithm.
        Returns a list of grid coordinates [(x1,y1), (x2,y2), ...] or None if no path found.
        """
        if (start_x == target_x and start_y == target_y):
            return [(start_x, start_y)]
            
        # Use A* to find path
        # Initialize data structures
        open_set = PriorityQueue()
        open_set.put((0, start_x, start_y))  # (priority, x, y)
        came_from = {}
        g_score = {(start_x, start_y): 0}  # Cost from start to current node
        f_score = {(start_x, start_y): self._heuristic(start_x, start_y, target_x, target_y)}  # Estimated total cost
        
        # Local visited set just for this path finding
        path_visited = set([(start_x, start_y)])
        path_visited.update(visited)  # Include global visited points to avoid loops
        
        # Store all points explored for visualization
        all_explored = [(start_x, start_y)]
        
        while not open_set.empty():
            _, current_x, current_y = open_set.get()
            current = (current_x, current_y)
            
            # Check if we've reached the target
            if current_x == target_x and current_y == target_y:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()  # Reverse to get start-to-target order
                return path
            
            # Consider neighbors
            for dx, dy in self.directions:
                neighbor_x, neighbor_y = current_x + dx, current_y + dy
                neighbor = (neighbor_x, neighbor_y)
                
                # Skip invalid or visited neighbors
                if (neighbor_x < 0 or neighbor_x >= self.grid_width or
                    neighbor_y < 0 or neighbor_y >= self.grid_height or
                    self.occupancy_grid[neighbor_y, neighbor_x] == 1 or
                    neighbor in path_visited):
                    continue
                
                # Calculate movement cost (including diagonal penalty)
                if abs(dx) == 1 and abs(dy) == 1:
                    # Diagonal move costs sqrt(2)
                    tentative_g_score = g_score[current] + 1.414
                else:
                    # Cardinal move costs 1
                    tentative_g_score = g_score[current] + 1.0
                
                # If using potential field to weight the path
                if self.potential_field is not None:
                    # Add cost based on potential (avoid high potential areas)
                    potential_cost = self.potential_field[neighbor_y, neighbor_x] * 0.01
                    if potential_cost != float('inf'):
                        tentative_g_score += potential_cost
                
                # If this is a better path to neighbor
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path info
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor_x, neighbor_y, target_x, target_y)
                    
                    # Add to open set if not already there
                    if neighbor not in path_visited:
                        path_visited.add(neighbor)
                        all_explored.append(neighbor)
                        open_set.put((f_score[neighbor], neighbor_x, neighbor_y))
        
        # If we get here, no path was found
        return None
        
    def _direct_path_with_obstacle_avoidance(self, start_x, start_y, goal_x, goal_y):
        """
        Create a direct path to goal with obstacle avoidance.
        Uses a simplified A* algorithm to find a path around obstacles.
        
        Parameters:
        -----------
        start_x, start_y : int
            Starting grid position
        goal_x, goal_y : int
            Goal grid position
            
        Returns:
        --------
        list or None
            List of grid positions forming a path, or None if no path found
        """
        # Initialize open and closed sets
        open_set = PriorityQueue()
        open_set.put((0, start_x, start_y))
        came_from = {}
        g_score = {(start_x, start_y): 0}
        f_score = {(start_x, start_y): self._heuristic(start_x, start_y, goal_x, goal_y)}
        
        # Track visited cells
        visited = set([(start_x, start_y)])
        
        # Maximum iterations to prevent infinite loops
        max_iterations = min(10000, self.grid_width * self.grid_height)
        
        for _ in range(max_iterations):
            if open_set.empty():
                return None
                
            # Get the node with the lowest f_score
            _, current_x, current_y = open_set.get()
            current = (current_x, current_y)
            
            # If we reached the goal
            if current_x == goal_x and current_y == goal_y:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            # Consider all neighbors
            for dx, dy in self.directions:
                neighbor_x = current_x + dx
                neighbor_y = current_y + dy
                neighbor = (neighbor_x, neighbor_y)
                
                # Skip invalid or already visited cells
                if (neighbor_x < 0 or neighbor_x >= self.grid_width or
                    neighbor_y < 0 or neighbor_y >= self.grid_height or
                    self.occupancy_grid[neighbor_y, neighbor_x] == 1 or
                    neighbor in visited):
                    continue
                
                # Calculate tentative g_score
                if abs(dx) == 1 and abs(dy) == 1:
                    # Diagonal move costs sqrt(2)
                    movement_cost = 1.414
                else:
                    # Cardinal move costs 1
                    movement_cost = 1.0
                    
                tentative_g_score = g_score[current] + movement_cost
                
                # If this is a better path to the neighbor
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path info
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor_x, neighbor_y, goal_x, goal_y)
                    
                    # Add to open set and visited set
                    if neighbor not in visited:
                        visited.add(neighbor)
                        open_set.put((f_score[neighbor], neighbor_x, neighbor_y))
        
        # If we've exhausted all options, return None
        return None
        
    def _try_random_walk(self, x, y, visited, path, goal_x, goal_y, max_steps=10):
        """Perform a random walk trying to get closer to the goal."""
        current_x, current_y = x, y
        best_dist = math.sqrt((current_x - goal_x)**2 + (current_y - goal_y)**2)
        best_point = None
        
        local_visited = set()
        
        for _ in range(max_steps):
            valid_moves = []
            
            for dx, dy in self.directions:
                new_x = current_x + dx
                new_y = current_y + dy
                
                if (0 <= new_x < self.grid_width and 
                    0 <= new_y < self.grid_height and
                    self.occupancy_grid[new_y, new_x] == 0 and
                    (new_x, new_y) not in visited and
                    (new_x, new_y) not in local_visited):
                    
                    valid_moves.append((new_x, new_y))
            
            if not valid_moves:
                break
                
            # Choose a random valid move
            next_x, next_y = random.choice(valid_moves)
            local_visited.add((next_x, next_y))
            
            # Check if this improved distance to goal
            dist = math.sqrt((next_x - goal_x)**2 + (next_y - goal_y)**2)
            if dist < best_dist:
                best_dist = dist
                best_point = (next_x, next_y)
            
            current_x, current_y = next_x, next_y
        
        return best_point
        
    def plan_path(self, start_pos, goal_pos, obstacles=None, obstacle_sizes=None, object_positions=None, sonar_data=None, robot_pose=None):
        """
        Generate path using combined wavefront and potential field approach.
        Automatically uses flow fields for escape and clearance-aware path planning.
        
        Parameters:
        -----------
        start_pos : np.array
            Start position [x, y, z]
        goal_pos : np.array
            Goal position [x, y, z]
        obstacles : list of np.array, optional
            List of obstacle positions [[x1, y1, z1], [x2, y2, z2], ...]
        obstacle_sizes : list of float, optional
            List of obstacle widths in meters
        object_positions : list, optional
            List of object positions (not used in APF)
        sonar_data : dict, optional
            Sonar data for additional obstacle detection (ignored)
        robot_pose : Pose, optional
            Current robot pose for visualization
            
        Returns:
        --------
        list of np.array
            Path as list of points [[x1, y1, z1], [x2, y2, z2], ...]
        """
        overall_start_time = time.time()
        
        if obstacles is None:
            obstacles = []
            
        if obstacle_sizes is None:
            obstacle_sizes = [0.3] * len(obstacles)
            
        if object_positions is None:
            object_positions = []
            
        sonar_data = {}
            
        self.oscillation_count = 0
        self.progress_stagnation_count = 0
        self.escape_attempt_count = 0
        self.random_walk_active = False
        self.prev_position = None
        self.prev_force = None
        
        if self.successful_paths_count > 0 and self.last_successful_params:
            self.k_att = self.last_successful_params['k_att']
            self.k_rep = self.last_successful_params['k_rep']
            self.rho_0 = self.last_successful_params['rho_0']
            
        if len(obstacles) > len(obstacle_sizes):
            obstacle_sizes.extend([0.3] * (len(obstacles) - len(obstacle_sizes)))
        elif len(obstacles) < len(obstacle_sizes):
            obstacle_sizes = obstacle_sizes[:len(obstacles)]
            
        obstacle_data = []
        for i, obs in enumerate(obstacles):
            size = obstacle_sizes[i]
            obstacle_data.append({
                "position": obs[:2],
                "size": size
            })
            
        self.update_grid_from_obstacles(
            start_pos[:2],
            object_positions,
            obstacle_data,
            None
        )
        
        # Time the wavefront expansion separately
        wavefront_start_time = time.time()
        wavefront_path = self.wavefront_expansion(
            start_pos[:2],
            goal_pos[:2]
        )
        wavefront_end_time = time.time()
        self.last_wavefront_time = wavefront_end_time - wavefront_start_time
        
        # Time the path computation (after wavefront)
        path_computation_start_time = time.time()
        
        if self.wavefront_grid_publisher is not None:
            if self.wavefront_grid is None:
                start_grid_x, start_grid_y = self.world_to_grid(start_pos[:2])
                goal_grid_x, goal_grid_y = self.world_to_grid(goal_pos[:2])
                
                self.wavefront_grid = np.full((self.grid_height, self.grid_width), -1, dtype=float)
                
                if (0 <= start_grid_x < self.grid_width and 
                    0 <= start_grid_y < self.grid_height):
                    self.wavefront_grid[start_grid_y, start_grid_x] = 0
                
                if (0 <= goal_grid_x < self.grid_width and 
                    0 <= goal_grid_y < self.grid_height):
                    self.wavefront_grid[goal_grid_y, goal_grid_x] = 1000
            
            self.force_visualization = True
            
            self.publish_wavefront_grid_visualization(
                self.wavefront_grid_publisher,
                frame_id="map",
                robot_pose=robot_pose
            )
        
        if wavefront_path:
            path_3d = [np.append(point, [0]) for point in wavefront_path]
            
            self.last_successful_params = {
                'k_att': self.k_att,
                'k_rep': self.k_rep,
                'rho_0': self.rho_0
            }
            self.successful_paths_count += 1
            
            path_3d = self.smooth_path(path_3d, obstacles, obstacle_sizes)
            
            valid_path = []
            for point in path_3d:
                grid_x, grid_y = self.world_to_grid(point[:2])
                if (0 <= grid_x < self.grid_width and 
                    0 <= grid_y < self.grid_height and
                    self.occupancy_grid[grid_y, grid_x] == 0):
                    valid_path.append(point)
            
            path_computation_end_time = time.time()
            self.last_path_computation_time = path_computation_end_time - path_computation_start_time
            
            overall_end_time = time.time()
            self.last_planning_time = overall_end_time - overall_start_time
            
            return valid_path
            
        # APF fallback path (rest of the function continues with APF...)
        path = [start_pos]
        current_pos = start_pos.copy()
        
        current_pos = np.array(current_pos[:2])
        goal_pos = np.array(goal_pos[:2])
        
        obstacle_positions = []
        for obs in obstacles:
            if isinstance(obs, np.ndarray):
                obstacle_positions.append(obs[:2])
            elif isinstance(obs, list):
                obstacle_positions.append(np.array(obs[:2]))
            else:
                obstacle_positions.append(np.array([obs[0], obs[1]]))
        
        max_steps = 1000
        min_dist_to_goal = np.linalg.norm(goal_pos - current_pos)
        steps_since_improvement = 0
        step_size = self.resolution
        
        prev_step = np.zeros(2)
        
        k_att_current = self.k_att
        k_rep_current = self.k_rep
        
        has_wavefront = self.wavefront_grid is not None
        
        for step_count in range(max_steps):
            grid_x, grid_y = self.world_to_grid(current_pos)
            if not (0 <= grid_x < self.grid_width and 
                    0 <= grid_y < self.grid_height) or self.occupancy_grid[grid_y, grid_x] == 1:
                valid_pos = self.find_safe_point_nearby(grid_x, grid_y)
                if valid_pos:
                    grid_x, grid_y = valid_pos
                    current_pos = np.array(self.grid_to_world(grid_x, grid_y))
                    path[-1] = np.append(current_pos, [0])
                else:
                    break
            
            dist_to_goal = np.linalg.norm(goal_pos - current_pos)
            
            if dist_to_goal < min_dist_to_goal:
                min_dist_to_goal = dist_to_goal
                steps_since_improvement = 0
                self.progress_stagnation_count = 0
            else:
                steps_since_improvement += 1
                self.progress_stagnation_count += 1
                
            if steps_since_improvement > 100:
                break
                
            if dist_to_goal < self.resolution:
                goal_grid_x, goal_grid_y = self.world_to_grid(goal_pos)
                if (0 <= goal_grid_x < self.grid_width and 
                    0 <= goal_grid_y < self.grid_height and
                    self.occupancy_grid[goal_grid_y, goal_grid_x] == 0):
                    path.append(np.append(goal_pos, [0]))
                
                self.last_successful_params = {
                    'k_att': k_att_current,
                    'k_rep': k_rep_current,
                    'rho_0': self.rho_0
                }
                self.successful_paths_count += 1
                
                path_computation_end_time = time.time()
                self.last_path_computation_time = path_computation_end_time - path_computation_start_time
                
                overall_end_time = time.time()
                self.last_planning_time = overall_end_time - overall_start_time
                
                break
                
            if self.adaptive_parameters and steps_since_improvement > self.local_minima_detection_threshold:
                adaptation_phase = (steps_since_improvement // 10) % 3
                
                if adaptation_phase == 0:
                    k_att_current = min(self.adaptive_k_att_range[1], 
                                    self.k_att * (1.0 + 0.2 * (steps_since_improvement // 10)))
                elif adaptation_phase == 1:
                    k_rep_current = max(self.adaptive_k_rep_range[0],
                                    self.k_rep * (1.0 - 0.2 * (steps_since_improvement // 10)))
                else:
                    k_att_current = self.k_att * 1.5
                    k_rep_current = self.k_rep * 0.7
            else:
                k_att_current = self.k_att
                k_rep_current = self.k_rep
                
            if steps_since_improvement > self.local_minima_detection_threshold:
                next_pos = self.handle_local_minima(
                    current_pos, goal_pos, step_size, 
                    obstacle_positions, obstacle_sizes
                )
                
                next_grid_x, next_grid_y = self.world_to_grid(next_pos)
                if not (0 <= next_grid_x < self.grid_width and 
                        0 <= next_grid_y < self.grid_height) or self.occupancy_grid[next_grid_y, next_grid_x] == 1:
                    valid_pos = self.find_safe_point_nearby(next_grid_x, next_grid_y)
                    if valid_pos:
                        next_pos = np.array(self.grid_to_world(*valid_pos))
                    else:
                        next_pos = current_pos
                
                path.append(np.append(next_pos, [0]))
                
                current_pos = next_pos
                continue
                
            f_att = self.compute_attractive_force(current_pos, goal_pos)
            f_rep = self.compute_repulsive_force(current_pos, obstacle_positions, obstacle_sizes)
            
            if self.adaptive_parameters and steps_since_improvement > self.local_minima_detection_threshold:
                f_att = f_att * (k_att_current / self.k_att)
                f_rep = f_rep * (k_rep_current / self.k_rep)
            
            f_total = f_att + f_rep
            
            if has_wavefront:
                f_total = self.apply_wavefront_guidance(current_pos, goal_pos, f_total)
            
            if np.linalg.norm(prev_step) > 0:
                momentum_weight = self.momentum_factor * min(1.0, steps_since_improvement / 20.0)
                momentum = prev_step * momentum_weight
                f_total += momentum
            
            f_magnitude = np.linalg.norm(f_total)
            if f_magnitude > 0:
                step = step_size * f_total / f_magnitude
            else:
                step = np.zeros(2)
                
            prev_step = step.copy()
                
            next_pos = current_pos + step
            
            next_grid_x, next_grid_y = self.world_to_grid(next_pos)
            if not (0 <= next_grid_x < self.grid_width and 
                    0 <= next_grid_y < self.grid_height) or self.occupancy_grid[next_grid_y, next_grid_x] == 1:
                found_valid = False
                for angle in [30, 60, -30, -60, 90, -90]:
                    rad_angle = math.radians(angle)
                    rot_x = step[0] * math.cos(rad_angle) - step[1] * math.sin(rad_angle)
                    rot_y = step[0] * math.sin(rad_angle) + step[1] * math.cos(rad_angle)
                    rot_step = np.array([rot_x, rot_y])
                    test_pos = current_pos + rot_step
                    
                    test_grid_x, test_grid_y = self.world_to_grid(test_pos)
                    if (0 <= test_grid_x < self.grid_width and 
                        0 <= test_grid_y < self.grid_height and
                        self.occupancy_grid[test_grid_y, test_grid_x] == 0):
                        next_pos = test_pos
                        found_valid = True
                        break
                        
                if not found_valid:
                    for scale in [0.5, 0.25, 0.1]:
                        test_pos = current_pos + step * scale
                        test_grid_x, test_grid_y = self.world_to_grid(test_pos)
                        if (0 <= test_grid_x < self.grid_width and 
                            0 <= test_grid_y < self.grid_height and
                            self.occupancy_grid[test_grid_y, test_grid_x] == 0):
                            next_pos = test_pos
                            found_valid = True
                            break
                            
                if not found_valid:
                    next_pos = current_pos
                    self.oscillation_count += 1
            
            current_pos = next_pos
            
            path.append(np.append(current_pos, [0]))
        
        valid_path = []
        for point in path:
            grid_x, grid_y = self.world_to_grid(point[:2])
            if (0 <= grid_x < self.grid_width and 
                0 <= grid_y < self.grid_height and
                self.occupancy_grid[grid_y, grid_x] == 0):
                valid_path.append(point)
        
        path_computation_end_time = time.time()
        self.last_path_computation_time = path_computation_end_time - path_computation_start_time
        
        overall_end_time = time.time()
        self.last_planning_time = overall_end_time - overall_start_time
            
        return self.smooth_path(valid_path, obstacles, obstacle_sizes)

    def smooth_path(self, path, obstacles=None, obstacle_sizes=None):
        """Apply path smoothing to remove jitters and consider robot size."""
        if len(path) < 3:
            return path
            
        smoothed = [path[0]]
        window_size = 3
        
        # Robot dimensions influence the smoothing process
        # Larger robots need smoother paths with wider turns
        smoothing_factor = max(1.0, (self.robot_length + self.robot_width) / 2.0)
        
        for i in range(1, len(path)-1):
            # Moving average smoothing with weighting based on robot size
            window = path[max(0,i-window_size):min(len(path),i+window_size+1)]
            
            # Apply smoothing - larger robots get more smoothing
            smoothed_point = np.mean(window, axis=0)
            
            # For larger robots, bias towards wider turns by mixing with original point
            if smoothing_factor > 1.5:
                # Blend original point with smoothed point based on robot size
                blend_factor = min(0.7, (smoothing_factor - 1.0) / 2.0)  # 0 to 0.7
                smoothed_point = (1 - blend_factor) * path[i] + blend_factor * smoothed_point
            
            # Check if smoothed point would cause collision
            if obstacles is not None and self.check_collision(smoothed_point[:2], obstacles, obstacle_sizes):
                # If collision, use original point
                smoothed.append(path[i])
            else:
                smoothed.append(smoothed_point)
            
        smoothed.append(path[-1])
        
        # Add a second pass for larger robots to ensure very smooth paths
        if smoothing_factor > 1.5:
            smoothed = self.smooth_path_second_pass(smoothed, obstacles, obstacle_sizes)
            
        return smoothed
    
    def smooth_path_second_pass(self, path, obstacles=None, obstacle_sizes=None):
        """Apply a second pass of smoothing for large robots."""
        if len(path) < 5:  # Need enough points for meaningful smoothing
            return path
            
        result = [path[0]]  # Keep the start point unchanged
        
        # Apply spline-like smoothing by considering multiple neighboring points
        for i in range(1, len(path)-1):
            # Get neighboring points
            p_prev = path[max(0, i-2)]
            p_curr = path[i]
            p_next = path[min(len(path)-1, i+2)]
            
            # Calculate a weighted average, giving more weight to the current point
            smoothed = 0.25 * p_prev + 0.5 * p_curr + 0.25 * p_next
            
            # Check for collisions
            if obstacles is not None and self.check_collision(smoothed[:2], obstacles, obstacle_sizes):
                # If collision, use original point
                result.append(path[i])
            else:
                result.append(smoothed)
            
        result.append(path[-1])  # Keep the end point unchanged
        return result
    
    def create_ros_path(self, path_points: list, frame_id: str = "map") -> Path:
        """Convert path points to ROS Path message."""
        path_msg = Path()
        path_msg.header.frame_id = frame_id
        path_msg.header.stamp = rclpy.time.Time().to_msg()

        for i, point in enumerate(path_points):
            pose = PoseStamped()
            pose.header = path_msg.header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = float(point[2])
            
            # Calculate orientation to face next point
            if i < len(path_points) - 1:
                next_point = path_points[i + 1]
                dx = next_point[0] - point[0]
                dy = next_point[1] - point[1]
                theta = math.atan2(dy, dx)
                pose.pose.orientation.z = math.sin(theta/2.0)
                pose.pose.orientation.w = math.cos(theta/2.0)
            elif i > 0:
                # Keep previous orientation for last point
                pose.pose.orientation = path_msg.poses[-1].pose.orientation
            else:
                # Default orientation for first point if no next point
                pose.pose.orientation.w = 1.0
                
            path_msg.poses.append(pose)
            
        return path_msg
    
    def publish_wavefront_grid_visualization(self, publisher, frame_id="map", robot_pose=None, show_distance=True):
        """
        Enhanced visualization of wavefront grid with flow directions and additional information.
        Now also shows distance field for better passage visualization.
        
        Parameters:
        -----------
        publisher : rclpy.publisher
            ROS publisher for the visualization markers
        frame_id : str
            Frame ID for the markers
        robot_pose : Pose, optional
            Current robot pose for visualization
        show_distance : bool, optional
            Whether to show the distance field (True) or the wavefront grid (False)
        """
        # Get current time
        current_time = time.time()
        
        # Check if we need to update the visualization
        update_needed = False
        
        # Update if enough time has passed since last visualization
        if current_time - self.last_visualization_time >= self.visualization_interval:
            update_needed = True
        
        # Check if grid has changed significantly
        if self.wavefront_grid is not None and not update_needed:
            # Create a simple hash for a portion of the grid to detect changes
            hasher = hashlib.md5()
            sample = self.wavefront_grid[::5, ::5]  # Sample every 5th element
            hasher.update(sample.tobytes())
            current_hash = hasher.hexdigest()
            
            # Update if grid has changed
            if self.last_grid_hash != current_hash:
                update_needed = True
                self.last_grid_hash = current_hash
        
        # Update if grid hasn't been visualized yet or if we're forced to
        if self.last_visualization_markers is None:
            update_needed = True
        
        # If robot pose has changed significantly, update visualization
        if robot_pose is not None and self.last_robot_pos is not None:
            robot_moved = (abs(robot_pose.position.x - self.last_robot_pos[0]) > 0.5 or
                        abs(robot_pose.position.y - self.last_robot_pos[1]) > 0.5)
            if robot_moved:
                update_needed = True
        
        # Update if forced
        if self.force_visualization:
            update_needed = True
            self.force_visualization = False
        
        # If no update needed and we have previous markers, republish them
        if not update_needed and self.last_visualization_markers is not None:
            publisher.publish(self.last_visualization_markers)
            return
        
        # Update the last visualization time
        self.last_visualization_time = current_time
        
        # If distance field visualization is requested and available
        if show_distance and self.distance_field is not None:
            # Publish the distance field visualization 
            self.publish_distance_field_visualization(publisher, frame_id, robot_pose)
            return
            
        # Create new marker array for wavefront grid visualization
        marker_array = MarkerArray()
        
        # If no wavefront grid, publish empty marker array
        if self.wavefront_grid is None:
            publisher.publish(marker_array)
            self.last_visualization_markers = marker_array
            return
        
        # Clear previous markers
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rclpy.time.Time().to_msg()
        marker.ns = "wavefront_grid"
        marker.id = 0
        marker.action = Marker.DELETEALL
        marker_array.markers.append(marker)
        publisher.publish(marker_array)
        
        # Now create the actual visualization markers
        marker_array = MarkerArray()
        
        # Get min and max values for color scaling
        valid_cells = self.wavefront_grid[self.wavefront_grid != -1]
        if valid_cells.size == 0:
            self.last_visualization_markers = marker_array
            publisher.publish(marker_array)
            return
        
        min_val = valid_cells.min()
        max_val = valid_cells.max()
        
        # Get robot pose information for transformation
        robot_x, robot_y, robot_yaw = 0.0, 0.0, 0.0
        if robot_pose is not None:
            robot_x = robot_pose.position.x
            robot_y = robot_pose.position.y
            
            # Extract yaw from quaternion
            qx = robot_pose.orientation.x
            qy = robot_pose.orientation.y
            qz = robot_pose.orientation.z
            qw = robot_pose.orientation.w
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Determine step size based on visualization level and grid size
        if self.visualization_level == 0:  # Minimal
            grid_step = 6
            show_flow = False
        elif self.visualization_level == 1:  # Normal
            grid_step = 4
            show_flow = True
            flow_step = 10
        else:  # Detailed
            grid_step = 3
            show_flow = True
            flow_step = 8
        
        # Add a marker for each cell with a valid wavefront value (downsample for efficiency)
        marker_id = 0
        for y in range(0, self.grid_height, grid_step):
            for x in range(0, self.grid_width, grid_step):
                if self.wavefront_grid[y, x] != -1:
                    # Create a cube marker for each cell
                    marker = Marker()
                    marker.header.frame_id = frame_id
                    marker.header.stamp = rclpy.time.Time().to_msg()
                    marker.ns = "wavefront_grid"
                    marker.id = marker_id
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    
                    # Position at cell center (convert grid to world)
                    local_x, local_y = self.grid_to_world(x, y)
                    
                    # Transform to map coordinates (relative to robot)
                    if robot_pose is not None:
                        # Rotate and translate based on robot pose
                        world_x = robot_x + local_x * math.cos(robot_yaw) - local_y * math.sin(robot_yaw)
                        world_y = robot_y + local_x * math.sin(robot_yaw) + local_y * math.cos(robot_yaw)
                    else:
                        world_x, world_y = local_x, local_y
                    
                    marker.pose.position.x = float(world_x)
                    marker.pose.position.y = float(world_y)
                    marker.pose.position.z = 0.02  # Just above ground
                    
                    # Default orientation
                    marker.pose.orientation.w = 1.0
                    
                    # Size based on grid step
                    marker.scale.x = self.resolution * grid_step
                    marker.scale.y = self.resolution * grid_step
                    marker.scale.z = 0.01  # Very thin in z
                    
                    # Color based on wavefront value (blue to red gradient)
                    val = self.wavefront_grid[y, x]
                    color = self._get_heat_map_color(val, min_val, max_val)
                    
                    marker.color.r = float(color[0])
                    marker.color.g = float(color[1])
                    marker.color.b = float(color[2])
                    marker.color.a = 0.6  # Semi-transparent
                    
                    marker_array.markers.append(marker)
                    marker_id += 1
        
        # Add flow field visualization if available and requested
        if show_flow and self.flow_field is not None:
            for y in range(0, self.grid_height, flow_step):
                for x in range(0, self.grid_width, flow_step):
                    if self.wavefront_grid[y, x] != -1 and self.occupancy_grid[y, x] == 0:
                        flow_dir = self.flow_field[y, x]
                        
                        # Skip cells with no clear flow direction
                        if np.linalg.norm(flow_dir) < 0.1:
                            continue
                            
                        # Create arrow marker for flow direction
                        arrow = Marker()
                        arrow.header.frame_id = frame_id
                        arrow.header.stamp = rclpy.time.Time().to_msg()
                        arrow.ns = "flow_field"
                        arrow.id = marker_id
                        arrow.type = Marker.ARROW
                        arrow.action = Marker.ADD
                        
                        # Position at cell center
                        local_x, local_y = self.grid_to_world(x, y)
                        
                        # Transform to map coordinates
                        if robot_pose is not None:
                            world_x = robot_x + local_x * math.cos(robot_yaw) - local_y * math.sin(robot_yaw)
                            world_y = robot_y + local_x * math.sin(robot_yaw) + local_y * math.cos(robot_yaw)
                            
                            # Rotate flow direction
                            rotated_dx = flow_dir[0] * math.cos(robot_yaw) - flow_dir[1] * math.sin(robot_yaw)
                            rotated_dy = flow_dir[0] * math.sin(robot_yaw) + flow_dir[1] * math.cos(robot_yaw)
                            flow_dir = np.array([rotated_dx, rotated_dy])
                        else:
                            world_x, world_y = local_x, local_y
                        
                        # Normalize and scale flow direction
                        flow_dir = flow_dir / np.linalg.norm(flow_dir)
                        scale = 0.3  # Arrow length
                        
                        # Start of arrow
                        arrow.points.append(Point(x=float(world_x), y=float(world_y), z=0.05))
                        
                        # End of arrow
                        arrow.points.append(Point(
                            x=float(world_x + flow_dir[0] * scale),
                            y=float(world_y + flow_dir[1] * scale),
                            z=0.05
                        ))
                        
                        # Arrow properties
                        arrow.scale.x = 0.02  # Shaft diameter
                        arrow.scale.y = 0.05  # Head diameter
                        arrow.scale.z = 0.05  # Head length
                        
                        # Blue color for flow direction
                        arrow.color.r = 0.0
                        arrow.color.g = 0.5
                        arrow.color.b = 1.0
                        arrow.color.a = 0.8
                        
                        marker_array.markers.append(arrow)
                        marker_id += 1
        
        # Visualize obstacles
        obstacle_step = grid_step - 1 if grid_step > 2 else 2
        for y in range(0, self.grid_height, obstacle_step):
            for x in range(0, self.grid_width, obstacle_step):
                if self.occupancy_grid[y, x] == 1:
                    # Create a cube marker for each obstacle
                    marker = Marker()
                    marker.header.frame_id = frame_id
                    marker.header.stamp = rclpy.time.Time().to_msg()
                    marker.ns = "obstacles"
                    marker.id = marker_id
                    marker.type = Marker.CUBE
                    marker.action = Marker.ADD
                    
                    # Position at cell center
                    local_x, local_y = self.grid_to_world(x, y)
                    
                    # Transform to map coordinates
                    if robot_pose is not None:
                        world_x = robot_x + local_x * math.cos(robot_yaw) - local_y * math.sin(robot_yaw)
                        world_y = robot_y + local_x * math.sin(robot_yaw) + local_y * math.cos(robot_yaw)
                    else:
                        world_x, world_y = local_x, local_y
                    
                    marker.pose.position.x = float(world_x)
                    marker.pose.position.y = float(world_y)
                    marker.pose.position.z = 0.1  # Higher than wavefront
                    
                    # Default orientation
                    marker.pose.orientation.w = 1.0
                    
                    # Size is resolution * step
                    marker.scale.x = self.resolution * obstacle_step
                    marker.scale.y = self.resolution * obstacle_step
                    marker.scale.z = 0.05  # Thicker than wavefront
                    
                    # Red for obstacles
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 0.8
                    
                    marker_array.markers.append(marker)
                    marker_id += 1
        
        # Store and publish the new markers
        self.last_visualization_markers = marker_array
        publisher.publish(marker_array)
        
        # Also visualize detected passages if enabled
        if self.enable_local_minima_optimization and len(self.local_minima_regions) > 0:
            self.publish_detected_passages(publisher, frame_id, robot_pose)


    
        
    def publish_distance_field_visualization(self, publisher, frame_id="map", robot_pose=None):
        """
        Visualize the distance field showing clearance between obstacles.
        This highlights potential passages between obstacles.
        """
        if self.distance_field is None or publisher is None:
            return
            
        marker_array = MarkerArray()
        
        # Determine visualization detail based on level
        if self.visualization_level == 0:  # Minimal
            grid_step = 8
        elif self.visualization_level == 1:  # Normal
            grid_step = 5
        else:  # Detailed
            grid_step = 3
            
        # Get min and max values for color scaling
        min_val = 0.0  # Distance 0 means obstacle
        max_val = np.max(self.distance_field)
        if max_val == 0:
            return
            
        # Get robot pose information for transformation
        robot_x, robot_y, robot_yaw = 0.0, 0.0, 0.0
        if robot_pose is not None:
            robot_x = robot_pose.position.x
            robot_y = robot_pose.position.y
            
            # Extract yaw from quaternion
            qx = robot_pose.orientation.x
            qy = robot_pose.orientation.y
            qz = robot_pose.orientation.z
            qw = robot_pose.orientation.w
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Visualize the distance field
        marker_id = 0
        for y in range(0, self.grid_height, grid_step):
            for x in range(0, self.grid_width, grid_step):
                # Skip obstacle cells
                if self.occupancy_grid[y, x] == 1:
                    continue
                    
                # Get the distance at this cell
                distance = self.distance_field[y, x]
                
                # Create marker
                marker = Marker()
                marker.header.frame_id = frame_id
                marker.header.stamp = rclpy.time.Time().to_msg()
                marker.ns = "distance_field"
                marker.id = marker_id
                marker.type = Marker.CUBE
                marker.action = Marker.ADD
                
                # Position at cell center (convert grid to world)
                local_x, local_y = self.grid_to_world(x, y)
                
                # Transform to map coordinates (relative to robot)
                if robot_pose is not None:
                    # Rotate and translate based on robot pose
                    world_x = robot_x + local_x * math.cos(robot_yaw) - local_y * math.sin(robot_yaw)
                    world_y = robot_y + local_x * math.sin(robot_yaw) + local_y * math.cos(robot_yaw)
                else:
                    world_x, world_y = local_x, local_y
                
                marker.pose.position.x = float(world_x)
                marker.pose.position.y = float(world_y)
                marker.pose.position.z = -0.1 
                
                # Default orientation
                marker.pose.orientation.w = 1.0
                
                # Size based on grid step
                marker.scale.x = self.resolution * grid_step
                marker.scale.y = self.resolution * grid_step
                marker.scale.z = 0.01  # Very thin in z
                
                # Color based on normalized distance (from green to yellow)
                # Green for wide passages, yellow/red for narrow passages
                norm_val = distance / max_val
                
                # Use a color gradient:
                # - Red for very narrow passages (just barely passable)
                # - Yellow for medium passages
                # - Green for wide passages
                
                if norm_val < 0.3:  # Narrow passage
                    r = 1.0
                    g = norm_val * 3.0  # 0 to 1.0 as norm_val goes from 0 to 0.33
                    b = 0.0
                elif norm_val < 0.6:  # Medium passage
                    r = 1.0 - (norm_val - 0.3) * 3.0  # 1.0 to 0 as norm_val goes from 0.33 to 0.67
                    g = 1.0
                    b = 0.0
                else:  # Wide passage
                    r = 0.0
                    g = 1.0
                    b = (norm_val - 0.6) * 2.5  # 0 to 1.0 as norm_val goes from 0.67 to 1.0
                    
                marker.color.r = float(r)
                marker.color.g = float(g)
                marker.color.b = float(b)
                marker.color.a = 0.7  # Semi-transparent
                
                marker_array.markers.append(marker)
                marker_id += 1
        
        # Publish the markers
        publisher.publish(marker_array)
    
    def publish_detected_passages(self, publisher, frame_id="map", robot_pose=None):
        """
        Visualize the detected passages between obstacles.
        These are potential paths through local minima.
        """
        if not self.local_minima_regions or publisher is None:
            return
            
        marker_array = MarkerArray()
        
        # Get robot pose information for transformation
        robot_x, robot_y, robot_yaw = 0.0, 0.0, 0.0
        if robot_pose is not None:
            robot_x = robot_pose.position.x
            robot_y = robot_pose.position.y
            
            # Extract yaw from quaternion
            qx = robot_pose.orientation.x
            qy = robot_pose.orientation.y
            qz = robot_pose.orientation.z
            qw = robot_pose.orientation.w
            siny_cosp = 2.0 * (qw * qz + qx * qy)
            cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
            robot_yaw = math.atan2(siny_cosp, cosy_cosp)
        
        # Add markers for each detected passage
        for i, passage in enumerate(self.local_minima_regions):
            # Position from passage info
            local_x, local_y = passage['position']
            
            # Transform to map coordinates if needed
            if robot_pose is not None:
                # Rotate and translate based on robot pose
                world_x = robot_x + local_x * math.cos(robot_yaw) - local_y * math.sin(robot_yaw)
                world_y = robot_y + local_x * math.sin(robot_yaw) + local_y * math.cos(robot_yaw)
            else:
                world_x, world_y = local_x, local_y
            
            # Create sphere marker for passage
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = rclpy.time.Time().to_msg()
            marker.ns = "passages"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            
            marker.pose.position.x = world_x
            marker.pose.position.y = world_y
            marker.pose.position.z = 0.1  # Slightly above ground
            marker.pose.orientation.w = 1.0
            
            # Size based on clearance
            clearance = passage['clearance']
            fitness = passage['fitness']
            
            # Scale for visualization
            marker.scale.x = min(2.0, clearance)
            marker.scale.y = min(2.0, clearance)
            marker.scale.z = 0.1  # Thin in z-axis
            
            # Color based on fitness (green for good, yellow for borderline)
            if fitness > 0.7:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
                
            marker.color.a = 0.6  # Semi-transparent
            
            marker_array.markers.append(marker)
            
            # Add text label with clearance info
            text = Marker()
            text.header = marker.header
            text.ns = "passage_labels"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            
            text.pose.position.x = world_x
            text.pose.position.y = world_y
            text.pose.position.z = 0.3  # Above the sphere
            text.pose.orientation.w = 1.0
            
            text.text = f"C: {clearance:.2f}m\nF: {fitness:.2f}"
            text.scale.z = 0.2  # Text size
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            
            marker_array.markers.append(text)
        
        publisher.publish(marker_array)