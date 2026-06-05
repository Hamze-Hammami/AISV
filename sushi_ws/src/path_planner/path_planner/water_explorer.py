#!/usr/bin/env python3
"""
Lightweight Reactive Pixel-Based Water Explorer using Density Analysis.

This module evaluates water pixels individually using:
1. Density Score: How much water surrounds each pixel
2. Directional Bias: Whether more water is on the left or right
3. Depth Information: How deep the water is at each potential target

The combination of these scores determines the best exploration target.
"""

import numpy as np
import math
import cv2
from geometry_msgs.msg import Pose, PoseStamped
import time
from cv_bridge import CvBridge
from collections import deque
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import MarkerArray, Marker


class PixelBasedWaterExplorer:
    """
    Lightweight reactive water explorer that uses density and depth information
    to find optimal exploration targets.
    """
    
    def __init__(self, node=None):
        """Initialize the explorer with minimal parameters."""
        # Store reference to ROS node for logging
        self.node = node
        self.enable_water_explorer = True  # Enable the water explorer
        
        # Exploration parameters
        self.min_distance = 1.0        # Minimum exploration distance (m)
        self.max_distance = 3.5        # Maximum exploration distance (m)
        self.mask_resolution = 0.05    # Meters per pixel in water mask (m/px)
        self.fov_degrees = 72.0        # Camera field of view in degrees
        
        # Density analysis parameters
        self.density_kernel_size = 15  # Reduced from 21 to 15 for performance
        self.density_sigma = 3.0       # Sigma for Gaussian kernel
        
        # Weighting parameters (modified to remove coverage weight)
        self.density_weight = 0.7      # Increased weight for water density score
        
        # Grid sampling parameters - increased step for better performance
        self.grid_step = 20            # Sample every 20th pixel (increased from 10)
        self.max_candidates = 50       # Reduced from 100 to 50
        
        # Direction bias for exploration
        self.direction_bias = 0.0      # Bias between left/right sides (-0.5 to 0.5)
        self.left_coverage = 0.0       # Water coverage on left side (0-1)
        self.right_coverage = 0.0      # Water coverage on right side (0-1)
        
        # Store the current best score separately (not as an attribute of the Pose object)
        self.current_goal_score = 0.0  # Score of the current goal
        
        # Processing and visualization parameters
        self.bridge = CvBridge()
        
        # State variables
        self.water_mask = None         # Latest water mask
        self.depth_map = None          # Latest depth map (in cm)
        self.mask_lock = None          # Thread lock for water mask
        self.depth_lock = None         # Thread lock for depth map
        self.current_goal = None       # Current exploration goal
        self.goal_timestamp = 0        # When the current goal was generated
        self.goal_timeout = 0.2        # Reduced from 5.0 to 2.0 seconds for faster reactions
        
        # Visualization data
        self.density_map = None        # Visualization of density scores
        self.combined_map = None       # Visualization of combined scores
        self.best_point = None         # Selected best point
        self.candidate_points = []     # Candidate exploration points
        
        # Create the density kernel (smaller for performance)
        self.density_kernel = cv2.getGaussianKernel(self.density_kernel_size, self.density_sigma)
        self.density_kernel = self.density_kernel * self.density_kernel.T
        
        # Computation monitoring
        self.last_computation_time = 0
        self.computation_timeout = 1.0  # Reduced from 2.0 to 1.0 seconds
        self.is_computing = False       # Flag to track computation state
        
        # Logging
        self.log("info", "PixelBasedWaterExplorer without coverage tracking initialized")
        self.clock = node.get_clock() if node else None  # Pass the clock from the node
        self.frame_id = "camera_link"  # Default frame ID for visualization
    
    def get_logger(self):
        """Return the node's logger if available, otherwise return a dummy logger."""
        if self.node and hasattr(self.node, "get_logger"):
            return self.node.get_logger()
        else:
            # Return a dummy logger that implements the basic logging methods
            class DummyLogger:
                def info(self, msg): print(f"[INFO] {msg}")
                def warn(self, msg): print(f"[WARN] {msg}")
                def error(self, msg): print(f"[ERROR] {msg}")
                def debug(self, msg): print(f"[DEBUG] {msg}")
            return DummyLogger()
            
    def log(self, level, message):
        """Log messages using the node's logger if available."""
        if self.node and hasattr(self.node, "get_logger"):
            logger = self.node.get_logger()
            if level == "info":
                logger.info(message)
            elif level == "warn":
                logger.warn(message)
            elif level == "error":
                logger.error(message)
            elif level == "debug":
                logger.debug(message)
        else:
            print(f"[{level.upper()}] {message}")
    
    def update_water_mask(self, mask_msg):
        """Update the stored water mask."""
        try:
            # Convert message to CV2 image if needed
            if hasattr(mask_msg, "encoding"):
                cv_mask = self.bridge.imgmsg_to_cv2(mask_msg, "mono8")
            else:
                cv_mask = mask_msg
                
            # Normalize mask if needed (ensure 0-1 range where 1=water)
            if np.max(cv_mask) > 1.0:
                cv_mask = (cv_mask > 128).astype(np.uint8)
            
            # Store mask with thread safety if lock exists
            if self.mask_lock:
                with self.mask_lock:
                    self.water_mask = cv_mask
            else:
                self.water_mask = cv_mask
            
            # Reset goal if mask has changed significantly
            if self.current_goal:
                self._check_for_mask_change(cv_mask)
                
            return True
        except Exception as e:
            self.log("error", f"Error updating water mask: {e}")
            return False
    
    def update_depth_map(self, depth_msg):
        """Update the stored depth map (expects 16UC1 format in cm)."""
        try:
            # Convert message to CV2 image if needed
            if hasattr(depth_msg, "encoding"):
                # DepthAnything publishes 16UC1 where values are in centimeters
                cv_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="16UC1")
            else:
                cv_depth = depth_msg
            
            # Check if we have valid depth data
            if np.max(cv_depth) > 0:
                # Store with thread safety if lock exists
                if self.depth_lock:
                    with self.depth_lock:
                        self.depth_map = cv_depth
                else:
                    self.depth_map = cv_depth
                
                return True
            else:
                self.log("warn", "Received depth map with all zeros")
                return False
                
        except Exception as e:
            self.log("error", f"Error updating depth map: {e}")
            return False
    
    def _check_for_mask_change(self, new_mask):
        """Check if water mask has changed significantly."""
        if self.water_mask is None:
            return
            
        # Only check if we have a current goal
        if self.current_goal is not None:
            try:
                # Ensure masks have same dimensions
                if self.water_mask.shape != new_mask.shape:
                    self.current_goal = None
                    return
                
                # Calculate change percentage
                diff = np.abs(self.water_mask - new_mask)
                change_percentage = np.sum(diff) / (new_mask.shape[0] * new_mask.shape[1])
                
                # If more than 15% of the mask has changed, reset goal
                if change_percentage > 0.15:
                    self.current_goal = None
            except Exception as e:
                self.log("error", f"Error comparing masks: {e}")
    
    def _calculate_density_map(self, water_mask):
        """Calculate water density map with stricter density requirements."""
        if np.sum(water_mask) < 100:
            self.log("debug", "Minimal water in mask - using simplified density")
            return water_mask.astype(np.float32)
            
        # Use a larger kernel for better discrimination of dense water areas
        kernel_size = max(31, self.density_kernel_size * 2)
        sigma = self.density_sigma * 1.5
        
        # Create a larger Gaussian kernel for better area coverage analysis
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel * kernel.T
        
        # Apply the convolution
        density_map = cv2.filter2D(water_mask.astype(np.float32), -1, kernel)
        
        # Apply non-linear scaling to enhance density differences
        if np.max(density_map) > 0:
            density_map = density_map / np.max(density_map)
            # Apply power function to make high density areas more distinct
            density_map = np.power(density_map, 2)
        
        return density_map

    def _analyze_directional_bias(self, water_mask):
        """
        Analyze left-right directional bias in the water mask.
        Returns a directional bias map that favors less explored directions.
        """
        h, w = water_mask.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # Start with a uniform map for water areas
        directional_map = water_mask.astype(np.float32)
        
        # Divide mask into left and right halves - from robot's perspective
        right_half = water_mask.copy()
        right_half[:, center_x:] = 0  # Zero out left side
        
        left_half = water_mask.copy()
        left_half[:, :center_x] = 0  # Zero out right side
        
        # Calculate water coverage in each half
        total_right_pixels = right_half.shape[0] * (right_half.shape[1] // 2) if right_half.shape[1] > 0 else 1
        total_left_pixels = left_half.shape[0] * (left_half.shape[1] // 2) if left_half.shape[1] > 0 else 1
        
        right_coverage = np.sum(right_half) / total_right_pixels
        left_coverage = np.sum(left_half) / total_left_pixels
        
        # Calculate direction bias (positive = favor right, negative = favor left)
        direction_bias = right_coverage - left_coverage
        
        # Normalize bias to -0.5 to 0.5 range
        max_bias = max(abs(direction_bias), 0.001)  # Avoid division by zero
        normalized_bias = direction_bias / (max_bias * 2)
        
        # Store the values for logging/visualization
        self.direction_bias = normalized_bias
        self.right_coverage = right_coverage
        self.left_coverage = left_coverage
        
        self.log("debug", f"Right: {right_coverage:.3f}, Left: {left_coverage:.3f}, Bias: {normalized_bias:.3f}")
        
        # Create left-right gradient mask from -0.5 (left) to 0.5 (right)
        x_coords = np.linspace(-0.5, 0.5, w)
        direction_mask = np.tile(x_coords, (h, 1))
        
        # Apply the bias: 
        # - Positive bias boosts right side
        # - Negative bias boosts left side
        bias_mask = 0.5 - np.abs(direction_mask - normalized_bias)
        
        # Enhance directional map based on direction bias (0.5 to 1.5 range)
        directional_map = directional_map * (1.0 + bias_mask)
        
        # Apply distance transform to prioritize points far from non-water
        if np.sum(water_mask) > 0:
            dist_transform = cv2.distanceTransform(water_mask, cv2.DIST_L2, 3)
            # Normalize
            if np.max(dist_transform) > 0:
                dist_transform = dist_transform / np.max(dist_transform)
            
            # Combine with directional map
            directional_map = directional_map * (0.5 + 0.5 * dist_transform)
        
        # Normalize final directional map
        if np.max(directional_map) > 0:
            directional_map = directional_map / np.max(directional_map)
        
        return directional_map

    def _analyze_water_directionality(self, water_mask):
        """
        Analyze water distribution to detect directional trends and potential exits from dead ends.
        Enhanced to detect escape routes in all directions with multi-scale reactive sampling.
        """
        h, w = water_mask.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # 1. Calculate distance transform to find distance to nearest non-water pixel
        water_dist = cv2.distanceTransform(water_mask.astype(np.uint8), cv2.DIST_L2, 3)
        
        # 2. MULTI-DIRECTIONAL ANALYSIS: Detect promising water areas in all directions
        # Instead of explicitly favoring left/right, we'll analyze all directions more evenly
        
        # First, create a directional bias that's less biased toward left/right
        # This creates a more uniform radial preference
        directional_bias = np.zeros_like(water_dist, dtype=np.float32)
        
        # Calculate distance from center for each pixel
        Y, X = np.ogrid[:h, :w]
        Y = Y - center_y
        X = X - center_x
        
        # Calculate radial distance
        distance_from_center = np.sqrt(X**2 + Y**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        normalized_distance = np.minimum(distance_from_center / (max_distance/2), 1.0)
        
        # Apply moderate bias that favors points farther from center
        # This is directionally uniform (doesn't prefer left/right/up/down)
        directional_bias = np.power(normalized_distance, 1.2)
        
        # 3. REACTIVE WATER SAMPLING: Use a grid of sample points across entire water area
        # This creates a reactivity map that identifies promising areas regardless of direction
        reactivity_map = np.zeros_like(water_mask, dtype=np.float32)
        
        try:
            # Create a grid of sample points across the entire water mask
            # This ensures we don't miss any potential escape routes
            grid_step = max(5, min(w, h) // 40)  # Denser grid (adapted to image size)
            
            # Store the coordinates of water points in the grid
            grid_water_points = []
            
            # Sample the water mask using the grid
            for y in range(0, h, grid_step):
                for x in range(0, w, grid_step):
                    if y < h and x < w and water_mask[y, x]:
                        # This is a water point on our grid
                        grid_water_points.append((y, x))
            
            if len(grid_water_points) > 10:  # Need enough points for analysis
                # For each grid point, analyze local water structure
                for y, x in grid_water_points:
                    # Define local neighborhood with bounds checking
                    size = grid_step * 2
                    y_min = max(0, y - size)
                    y_max = min(h - 1, y + size)
                    x_min = max(0, x - size)
                    x_max = min(w - 1, x + size)
                    
                    # Extract neighborhood
                    neighborhood = water_mask[y_min:y_max+1, x_min:x_max+1]
                    
                    if neighborhood.size == 0:
                        continue
                    
                    # Calculate local water coverage
                    local_coverage = np.sum(neighborhood) / neighborhood.size
                    
                    # Calculate local distance transform
                    local_dist = water_dist[y_min:y_max+1, x_min:x_max+1]
                    avg_dist = np.mean(local_dist[neighborhood > 0]) if np.sum(neighborhood) > 0 else 0
                    
                    # Calculate score based on local properties
                    # Higher score for areas with good water coverage and distance from boundaries
                    local_score = 0.6 * local_coverage + 0.4 * (avg_dist / size)
                    
                    # Apply score to the reactive map
                    reactivity_map[y, x] = local_score
            
            # Interpolate between grid points to create a smooth reactivity map
            if np.max(reactivity_map) > 0:
                # Dilate the points
                dilated_map = cv2.dilate(reactivity_map, np.ones((grid_step, grid_step), np.float32))
                # Smooth the map
                smooth_map = cv2.GaussianBlur(dilated_map, (grid_step*2+1, grid_step*2+1), grid_step/3)
                
                # Normalize
                if np.max(smooth_map) > 0:
                    reactivity_map = smooth_map / np.max(smooth_map)
        
        except Exception as e:
            self.log("warn", f"Error in reactive sampling: {e}")
        
        # 4. MULTI-SCALE FEATURE DETECTION: Find promising water features at different scales
        # This detects potential escape routes at multiple scales in all directions
        feature_map = np.zeros_like(water_mask, dtype=np.float32)
        
        try:
            # Create kernels at different scales and orientations
            kernels = []
            
            # Small square kernel (detects small features in any direction)
            kernels.append(np.ones((3, 3), np.uint8))
            
            # Horizontal kernels (detect horizontal passages)
            kernels.append(np.ones((2, 5), np.uint8))  # Small horizontal
            kernels.append(np.ones((3, 9), np.uint8))  # Medium horizontal
            
            # Vertical kernels (detect vertical passages)
            kernels.append(np.ones((5, 2), np.uint8))  # Small vertical
            kernels.append(np.ones((9, 3), np.uint8))  # Medium vertical
            
            # Diagonal kernels (detect diagonal passages)
            diagonal1 = np.zeros((5, 5), np.uint8)
            diagonal2 = np.zeros((5, 5), np.uint8)
            for i in range(5):
                diagonal1[i, i] = 1
                diagonal2[i, 4-i] = 1
            kernels.append(diagonal1)  # Diagonal top-left to bottom-right
            kernels.append(diagonal2)  # Diagonal top-right to bottom-left
            
            # Process with all kernels
            all_features = np.zeros_like(water_mask, dtype=np.uint8)
            
            for i, kernel in enumerate(kernels):
                # Erode to find regions that would be removed by this kernel
                eroded = cv2.erode(water_mask.astype(np.uint8), kernel, iterations=1)
                # Find the difference - these are potential features
                features = water_mask.astype(np.uint8) - eroded
                
                # Add to all_features with binary OR
                all_features = np.logical_or(all_features, features).astype(np.uint8)
            
            # Use connected components to analyze features
            cc_retval, cc_labels, cc_stats, cc_centroids = cv2.connectedComponentsWithStats(
                all_features, connectivity=8)
            
            # Analyze each potential feature
            min_feature_size = 3  # Very small features can be detected
            feature_scores = np.zeros_like(all_features, dtype=np.float32)
            
            for i in range(1, cc_retval):  # Skip background (0)
                area = cc_stats[i, cv2.CC_STAT_AREA]
                
                if area >= min_feature_size:
                    # Create mask for this component
                    component_mask = (cc_labels == i)
                    
                    # Calculate properties
                    indices = np.where(component_mask)
                    if len(indices[1]) > 0:  # Check if indices exist
                        avg_y = np.mean(indices[0])
                        avg_x = np.mean(indices[1])
                        
                        # Calculate relative position from center
                        rel_y = (avg_y - center_y) / center_y  # -1 to +1
                        rel_x = (avg_x - center_x) / center_x  # -1 to +1
                        
                        # Calculate distance from center
                        dist_from_center = np.sqrt(rel_x**2 + rel_y**2)
                        
                        # Calculate score based on size and position
                        # Slightly favor features farther from center
                        feature_score = 0.5 + 0.5 * min(1.0, dist_from_center * 1.5)
                        
                        # Apply to feature map
                        feature_map[component_mask] = feature_score
            
            # Smooth the feature map
            if np.max(feature_map) > 0:
                feature_map = cv2.GaussianBlur(feature_map, (5, 5), 1.0)
                feature_map = feature_map / np.max(feature_map)
        
        except Exception as e:
            self.log("warn", f"Error in multi-scale feature detection: {e}")
        
        # 5. UNIFORM SECTOR ANALYSIS: Analyze water in all directions more evenly
        # Instead of heavily favoring left/right, we'll analyze all sectors more uniformly
        num_sectors = 16  # For precise directional resolution
        sector_reactivity = np.zeros(num_sectors)
        sector_features = np.zeros(num_sectors)
        sector_water_amount = np.zeros(num_sectors)
        
        # Calculate angles for all pixels
        angles = np.arctan2(Y, X) * 180 / np.pi
        # Convert to 0-360 range
        angles = (angles + 360) % 360
        
        # Calculate sector for each pixel
        sector_size = 360 / num_sectors
        sectors = (angles / sector_size).astype(np.int32) % num_sectors
        
        # Add mild emphasis to lateral sectors - but less than before
        lateral_boost = np.ones(num_sectors) 
        # Right sectors (around 0°, 360°)
        right_sectors = [0, 15, 1]
        # Left sectors (around 180°)
        left_sectors = [7, 8, 9]
        
        # Apply small boost to lateral sectors
        for sector in range(num_sectors):
            if sector in right_sectors or sector in left_sectors:
                lateral_boost[sector] = 1.2  # Reduced from 1.5
        
        # Calculate metrics for each sector
        for sector in range(num_sectors):
            sector_mask = (sectors == sector)
            sector_water = water_mask & sector_mask
            
            # Calculate water amount
            water_amount = np.sum(sector_water)
            if water_amount > 0:
                sector_water_amount[sector] = water_amount / np.sum(sector_mask)
            
            # Calculate reactivity score
            sector_reactivity_pixels = reactivity_map * sector_mask
            reactivity_score = np.sum(sector_reactivity_pixels) / max(1, np.sum(sector_mask))
            sector_reactivity[sector] = reactivity_score
            
            # Calculate feature score
            sector_feature_pixels = feature_map * sector_mask
            feature_score = np.sum(sector_feature_pixels) / max(1, np.sum(sector_mask))
            sector_features[sector] = feature_score
        
        # Normalize sector metrics
        max_water = np.max(sector_water_amount) if np.max(sector_water_amount) > 0 else 1
        max_reactivity = np.max(sector_reactivity) if np.max(sector_reactivity) > 0 else 1
        max_features = np.max(sector_features) if np.max(sector_features) > 0 else 1
        
        norm_water = sector_water_amount / max_water
        norm_reactivity = sector_reactivity / max_reactivity
        norm_features = sector_features / max_features
        
        # Calculate sector scores - more balanced approach
        sector_scores = (0.2 * norm_water + 
                         0.4 * norm_reactivity + 
                         0.4 * norm_features) * lateral_boost
        
        # Log the sector analysis
        for sector in range(num_sectors):
            sector_deg = sector * sector_size
            position_label = ""
            if sector in right_sectors:
                position_label = " (RIGHT)"
            elif sector in left_sectors:
                position_label = " (LEFT)"
                
            self.log("debug", f"Sector {sector} ({sector_deg}°){position_label}: " +
                           f"Water={norm_water[sector]:.2f}, " +
                           f"Reactivity={norm_reactivity[sector]:.2f}, " +
                           f"Features={norm_features[sector]:.2f}, " +
                           f"Score={sector_scores[sector]:.2f}")
        
        # Find the top sectors by score (favoring high-scoring sectors in all directions)
        top_sectors = np.argsort(sector_scores)[-6:]  # Increased from 4 to 6
        top_sector_scores = sector_scores[top_sectors]
        
        # Log top sectors
        self.log("info", f"Top sectors: {top_sectors} with scores {top_sector_scores}")
        
        # Create a sector map (each pixel contains its sector index)
        sector_map = sectors.copy()
        
        # 6. COMBINE ALL ANALYSES into a unified directionality map
        # This creates a comprehensive, multi-directional reactivity map
        directionality_map = np.zeros_like(water_mask, dtype=np.float32)
        
        # Apply sector scores
        for y in range(h):
            for x in range(w):
                if water_mask[y, x]:
                    # Get corresponding sector
                    sector = sector_map[y, x]
                    # Start with sector score
                    directionality_map[y, x] = 0.3 * sector_scores[sector]
        
        # Add reactive sampling information
        if np.max(reactivity_map) > 0:
            directionality_map += 0.4 * reactivity_map * water_mask.astype(np.float32)
            
        # Add feature information
        if np.max(feature_map) > 0:
            directionality_map += 0.3 * feature_map * water_mask.astype(np.float32)
            
        # Normalize the combined map
        if np.max(directionality_map) > 0:
            directionality_map = directionality_map / np.max(directionality_map)
            
        # Apply directional bias
        directionality_map = directionality_map * (0.8 + 0.2 * directional_bias)
        
        # Final normalization
        if np.max(directionality_map) > 0:
            directionality_map = directionality_map / np.max(directionality_map)
            
        return directionality_map, sector_map, top_sectors

    def _calculate_point_depth(self, point, depth_map, window_size=20):
        """
        Calculate depth at a point using a robust method.
        Takes a small window around the point and applies filtering logic.
        
        Args:
            point: (y, x) coordinates of the point
            depth_map: Depth map (in cm)
            window_size: Size of window to sample around point
        
        Returns:
            depth_m: Depth in meters or None if no valid depth
        """
        if depth_map is None:
            return None
            
        y, x = point
        h, w = depth_map.shape[:2]
        
        # Calculate window boundaries with bounds checking
        half_window = window_size // 2
        x1 = max(0, x - half_window)
        x2 = min(w - 1, x + half_window)
        y1 = max(0, y - half_window)
        y2 = min(h - 1, y + half_window)
        
        # Check if window is valid
        if x2 <= x1 or y2 <= y1:
            return None
        
        # Extract ROI
        roi = depth_map[y1:y2, x1:x2]
        
        # Get valid depths (non-zero) in mm
        valid_depths = roi[roi > 0]
        
        if valid_depths.size > 5:  # Need some minimum number of valid points
            # Use a low percentile to get reliable depth - ignore outliers
            depth_cm = np.percentile(valid_depths, 15)
            # Convert to meters and limit to reasonable range
            depth_m = max(0.5, min(5.0, depth_cm / 100.0))
            return depth_m
        
        return None

    def _find_candidate_pixels(self, density_map, water_mask, depth_map=None):
        """Find candidate pixels using multi-directional reactive sampling."""
        h, w = density_map.shape[:2]
        center_y, center_x = h // 2, w // 2
        
        # CRITICAL SAFETY IMPROVEMENT: Filter small isolated water regions
        # Apply connected component analysis to identify and remove small water regions
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(water_mask.astype(np.uint8), connectivity=8)
        
        # Create a safe water mask by removing small regions
        safe_water_mask = np.zeros_like(water_mask)
        min_safe_size = 100  # Minimum size of a water region to be considered safe
        
        # Skip label 0 (background)
        for label in range(1, num_labels):
            size = stats[label, cv2.CC_STAT_AREA]
            if size >= min_safe_size:
                safe_water_mask[labels == label] = 1
                
        # Log how much water was filtered out as unsafe
        original_water_pixels = np.sum(water_mask)
        safe_water_pixels = np.sum(safe_water_mask)
        if original_water_pixels > 0:
            filtered_percentage = 100 * (1 - safe_water_pixels / original_water_pixels)
            self.log("debug", f"Filtered out {filtered_percentage:.1f}% of water pixels as potentially unsafe")
        
        # Use the safe water mask for all further operations
        water_mask = safe_water_mask
        
        # Analyze water directionality to detect promising exploration directions
        directionality_map, sector_map, top_sectors = self._analyze_water_directionality(water_mask)
        
        # Create a mask that includes water pixels from the top sectors
        # Increased the number of top sectors from 4 to 6 for more comprehensive sampling
        top_sectors_mask = np.zeros_like(water_mask)
        for sector in top_sectors:
            top_sectors_mask = np.logical_or(top_sectors_mask, (sector_map == sector) & water_mask)
        
        # If the top sectors mask has too few pixels, fall back to the original water mask
        if np.sum(top_sectors_mask) < 50:
            self.log("warn", "Top sectors have insufficient water, using full water mask")
            sector_restricted_mask = water_mask
        else:
            self.log("info", f"Restricting search to top sectors {top_sectors} with {np.sum(top_sectors_mask)} pixels")
            sector_restricted_mask = top_sectors_mask
        
        # Calculate combined score map - now using primarily directionality map
        # This favors water in promising directions in all sectors, not just left/right
        combined_score = directionality_map * sector_restricted_mask
        
        # Apply threshold to eliminate low-score candidates
        score_threshold = 0.3  # Reduced from 0.4 for more comprehensive sampling
        combined_score[combined_score < score_threshold] = 0
        
        # Normalize combined score
        if np.max(combined_score) > 0:
            combined_score = combined_score / np.max(combined_score)
            
        # ENHANCED WATER AREA SAMPLING: Use a dense grid across entire water area
        # This ensures we don't miss subtle escape routes in any direction
        # Create a sampling grid that's adaptive to the image size
        grid_step = max(5, min(w, h) // 40)  # Dense sampling grid
        
        # Apply local water sampling analysis
        local_water_coverage = np.zeros_like(water_mask, dtype=np.float32)
        
        # Create a larger sampling window
        window_size = grid_step * 3
        kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        
        # Calculate local water coverage across the entire mask
        local_water_coverage = cv2.filter2D(water_mask.astype(np.float32), -1, kernel)
        
        # Apply threshold to local water coverage
        water_threshold = 0.4  # Reduced from 0.5 for more sensitivity
        local_water_coverage[local_water_coverage < water_threshold] = 0
        
        # Normalize water coverage
        if np.max(local_water_coverage) > 0:
            local_water_coverage = local_water_coverage / np.max(local_water_coverage)
        
        # Add local water coverage to combined score
        combined_score = 0.6 * combined_score + 0.4 * local_water_coverage * sector_restricted_mask
        
        # Normalize again after adding local water coverage
        if np.max(combined_score) > 0:
            combined_score = combined_score / np.max(combined_score)
        else:
            self.log("warn", "No viable exploration areas found after water safety filtering")
            
        # Store the combined map for visualization
        self.combined_map = combined_score.copy()
        
        # FOCUS ON BEST POINTS: Find global maximum first
        best_score_idx = np.argmax(combined_score)
        if best_score_idx > 0:  # Valid maximum found
            best_y, best_x = np.unravel_index(best_score_idx, combined_score.shape)
            best_score = combined_score[best_y, best_x]
            if best_score > 0.7:  # If we have a really good maximum
                self.log("info", f"Found excellent candidate with score {best_score:.3f}")
                
                # Generate a single candidate from the global maximum
                rx = (center_x - best_x) * self.mask_resolution  # Flipped X-axis
                ry = (best_y - center_y) * self.mask_resolution
                distance = math.sqrt(rx**2 + ry**2)
                
                if self.min_distance <= distance <= self.max_distance:
                    # Quick depth check
                    depth_cm = None
                    if depth_map is not None:
                        depth_m = self._calculate_point_depth((best_y, best_x), depth_map)
                        if depth_m is not None:
                            depth_cm = int(depth_m * 100.0)
                    
                    # Return a single-item list with the best point
                    return [(best_y, best_x, best_score, distance, depth_cm)]
        
        # MULTI-DIRECTIONAL GRID SAMPLING: Comprehensive sampling across all water areas
        # Create reactive candidates from all promising areas, not just left/right
        candidates = []
        
        # Use a denser grid for more comprehensive sampling
        sampling_grid_step = max(3, grid_step // 2)  # Even denser grid for candidate selection
        
        # Regular grid sampling across all water areas to find candidates
        for y in range(0, h, sampling_grid_step):
            for x in range(0, w, sampling_grid_step):
                if y >= h or x >= w:  # Safety check
                    continue
                    
                # Skip pixels not in water or with low scores
                if not water_mask[y, x] or combined_score[y, x] < score_threshold:
                    continue
                    
                # Convert to robot coordinates
                rx = (center_x - x) * self.mask_resolution  # Flipped X-axis
                ry = (y - center_y) * self.mask_resolution
                
                distance = math.sqrt(rx**2 + ry**2)
                
                if self.min_distance <= distance <= self.max_distance:
                    score = combined_score[y, x]
                    
                    # Add depth-based scoring if available
                    if depth_map is not None:
                        depth_m = self._calculate_point_depth((y, x), depth_map)
                        if depth_m is None:
                            continue
                            
                        # Adjust depth scoring
                        depth_score = 1.0
                        if depth_m < 0.5:
                            depth_score = 0.3
                        elif depth_m < 1.0:
                            depth_score = 0.7
                        elif depth_m > 4.0:
                            depth_score = 0.4
                        elif depth_m > 3.0:
                            depth_score = 0.8
                            
                        score = score * depth_score
                        depth_cm = int(depth_m * 100.0)
                    else:
                        depth_cm = None
                        
                    candidates.append((y, x, score, distance, depth_cm))
                    
                    # Early termination if we find an excellent candidate and have enough candidates
                    if score > 0.9 and len(candidates) >= 10:
                        self.log("debug", f"Early termination with excellent candidate: {score:.3f}")
                        candidates.sort(key=lambda x: x[2], reverse=True)
                        return candidates[:self.max_candidates]
        
        # If we have candidates, sort and return them
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            return candidates[:self.max_candidates]
            
        # If we STILL have no candidates after comprehensive sampling, create a few random samples
        # as a last resort to ensure the robot always has some exploration options
        self.log("warn", "No candidates found, creating fallback random samples")
        random_candidates = []
        
        # Try up to 100 random samples
        for _ in range(100):
            # Generate random coordinates
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            
            if water_mask[y, x]:
                # Convert to robot coordinates
                rx = (center_x - x) * self.mask_resolution
                ry = (y - center_y) * self.mask_resolution
                
                distance = math.sqrt(rx**2 + ry**2)
                
                if self.min_distance <= distance <= self.max_distance:
                    score = 0.5  # Moderate score for random candidates
                    depth_cm = None
                    
                    if depth_map is not None:
                        depth_m = self._calculate_point_depth((y, x), depth_map)
                        if depth_m is not None:
                            depth_cm = int(depth_m * 100.0)
                            
                    random_candidates.append((y, x, score, distance, depth_cm))
                    
                    # Once we have a few random candidates, that's enough
                    if len(random_candidates) >= 5:
                        break
                        
        if random_candidates:
            return random_candidates
            
        # If all else fails, use the default goal
        return []
        
        # If we didn't find an excellent global maximum or it was out of range,
        # proceed with the regular grid search but still restricted to top sectors
        
        # PHASE 1: Coarse grid search to find promising regions
        coarse_candidates = []
        # Use less sparse grid step for coarse search
        coarse_grid_step = self.grid_step * 2
        
        # Fixed grid offset (no randomness) to ensure consistent sampling
        # This prevents missing good areas due to unlucky grid alignment
        for y in range(0, h, coarse_grid_step):
            for x in range(0, w, coarse_grid_step):
                if y >= h or x >= w:  # Safety check
                    continue
                    
                # Skip pixels not in top sectors
                if not sector_restricted_mask[y, x]:
                    continue
                    
                # CRITICAL: Check for sufficient local water coverage
                if local_water_coverage[y, x] < water_threshold:
                    continue
                    
                if not water_mask[y, x] or combined_score[y, x] < score_threshold:
                    continue
                    
                # Convert to robot coordinates with CORRECT AXIS MAPPING
                # CRITICAL: Flip the X-axis to match robot's coordinate system
                rx = (center_x - x) * self.mask_resolution  # Flipped X-axis: center_x - x instead of x - center_x
                ry = (y - center_y) * self.mask_resolution  # Y-axis stays the same
                
                distance = math.sqrt(rx**2 + ry**2)
                
                if self.min_distance <= distance <= self.max_distance:
                    score = combined_score[y, x]
                    
                    # Quick depth check (simpler than full calculation)
                    if depth_map is not None:
                        # Just check if we have valid depth
                        if depth_map[y, x] == 0:
                            continue
                    
                    # Add to coarse candidates with water coverage as an added factor
                    # Plus directionality score for preferring open water directions
                    water_coverage_score = local_water_coverage[y, x]
                    directionality_score = directionality_map[y, x]
                    # New combined score with directional component
                    adjusted_score = 0.3 * score + 0.3 * water_coverage_score + 0.4 * directionality_score
                    coarse_candidates.append((y, x, adjusted_score))
        
        # If coarse search produced no candidates, fall back to standard grid search
        if not coarse_candidates:
            self.log("warn", "Coarse search found no candidates, falling back to standard grid search")
            candidates = self._standard_grid_search(combined_score, water_mask, depth_map, 
                                                  score_threshold, center_x, center_y, h, w,
                                                  local_water_coverage, water_threshold,
                                                  directionality_map, sector_restricted_mask)
            return candidates
        
        # Sort by score and take top N regions to explore further
        coarse_candidates.sort(key=lambda x: x[2], reverse=True)
        num_regions = min(3, len(coarse_candidates))  # Up to 3 promising regions - reduced from 5
        
        # PHASE 2: Fine grid search around promising regions
        candidates = []
        # Use smaller region size to focus more precisely
        region_size = self.grid_step * 2  # Reduced from 3x
        
        # Fine search around each promising region
        for region_idx in range(num_regions):
            if region_idx >= len(coarse_candidates):
                break
                
            y_center, x_center, _ = coarse_candidates[region_idx]
            
            # Define region bounds with bounds checking
            y_min = max(0, y_center - region_size)
            y_max = min(h - 1, y_center + region_size)
            x_min = max(0, x_center - region_size)
            x_max = min(w - 1, x_center + region_size)
            
            # Fine grid step (half of normal grid step for better precision)
            fine_grid_step = max(1, self.grid_step // 2)
            
            # Search within this region with fine grid
            for y in range(y_min, y_max + 1, fine_grid_step):
                for x in range(x_min, x_max + 1, fine_grid_step):
                    if y >= h or x >= w:  # Safety check
                        continue
                        
                    # Skip pixels not in top sectors
                    if not sector_restricted_mask[y, x]:
                        continue
                        
                    # CRITICAL: Check for sufficient local water coverage
                    if local_water_coverage[y, x] < water_threshold:
                        continue
                        
                    if not water_mask[y, x] or combined_score[y, x] < score_threshold:
                        continue
                        
                    # Convert to robot coordinates with CORRECT AXIS MAPPING
                    # CRITICAL: Flip the X-axis to match robot's coordinate system
                    rx = (center_x - x) * self.mask_resolution  # Flipped X-axis: center_x - x instead of x - center_x
                    ry = (y - center_y) * self.mask_resolution  # Y-axis stays the same
                    
                    distance = math.sqrt(rx**2 + ry**2)
                    
                    if self.min_distance <= distance <= self.max_distance:
                        score = combined_score[y, x]
                        
                        # Add depth-based scoring
                        if depth_map is not None:
                            depth_m = self._calculate_point_depth((y, x), depth_map)
                            if depth_m is None:
                                continue
                                
                            # Stricter depth scoring
                            depth_score = 1.0
                            if depth_m < 0.5:
                                depth_score = 0.2  # Reduced from 0.3
                            elif depth_m < 1.0:
                                depth_score = 0.6  # Reduced from 0.7
                            elif depth_m > 4.0:
                                depth_score = 0.3  # Reduced from 0.4
                            elif depth_m > 3.0:
                                depth_score = 0.7  # Reduced from 0.8
                                
                            score = score * depth_score
                            depth_cm = int(depth_m * 100.0)
                        else:
                            depth_cm = None
                        
                        # Add water coverage and directionality as additional factors in final score
                        water_coverage_score = local_water_coverage[y, x]
                        directionality_score = directionality_map[y, x]
                        # Heavily weight directionality (open water areas)
                        final_score = 0.3 * score + 0.3 * water_coverage_score + 0.4 * directionality_score
                            
                        candidates.append((y, x, final_score, distance, depth_cm))
                        
                        # Early termination with higher threshold
                        # If we have found an excellent candidate and have enough candidates, return
                        if final_score > 0.9 and len(candidates) >= 5:
                            self.log("debug", f"Early termination with excellent candidate: {final_score:.3f}")
                            candidates.sort(key=lambda x: x[2], reverse=True)
                            return candidates[:self.max_candidates]
        
        # If we have candidates, return them
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            return candidates[:self.max_candidates]
            
        # If we still have no candidates, use coarse candidates as a fallback
        if coarse_candidates:
            self.log("warn", "Fine search found no candidates, using coarse candidates")
            # Convert coarse candidates to full format
            full_candidates = []
            for y, x, score in coarse_candidates[:self.max_candidates]:
                rx = (center_x - x) * self.mask_resolution
                ry = (y - center_y) * self.mask_resolution
                distance = math.sqrt(rx**2 + ry**2)
                depth_cm = None
                if depth_map is not None:
                    depth_m = self._calculate_point_depth((y, x), depth_map)
                    if depth_m is not None:
                        depth_cm = int(depth_m * 100.0)
                full_candidates.append((y, x, score, distance, depth_cm))
            return full_candidates
            
        # If we STILL have no candidates, fall back to standard search
        return self._standard_grid_search(combined_score, water_mask, depth_map,
                                         score_threshold, center_x, center_y, h, w,
                                         local_water_coverage, water_threshold,
                                         directionality_map, sector_restricted_mask)

    def _standard_grid_search(self, combined_score, water_mask, depth_map, 
                             score_threshold, center_x, center_y, h, w,
                             local_water_coverage=None, water_threshold=0.5,
                             directionality_map=None, sector_restricted_mask=None):
        """Perform a standard grid search when hierarchical search fails."""
        candidates = []
        grid_step = self.grid_step
        
        # Use sector-restricted mask if available
        search_mask = sector_restricted_mask if sector_restricted_mask is not None else water_mask
        
        for y in range(0, h, grid_step):
            for x in range(0, w, grid_step):
                if y >= h or x >= w:  # Safety check
                    continue
                    
                # Skip points not in the search mask
                if not search_mask[y, x]:
                    continue
                    
                # Check for sufficient water coverage if available
                if local_water_coverage is not None and local_water_coverage[y, x] < water_threshold:
                    continue
                    
                if not water_mask[y, x] or combined_score[y, x] < score_threshold:
                    continue
                    
                # Convert to robot coordinates with CORRECT AXIS MAPPING
                # CRITICAL: Flip the X-axis to match robot's coordinate system
                rx = (center_x - x) * self.mask_resolution  # Flipped X-axis: center_x - x instead of x - center_x
                ry = (y - center_y) * self.mask_resolution  # Y-axis stays the same
                
                distance = math.sqrt(rx**2 + ry**2)
                
                if self.min_distance <= distance <= self.max_distance:
                    score = combined_score[y, x]
                    
                    # Add depth-based scoring
                    if depth_map is not None:
                        depth_m = self._calculate_point_depth((y, x), depth_map)
                        if depth_m is None:
                            continue
                            
                        # Stricter depth scoring
                        depth_score = 1.0
                        if depth_m < 0.5:
                            depth_score = 0.2
                        elif depth_m < 1.0:
                            depth_score = 0.6
                        elif depth_m > 4.0:
                            depth_score = 0.3
                        elif depth_m > 3.0:
                            depth_score = 0.7
                            
                        score = score * depth_score
                        depth_cm = int(depth_m * 100.0)
                    else:
                        depth_cm = None
                    
                    # Add water coverage and directionality as additional factors in final score if available
                    water_coverage_score = local_water_coverage[y, x] if local_water_coverage is not None else 0.0
                    directionality_score = directionality_map[y, x] if directionality_map is not None else 0.0
                    
                    if directionality_map is not None and local_water_coverage is not None:
                        # Use the full enhanced scoring formula with heavy directionality weight
                        final_score = 0.3 * score + 0.3 * water_coverage_score + 0.4 * directionality_score
                    elif local_water_coverage is not None:
                        # Use just water coverage enhancement
                        final_score = 0.5 * score + 0.5 * water_coverage_score
                    else:
                        # Use base score
                        final_score = score
                        
                    candidates.append((y, x, final_score, distance, depth_cm))
                    
                    # Early termination if we find an excellent candidate
                    if final_score > 0.9 and len(candidates) >= 5:
                        candidates.sort(key=lambda x: x[2], reverse=True)
                        return candidates[:self.max_candidates]
        
        # Sort by score and return
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:self.max_candidates]

    def get_exploration_goal(self, robot_pose=None, current_time=None):
        """Get an exploration goal based on density and depth."""
        # Use current time if not provided
        if current_time is None:
            current_time = time.time()
            
        # Set computation flag
        self.is_computing = True
        self.last_computation_time = current_time
            
        # Check if we need a new goal based on timeout
        if (self.current_goal is not None and 
            current_time - self.goal_timestamp < self.goal_timeout):
            self.is_computing = False
            return self.current_goal
        
        # Get current water mask (with thread safety if lock exists)
        water_mask = None
        if self.mask_lock:
            with self.mask_lock:
                if self.water_mask is not None:
                    water_mask = self.water_mask.copy()
        else:
            if self.water_mask is not None:
                water_mask = self.water_mask.copy()
        
        # Get current depth map (with thread safety if lock exists)
        depth_map = None
        if self.depth_lock:
            with self.depth_lock:
                if self.depth_map is not None:
                    depth_map = self.depth_map.copy()
        else:
            if self.depth_map is not None:
                depth_map = self.depth_map.copy()
        
        if water_mask is None:
            self.log("warn", "No water mask available for exploration")
            self.is_computing = False
            return self._create_default_goal()
        
        try:
            start_time = time.time()
            
            # Optimization: Downsample large masks for faster processing
            h, w = water_mask.shape[:2]
            if h > 300 or w > 300:  # Reduced threshold from 400 to 300
                # Determine scale factor to get to reasonable size
                scale = min(300 / h, 300 / w)
                # Resize mask for faster processing
                analysis_mask = cv2.resize(water_mask, None, fx=scale, fy=scale, 
                                    interpolation=cv2.INTER_NEAREST)
                
                # Also resize depth map if available
                small_depth = None
                if depth_map is not None:
                    small_depth = cv2.resize(depth_map, None, fx=scale, fy=scale,
                                        interpolation=cv2.INTER_NEAREST)
                
                scale_factor = 1.0 / scale
            else:
                analysis_mask = water_mask
                small_depth = depth_map
                scale_factor = 1.0
            
            # 1. Calculate water density map
            density_map = self._calculate_density_map(analysis_mask)
            self.density_map = density_map  # Store for visualization
            
            # 2. Find candidate pixels using density map and directional information
            candidates = self._find_candidate_pixels(density_map, analysis_mask, small_depth)
            
            # 3. Scale coordinates back to original image size if needed
            if scale_factor != 1.0:
                scaled_candidates = []
                for y, x, score, distance, depth_cm in candidates:
                    # Scale coordinates back to original image size
                    orig_y = int(y * scale_factor)
                    orig_x = int(x * scale_factor) 
                    scaled_candidates.append((orig_y, orig_x, score, distance, depth_cm))
                candidates = scaled_candidates
            
            self.candidate_points = candidates  # Store for visualization
            
            # 4. Select the best candidate
            if not candidates:
                # Should never happen with our fallbacks, but just in case
                self.is_computing = False
                return self._create_default_goal()
                
            best_candidate = candidates[0]
            self.best_point = best_candidate  # Store for visualization
            
            # 5. Extract coordinates and convert to robot frame
            h, w = water_mask.shape[:2]
            center_y, center_x = h // 2, w // 2
            best_y, best_x = best_candidate[0], best_candidate[1]
            
            # Get depth at best point
            best_depth_cm = best_candidate[4] if len(best_candidate) > 4 else None
            best_depth_m = None
            if best_depth_cm is not None and best_depth_cm > 0:
                best_depth_m = best_depth_cm / 100.0
                self.log("debug", f"Selected point has depth: {best_depth_m:.2f}m")
            else:
                # Default depth if not available
                best_depth_m = 2.0
                self.log("debug", "No valid depth - using default 2.0m")
                
            # Calculate the pixel-to-meter conversion factors based on depth and FOV
            fov_rad = math.radians(self.fov_degrees)
            
            # Convert to robot coordinates using pinhole camera model
            # First calculate angular offsets
            pixel_offset_x = best_x - center_x
            pixel_offset_y = best_y - center_y
            
            # Calculate the angle per pixel based on FOV and image width
            angle_per_pixel_x = fov_rad / w
            angle_per_pixel_y = angle_per_pixel_x * (h / w)  # Adjust for aspect ratio
            
            # Calculate angular offsets in radians
            # CRITICAL: Flip the X offset to match robot's coordinate system
            angle_x_rad = -pixel_offset_x * angle_per_pixel_x  # Negative sign to flip X direction
            angle_y_rad = pixel_offset_y * angle_per_pixel_y
            
            # Convert to 3D coordinates using depth
            if best_depth_m is not None:
                # X = forward, Y = left/right, Z = up/down in robot coordinates
                # Forward distance is depth * cos(angular_offset)
                angular_offset = math.sqrt(angle_x_rad**2 + angle_y_rad**2)
                rx = best_depth_m * math.cos(angular_offset)  # Forward distance
                ry = best_depth_m * math.sin(angle_x_rad)     # Lateral offset (left/right)
                rz = best_depth_m * math.sin(angle_y_rad)     # Vertical offset (up/down)
            else:
                # Fall back to simple scaling if no depth available
                # CRITICAL: Flip the X-axis to match robot's coordinate system
                rx = (center_x - best_x) * self.mask_resolution  # Flipped X-axis
                ry = (best_y - center_y) * self.mask_resolution
                rz = 0.0
            
            # Create a goal pose
            goal = Pose()
            goal.position.x = rx
            goal.position.y = ry
            goal.position.z = 0.0  # Keep Z at zero for navigation
            
            # Set orientation towards the goal
            # Calculate orientation based on the correct coordinate system
            theta = math.atan2(ry, rx)
            goal.orientation.w = math.cos(theta/2)
            goal.orientation.z = math.sin(theta/2)
            
            # Store goal
            self.current_goal = goal
            self.goal_timestamp = current_time
            
            # No need to store score since we're not comparing with previous goals
            
            elapsed_time = time.time() - start_time
            depth_str = f", depth={best_depth_m:.2f}m" if best_depth_m else ""
            self.log("info", f"Selected exploration target at ({rx:.2f}, {ry:.2f}){depth_str} in {elapsed_time:.3f}s")
            
            # Clear computation flag
            self.is_computing = False
            return goal
            
        except Exception as e:
            self.log("error", f"Error generating exploration goal: {e}")
            self.is_computing = False
            return self._create_default_goal()
    
    def _create_default_goal(self):
        """Create a default exploration goal when no good candidates found."""
        pose = Pose()
        pose.position.x = 2.0  # Forward by 2 meters
        pose.position.y = 0.0
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        return pose
    
    def create_exploration_heatmap(self):
        """
        Create a heatmap visualization showing exploration confidence with directional awareness.
        Red areas represent high confidence, blue areas represent low confidence.
        """
        if (self.water_mask is None or self.combined_map is None):
            return None

        try:
            # Ensure the water mask and combined map have the same dimensions
            h, w = self.water_mask.shape[:2]
            
            # Get the combined map if available, otherwise use density
            vis_map = self.combined_map
            if vis_map is None:
                vis_map = self.density_map
                
            if vis_map.shape[:2] != (h, w):
                vis_map = cv2.resize(vis_map, (w, h), interpolation=cv2.INTER_NEAREST)

            # Create a base canvas with zeroes (black background)
            heatmap = np.zeros((h, w, 3), dtype=np.uint8)

            # Use combined map directly for visualization
            water_area = self.water_mask > 0

            # Scale density map to 0-255 range
            norm_vis = (vis_map * 255).astype(np.uint8)

            # Map confidence to red and blue channels - no flipping needed
            red_channel = norm_vis.copy()  # High confidence -> Red
            blue_channel = 255 - norm_vis  # Low confidence -> Blue
            green_channel = np.zeros_like(norm_vis)

            # Apply to water areas only
            red_channel[~water_area] = 0
            blue_channel[~water_area] = 0
            green_channel[~water_area] = 0

            # Create RGB visualization
            heatmap[:, :, 0] = blue_channel
            heatmap[:, :, 1] = green_channel
            heatmap[:, :, 2] = red_channel

            # Draw a vertical line at the center
            center_x = w // 2
            cv2.line(heatmap, (center_x, 0), (center_x, h), (255, 255, 255), 1)

            # Draw left/right labels - Note: From robot's perspective
            cv2.putText(heatmap, "RIGHT", (20, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(heatmap, "LEFT", (w - 100, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                       
            # Show the left and right coverage percentages
            left_cov = getattr(self, 'left_coverage', 0.0)
            right_cov = getattr(self, 'right_coverage', 0.0)
            
            left_text = f"L: {left_cov*100:.1f}%"
            right_text = f"R: {right_cov*100:.1f}%"
            
            cv2.putText(heatmap, right_text, (20, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(heatmap, left_text, (w - 100, 40), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw best point if available
            if self.best_point:
                y, x = self.best_point[0], self.best_point[1]
                cv2.circle(heatmap, (x, y), 8, (0, 255, 0), -1)  # Green circle

            # Draw robot position at center
            center_y = h // 2
            cv2.circle(heatmap, (center_x, center_y), 6, (0, 165, 255), -1)  # Orange circle
            
            # Optionally overlay directional information (showing potential exits)
            try:
                # Generate directional flow indicators using the points in candidate_points
                if len(self.candidate_points) > 5:
                    # Get the top candidates
                    top_candidates = sorted(self.candidate_points, key=lambda x: x[2], reverse=True)[:10]
                    
                    for y, x, score, _, _ in top_candidates:
                        if score < 0.5:  # Skip low-scoring candidates
                            continue
                            
                        # Draw small arrows pointing in promising directions
                        # Calculate direction from center
                        dx = x - center_x
                        dy = y - center_y
                        
                        # Skip points too close to center
                        if abs(dx) < 20 and abs(dy) < 20:
                            continue
                            
                        # Normalize direction vector
                        magnitude = math.sqrt(dx**2 + dy**2)
                        if magnitude > 0:
                            dx = dx / magnitude * 20  # Scale to 20 pixels
                            dy = dy / magnitude * 20
                            
                            # Draw arrow from point back toward center
                            arrow_start = (int(x), int(y))
                            arrow_end = (int(x - dx), int(y - dy))
                            
                            # Arrow color based on score (higher score = more yellow)
                            arrow_color = (0, int(255 * score), 255)
                            
                            cv2.arrowedLine(heatmap, arrow_start, arrow_end, arrow_color, 2, tipLength=0.3)
            except Exception as e:
                self.log("debug", f"Failed to draw directional indicators: {e}")

            return heatmap
        except Exception as e:
            self.log("error", f"Error creating heatmap: {e}")
            return None
    
    def check_computation_timeout(self):
        """Check if current computation has exceeded timeout."""
        if not self.is_computing:
            return False
            
        current_time = time.time()
        if current_time - self.last_computation_time > self.computation_timeout:
            self.log("warn", f"Exploration computation timeout after {current_time - self.last_computation_time:.1f}s")
            self.is_computing = False
            return True
            
        return False

    def publish_exploration_visualization(self):
        """Publish visualization of the depth-aware water exploration system."""
        if not self.enable_water_explorer:
            return

        try:
            # Create marker array for visualization
            marker_array = MarkerArray()

            # Visualize candidate points with depth information
            self._visualize_candidate_points(marker_array, self.candidate_points, self.best_point)

            # Publish marker array if not empty
            if hasattr(self, 'exploration_viz_pub') and marker_array.markers:
                self.exploration_viz_pub.publish(marker_array)

            # Publish the heatmap visualization if available
            heatmap = self.create_exploration_heatmap()
            if heatmap is not None and hasattr(self, 'heatmap_pub'):
                try:
                    heatmap_msg = self.bridge.cv2_to_imgmsg(heatmap, "bgr8")
                    heatmap_msg.header.stamp = self.clock.now().to_msg() if self.clock else None
                    heatmap_msg.header.frame_id = self.frame_id
                    self.heatmap_pub.publish(heatmap_msg)
                except Exception as e:
                    self.log("error", f"Error publishing heatmap: {e}")

        except Exception as e:
            self.log("error", f"Error in exploration visualization: {e}")

    def _visualize_candidate_points(self, marker_array, candidate_points, best_point):
        """Visualize exploration candidate points with depth information."""
        if not candidate_points:
            return

        try:
            # Add markers for candidate points
            for i, point_data in enumerate(candidate_points):
                y, x, score, distance, depth_cm = point_data
                depth_m = depth_cm / 100.0 if depth_cm else None

                # Convert to robot coordinates
                h, w = self.water_mask.shape[:2]
                center_y, center_x = h // 2, w // 2
                rx = (center_x - x) * self.mask_resolution  # Flipped X-axis
                ry = (y - center_y) * self.mask_resolution

                # Create marker
                marker = Marker()
                marker.header.frame_id = self.frame_id
                marker.header.stamp = self.clock.now().to_msg() if self.clock else None
                marker.ns = "candidate_points"
                marker.id = i
                marker.type = Marker.SPHERE
                marker.action = Marker.ADD
                marker.pose.position.x = rx
                marker.pose.position.y = ry
                marker.pose.position.z = 0.05
                marker.pose.orientation.w = 1.0
                marker.scale.x = marker.scale.y = marker.scale.z = 0.1 + 0.1 * score
                marker.color.r = float(1.0 - score)  # Ensure float type
                marker.color.g = float(score)        # Ensure float type
                marker.color.b = 0.0
                marker.color.a = 0.7
                marker_array.markers.append(marker)

            # Highlight best point
            if best_point:
                y, x, score, distance, depth_cm = best_point
                h, w = self.water_mask.shape[:2]
                center_y, center_x = h // 2, w // 2
                rx = (center_x - x) * self.mask_resolution  # Flipped X-axis
                ry = (y - center_y) * self.mask_resolution
                
                best_marker = Marker()
                best_marker.header.frame_id = self.frame_id
                best_marker.header.stamp = self.clock.now().to_msg() if self.clock else None
                best_marker.ns = "best_point"
                best_marker.id = 0
                best_marker.type = Marker.SPHERE
                best_marker.action = Marker.ADD
                best_marker.pose.position.x = rx
                best_marker.pose.position.y = ry
                best_marker.pose.position.z = 0.1
                best_marker.pose.orientation.w = 1.0
                best_marker.scale.x = best_marker.scale.y = best_marker.scale.z = 0.3
                best_marker.color.r = 1.0
                best_marker.color.g = 0.0
                best_marker.color.b = 0.0
                best_marker.color.a = 0.9
                marker_array.markers.append(best_marker)

        except Exception as e:
            self.log("error", f"Error visualizing candidate points: {e}")