#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
import numpy as np
import math
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from concurrent.futures import ThreadPoolExecutor

class VisualTracker:
    def __init__(self):
        # Initialize ORB detector with optimized params
        self.detector = cv2.ORB_create(
            nfeatures=2000,  # More features for better tracking
            scaleFactor=1.1,  # Smaller scale factor for more accuracy
            nlevels=8,
            edgeThreshold=15,  # More sensitive to edges
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=21,  # Smaller patch for faster processing
            fastThreshold=10  # More sensitive feature detection
        )
        
        # Use FLANN matcher for faster matching
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                          table_number=6,
                          key_size=12,
                          multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        self.prev_frame = None
        self.prev_kp = None
        self.prev_desc = None
        self.prev_depth = None
        
        # Motion prediction
        self.last_motion = None
        self.motion_filter = np.zeros(3)  # [dx, dy, dyaw]
        self.alpha = 0.3  # Motion smoothing factor
        
        # Camera parameters for OAK-D
        self.fx = 860.0
        self.fy = 860.0
        self.cx = 640.0
        self.cy = 360.0
        self.camera_matrix = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
        self.current_features = None

    def detect_features_async(self, frame):
        kp = self.detector.detect(frame, None)
        kp, desc = self.detector.compute(frame, kp)
        return kp, desc

    def get_depth_value(self, depth_map, x, y):
        """Get depth value in meters from depth map"""
        if depth_map.dtype == np.uint16:
            return float(depth_map[y, x]) / 1000.0  # uint16 to meters
        else:
            # Assuming BGR8/grayscale format (0-255)
            return (float(depth_map[y, x]) / 255.0) * 4.5 + 0.5

    def process_frame(self, frame, depth):
        try:
            if frame is None or depth is None:
                return None, None, None

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis_img = frame.copy()

            # Async feature detection
            future = self.thread_pool.submit(self.detect_features_async, gray)
            kp, desc = future.result()

            if len(kp) == 0:
                self.prev_frame = gray
                self.prev_kp = None
                self.prev_desc = None
                self.prev_depth = depth
                self.current_features = vis_img
                return None, None, None

            # Handle first frame
            if self.prev_frame is None or self.prev_kp is None or self.prev_desc is None:
                self.prev_frame = gray
                self.prev_kp = kp
                self.prev_desc = desc
                self.prev_depth = depth
                # Draw features
                for k in kp:
                    pt = tuple(map(int, k.pt))
                    cv2.circle(vis_img, pt, 3, (0, 255, 0), -1)
                self.current_features = vis_img
                return None, None, None

            # Match features
            try:
                matches = self.matcher.knnMatch(self.prev_desc, desc, k=2)
                good_matches = []
                for match_tuple in matches:
                    if len(match_tuple) == 2:
                        m, n = match_tuple
                        if m.distance < 0.7 * n.distance:  # Lowe's ratio test
                            good_matches.append(m)
            except Exception as e:
                print(f"Matching error: {e}")
                good_matches = []

            if len(good_matches) < 10:
                self.prev_frame = gray
                self.prev_kp = kp
                self.prev_desc = desc
                self.prev_depth = depth
                return None, None, None

            matches = sorted(good_matches, key=lambda x: x.distance)
            matches = matches[:min(100, len(matches))]  # Keep more good matches

            # Collect valid point pairs
            valid_pts1 = []  # Previous frame points
            valid_pts2 = []  # Current frame points
            valid_pts3d = [] # 3D points from previous frame

            for match in matches:
                # Get 2D points
                prev_pt = self.prev_kp[match.queryIdx].pt
                curr_pt = kp[match.trainIdx].pt
                
                # Get depth for previous point
                x, y = int(prev_pt[0]), int(prev_pt[1])
                if 0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]:
                    z = self.get_depth_value(depth, x, y)
                    if 0.1 < z < 5.0:  # Filter unrealistic depths
                        # Add 3D point
                        pt3d = [
                            (x - self.cx) * z / self.fx,
                            (y - self.cy) * z / self.fy,
                            z
                        ]
                        valid_pts3d.append(pt3d)
                        valid_pts2.append(curr_pt)
                        valid_pts1.append(prev_pt)
                        
                        # Draw the match
                        prev_pt_int = tuple(map(int, prev_pt))
                        curr_pt_int = tuple(map(int, curr_pt))
                        cv2.circle(vis_img, prev_pt_int, 3, (0, 0, 255), -1)
                        cv2.circle(vis_img, curr_pt_int, 3, (0, 255, 0), -1)
                        cv2.line(vis_img, prev_pt_int, curr_pt_int, (255, 0, 0), 1)

            self.current_features = vis_img

            if len(valid_pts3d) < 8:
                self.prev_frame = gray
                self.prev_kp = kp
                self.prev_desc = desc
                self.prev_depth = depth
                return None, None, None

            # Convert to numpy arrays
            pts3d = np.array(valid_pts3d, dtype=np.float32)
            pts2d = np.array(valid_pts2, dtype=np.float32)

            # Estimate pose with better parameters
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d,
                pts2d,
                self.camera_matrix,
                None,
                confidence=0.95,
                reprojectionError=2.0,
                iterationsCount=150,
                flags=cv2.SOLVEPNP_EPNP
            )

            if not success or (inliers is not None and len(inliers) < 8):
                self.prev_frame = gray
                self.prev_kp = kp
                self.prev_desc = desc
                self.prev_depth = depth
                return None, None, None

            # Convert rotation vector to matrix
            R, _ = cv2.Rodrigues(rvec)

            # Extract Euler angles using proper order
            pitch = math.atan2(-R[2,1], R[2,2])
            cos_pitch = math.sqrt(R[2,1]**2 + R[2,2]**2)
            roll = math.atan2(R[1,0], R[0,0])
            yaw = math.atan2(-R[2,0], cos_pitch)

            # Get motion in camera frame
            transformed_tvec = np.dot(R.T, tvec).flatten()

            # Handle rotations
            roll_threshold = math.radians(5)
            pitch_threshold = math.radians(5)
            
            if abs(roll) > roll_threshold or abs(pitch) > pitch_threshold:
                yaw = 0.0
                cos_correction = math.cos(roll) * math.cos(pitch)
                transformed_tvec[0] *= cos_correction
                transformed_tvec[1] *= cos_correction

            # Calculate confidence
            confidence = len(inliers) / len(pts3d) if inliers is not None else 0

            # Motion prediction
            dx, dy, final_yaw = transformed_tvec[0], transformed_tvec[1], yaw
            
            if self.last_motion is not None and confidence > 0.5:
                predicted_motion = self.last_motion
                current_motion = np.array([dx, dy, final_yaw])
                filtered_motion = self.alpha * current_motion + (1 - self.alpha) * predicted_motion
                dx, dy, final_yaw = filtered_motion

            # Visualization
            rot_text = (f"Roll: {math.degrees(roll):.1f}, "
                       f"Pitch: {math.degrees(pitch):.1f}, "
                       f"Yaw: {math.degrees(final_yaw):.1f}")
            cv2.putText(self.current_features, rot_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            trans_text = f"dx: {dx:.3f}, dy: {dy:.3f}, conf: {confidence:.2f}"
            cv2.putText(self.current_features, trans_text, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Update previous frame data
            self.prev_frame = gray
            self.prev_kp = kp
            self.prev_desc = desc
            self.prev_depth = depth

            # Store motion for next frame
            if confidence > 0.5:
                self.last_motion = np.array([dx, dy, final_yaw])
                return float(dx), float(dy), float(final_yaw)
            
            return None, None, None

        except Exception as e:
            print(f"Error in visual tracking: {e}")
            return None, None, None

class VIONode(Node):
    def __init__(self):
        super().__init__('vio_node')
        
        # Declare parameters
        self.declare_parameter('rgb_topic', '/cam/rectified/gray')
        self.declare_parameter('depth_topic', '/toast/depth')
        self.declare_parameter('processing_rate', 60.0)
        self.declare_parameter('velocity_alpha', 0.7)
        
        # Get parameters
        rgb_topic = self.get_parameter('rgb_topic').value
        depth_topic = self.get_parameter('depth_topic').value
        
        self.visual_tracker = VisualTracker()
        self.position = np.zeros(3)
        self.orientation = 0.0
        self.bridge = CvBridge()
        
        # Subscribe to RGB and depth topics using parameters
        self.rgb_sub = self.create_subscription(
            Image,
            rgb_topic,
            self.rgb_callback,
            10
        )
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10
        )
        
        # Store latest frames
        self.latest_rgb = None
        self.latest_depth = None
        self.frame_count = 0  # For tracking processing rate
        self.last_frame_time = self.get_clock().now()
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, '/oak/vio/odometry', 10)
        self.path_pub = self.create_publisher(Path, '/oak/vio/path', 10)
        self.features_pub = self.create_publisher(Image, '/oak/vio/features', 10)
        
        self.path = Path()
        self.path.header.frame_id = 'odom'
        
        # Create timer for higher frequency processing
        self.create_timer(1/60.0, self.process_frame)  # 60Hz processing
        
        # Motion smoothing
        self.velocity = np.zeros(3)  # [vx, vy, vyaw]
        self.velocity_alpha = 0.7  # Velocity smoothing factor
        
        self.get_logger().info('Visual Odometry Node initialized')

    def rgb_callback(self, msg):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f'Error in RGB callback: {str(e)}')

    def depth_callback(self, msg):
        try:
            depth_data = self.bridge.imgmsg_to_cv2(msg)
            
            # Handle different depth formats
            if len(depth_data.shape) == 3:  # BGR format
                self.latest_depth = cv2.cvtColor(depth_data, cv2.COLOR_BGR2GRAY)
            else:  # uint16 format
                self.latest_depth = depth_data
                
        except Exception as e:
            self.get_logger().error(f'Error in Depth callback: {str(e)}')

    def process_frame(self):
        try:
            if self.latest_rgb is not None and self.latest_depth is not None:
                # Calculate processing rate
                current_time = self.get_clock().now()
                self.frame_count += 1
                if self.frame_count % 30 == 0:  # Print rate every 30 frames
                    dt = (current_time.nanoseconds - self.last_frame_time.nanoseconds) / 1e9
                    fps = 30 / dt
                    self.get_logger().info(f'Processing rate: {fps:.1f} FPS')
                    self.last_frame_time = current_time

                # Process visual data
                dx, dy, dyaw = self.visual_tracker.process_frame(
                    self.latest_rgb.copy(), 
                    self.latest_depth.copy()
                )
                
                # Publish feature visualization
                if self.visual_tracker.current_features is not None:
                    try:
                        features_msg = self.bridge.cv2_to_imgmsg(
                            self.visual_tracker.current_features, 
                            encoding="bgr8"
                        )
                        features_msg.header.stamp = self.get_clock().now().to_msg()
                        features_msg.header.frame_id = "oak_camera_frame"
                        self.features_pub.publish(features_msg)
                    except Exception as e:
                        self.get_logger().warning(f'Error publishing features: {str(e)}')
                
 # Update pose if motion was detected
                if dx is not None:
                    # Smooth velocity estimates
                    current_vel = np.array([dx, dy, dyaw])
                    self.velocity = (self.velocity_alpha * current_vel + 
                                   (1 - self.velocity_alpha) * self.velocity)
                    
                    # Update position and orientation
                    dt = 1/60.0  # Assuming 60Hz processing
                    self.position[0] += self.velocity[0]
                    self.position[1] += self.velocity[1]
                    self.orientation += self.velocity[2]
                    
                    # Normalize orientation to [-pi, pi]
                    self.orientation = math.atan2(math.sin(self.orientation), 
                                                math.cos(self.orientation))
                    
                    self.publish_odometry()
                    
        except Exception as e:
            self.get_logger().error(f'Error in process_frame: {str(e)}')

    def publish_odometry(self):
        current_time = self.get_clock().now()
        
        # Create odometry message
        odom_msg = Odometry()
        odom_msg.header.stamp = current_time.to_msg()
        odom_msg.header.frame_id = 'odom'
        odom_msg.child_frame_id = 'base_link'
        
        # Set position
        odom_msg.pose.pose.position.x = float(self.position[0])
        odom_msg.pose.pose.position.y = float(self.position[1])
        odom_msg.pose.pose.position.z = 0.0
        
        # Set orientation
        odom_msg.pose.pose.orientation.x = 0.0
        odom_msg.pose.pose.orientation.y = 0.0
        odom_msg.pose.pose.orientation.z = math.sin(self.orientation/2)
        odom_msg.pose.pose.orientation.w = math.cos(self.orientation/2)
        
        # Set velocities
        odom_msg.twist.twist.linear.x = float(self.velocity[0])
        odom_msg.twist.twist.linear.y = float(self.velocity[1])
        odom_msg.twist.twist.angular.z = float(self.velocity[2])
        
        # Add covariance
        position_cov = 0.1
        orientation_cov = 0.1
        velocity_cov = 0.1
        
        # Position covariance [x, y, z, roll, pitch, yaw]
        odom_msg.pose.covariance[0] = position_cov    # x
        odom_msg.pose.covariance[7] = position_cov    # y
        odom_msg.pose.covariance[14] = position_cov   # z
        odom_msg.pose.covariance[21] = orientation_cov # roll
        odom_msg.pose.covariance[28] = orientation_cov # pitch
        odom_msg.pose.covariance[35] = orientation_cov # yaw
        
        # Velocity covariance [vx, vy, vz, vroll, vpitch, vyaw]
        odom_msg.twist.covariance[0] = velocity_cov   # vx
        odom_msg.twist.covariance[7] = velocity_cov   # vy
        odom_msg.twist.covariance[14] = velocity_cov  # vz
        odom_msg.twist.covariance[21] = velocity_cov  # vroll
        odom_msg.twist.covariance[28] = velocity_cov  # vpitch
        odom_msg.twist.covariance[35] = velocity_cov  # vyaw
        
        # Publish odometry
        self.odom_pub.publish(odom_msg)
        
        # Update and publish path
        pose_stamped = PoseStamped()
        pose_stamped.header = odom_msg.header
        pose_stamped.pose = odom_msg.pose.pose
        
        self.path.header.stamp = current_time.to_msg()
        self.path.poses.append(pose_stamped)
        
        if len(self.path.poses) > 1000:
            self.path.poses.pop(0)
            
        self.path_pub.publish(self.path)

    def __del__(self):
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = VIONode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down VIO node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
