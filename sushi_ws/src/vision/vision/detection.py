#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
import sys
import math
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision.detection")

class SlicePrediction:
    def __init__(self, bbox: Tuple[float, float, float, float], score: float, class_id: int):
        self.bbox = bbox  # [x1, y1, x2, y2] format
        self.score = score
        self.class_id = class_id

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node')
        
        # Declare parameters individually
        self.declare_parameter('source_topic', '/sim_injection')
        self.declare_parameter('detection_confidence', 0.3)
        self.declare_parameter('detection_iou_threshold', 0.5)
        self.declare_parameter('max_detections', 100)
        self.declare_parameter('use_sahi', False)
        self.declare_parameter('sahi_num_slices_width', 1)
        self.declare_parameter('sahi_num_slices_height', 1)
        self.declare_parameter('sahi_overlap_ratio', 0.3)
        self.declare_parameter('model_path', '')
        self.declare_parameter('size_threshold', 5)
        
        # Get parameters
        self.source_topic = self.get_parameter('source_topic').get_parameter_value().string_value
        self.detection_confidence = self.get_parameter('detection_confidence').get_parameter_value().double_value
        self.detection_iou_threshold = self.get_parameter('detection_iou_threshold').get_parameter_value().double_value
        self.max_detections = self.get_parameter('max_detections').get_parameter_value().integer_value
        self.use_sahi = self.get_parameter('use_sahi').get_parameter_value().bool_value
        self.sahi_num_slices_width = self.get_parameter('sahi_num_slices_width').get_parameter_value().integer_value
        self.sahi_num_slices_height = self.get_parameter('sahi_num_slices_height').get_parameter_value().integer_value
        self.sahi_overlap_ratio = self.get_parameter('sahi_overlap_ratio').get_parameter_value().double_value
        self.model_path_param = self.get_parameter('model_path').get_parameter_value().string_value
        self.size_threshold = self.get_parameter('size_threshold').get_parameter_value().integer_value
        
        # Define class names for our model - FIXED MAPPING
        # Class 0 = obstacles, class 1 = goals/trash
        self.class_names = {
            0: "Obstacle",
            1: "Goal"
        }
        
        self.bridge = CvBridge()
        
        # Create publishers before loading models (to avoid context issues)
        self.detection_pub = self.create_publisher(Image, 'detection_image', 10)
        self.detections_pub = self.create_publisher(PoseArray, 'raw_detections', 10)
        
        # Subscribers
        self.image_sub = self.create_subscription(Image, self.source_topic, self.image_callback, 10)
        
        # Load model after setting up publishers and subscribers
        self.load_model()
        
        self.get_logger().info('Detection node initialized')
    
    def load_model(self):
        try:
            # Look for models in standard locations
            model_paths = [
                'models/debris-det-toast2.engine',
                'src/vision/models/debris-det-toast2.engine',
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/debris-det-toast2.engine')
            ]
            
            # Add the path from parameters if provided
            if self.model_path_param and os.path.exists(os.path.join(self.model_path_param, 'debris-det-toast2.engine')):
                model_paths.insert(0, os.path.join(self.model_path_param, 'debris-det-toast2.engine'))
            
            model_loaded = False
            for path in model_paths:
                if os.path.exists(path):
                    try:
                        from ultralytics import YOLO
                        self.model = YOLO(path, task='detect')
                        self.get_logger().info(f"YOLO model loaded successfully from {path}")
                        model_loaded = True
                        break
                    except Exception as e:
                        self.get_logger().error(f"Failed to load model from {path}: {e}")
            
            if not model_loaded:
                self.get_logger().error(f"Model not found in any of these locations: {model_paths}")
                # For testing, we'll continue without a model
                self.model = None
                
        except Exception as e:
            self.get_logger().error(f"Failed to load model: {e}")
            self.model = None
    
    def get_slice_bboxes(self, image_shape, slice_height=256, slice_width=256, overlap_ratio=0.2):
        """
        Generate slice bounding boxes for the image with specified overlap.
        """
        height, width = image_shape[:2]
        
        # Calculate stride (distance between slice starts)
        stride_h = int(slice_height * (1 - overlap_ratio))
        stride_w = int(slice_width * (1 - overlap_ratio))
        
        slices = []
        for y in range(0, height, stride_h):
            for x in range(0, width, stride_w):
                x2 = min(x + slice_width, width)
                y2 = min(y + slice_height, height)
                # Adjust coordinates if at the border
                x1 = max(0, x2 - slice_width)
                y1 = max(0, y2 - slice_height)
                slices.append([x1, y1, x2, y2])
        
        return slices

    def adjust_predictions(self, predictions, slice_bbox):
        """
        Adjust prediction coordinates based on slice position in original image.
        """
        x_offset, y_offset = slice_bbox[0], slice_bbox[1]
        adjusted_predictions = []
        
        for pred in predictions:
            x1, y1, x2, y2 = pred['bbox']
            adjusted_bbox = (
                x1 + x_offset,
                y1 + y_offset,
                x2 + x_offset,
                y2 + y_offset
            )
            adjusted_pred = {
                'bbox': adjusted_bbox,
                'score': pred['score'],
                'class_id': pred['class_id']
            }
            adjusted_predictions.append(adjusted_pred)
        
        return adjusted_predictions

    def compute_iou(self, boxA, boxB):
        """
        Compute Intersection over Union between two bounding boxes.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        
        return interArea / float(boxAArea + boxBArea - interArea) if interArea > 0 else 0.0

    def non_max_suppression(self, predictions, iou_threshold=0.5):
        """
        Apply Non-Maximum Suppression to remove overlapping predictions.
        """
        if not predictions:
            return []
        
        # Sort by confidence score (highest first)
        sorted_preds = sorted(predictions, key=lambda x: x['score'], reverse=True)
        
        keep = []
        
        while sorted_preds:
            # Take the detection with highest confidence
            current = sorted_preds.pop(0)
            keep.append(current)
            
            # Filter remaining detections
            remaining = []
            for pred in sorted_preds:
                # Calculate IoU
                iou = self.compute_iou(current['bbox'], pred['bbox'])
                # Keep predictions with IoU below threshold
                if iou <= iou_threshold:
                    remaining.append(pred)
                    
            sorted_preds = remaining
        
        return keep

    def process_slice(self, slice_img, slice_bbox):
        """Process a single slice of the image."""
        try:
            # Run detection on the slice
            results = self.model.predict(
                source=slice_img,
                conf=self.detection_confidence,
                iou=self.detection_iou_threshold
            )
            
            # Extract detections
            slice_predictions = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    scores = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy()
                    
                    for box, score, class_id in zip(boxes, scores, class_ids):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Apply size threshold
                        width = x2 - x1
                        height = y2 - y1
                        if width >= self.size_threshold and height >= self.size_threshold:
                            slice_predictions.append({
                                'bbox': (x1, y1, x2, y2),
                                'score': float(score),
                                'class_id': int(class_id)
                            })
            
            # Adjust coordinates to original image space
            adjusted_predictions = self.adjust_predictions(slice_predictions, slice_bbox)
            return adjusted_predictions
            
        except Exception as e:
            self.get_logger().error(f"Error processing slice: {e}")
            return []

    def process_with_sahi(self, frame):
        """Process frame using SAHI slicing approach."""
        try:
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate slice dimensions
            slice_width = frame_width // self.sahi_num_slices_width
            slice_height = frame_height // self.sahi_num_slices_height
            
            # Get slices
            slice_bboxes = self.get_slice_bboxes(
                frame.shape, 
                slice_height=slice_height,
                slice_width=slice_width,
                overlap_ratio=self.sahi_overlap_ratio
            )
            
            all_predictions = []
            
            # Process each slice
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = []
                
                for slice_bbox in slice_bboxes:
                    x1, y1, x2, y2 = slice_bbox
                    slice_img = frame[y1:y2, x1:x2].copy()
                    futures.append(executor.submit(self.process_slice, slice_img, slice_bbox))
                
                # Collect results
                for future in futures:
                    predictions = future.result()
                    all_predictions.extend(predictions)
            
            # Apply non-max suppression to combined predictions
            final_predictions = self.non_max_suppression(all_predictions, self.detection_iou_threshold)
            
            return final_predictions
            
        except Exception as e:
            self.get_logger().error(f"Error in SAHI processing: {e}")
            return []

    def image_callback(self, msg):
        """
        • Converts the ROS Image → OpenCV BGR frame
        • Runs YOLO (or SAHI-sliced YOLO) inference
        • Filters by confidence, IoU-NMS, and size_threshold
        • Publishes:
            – /detection_image   (annotated RGB)
            – /raw_detections    (PoseArray; empty if none)
        """
        try:
            # ---------- Safety checks ----------
            if self.model is None:
                self.get_logger().warn("DetectionNode: model not loaded – skipping frame")
                return

            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            # ---------- Inference ----------
            if self.use_sahi:
                self.get_logger().debug("DetectionNode: SAHI slicing enabled")
                detections = self.process_with_sahi(frame)
            else:
                self.get_logger().debug("DetectionNode: standard YOLO inference")
                results = self.model.predict(
                    source=frame,
                    conf=self.detection_confidence,
                    iou=self.detection_iou_threshold,
                    max_det=self.max_detections
                )

                detections = []
                for res in results:
                    if res.boxes is None or len(res.boxes) == 0:
                        continue
                    boxes     = res.boxes.xyxy.cpu().numpy()
                    scores    = res.boxes.conf.cpu().numpy()
                    class_ids = res.boxes.cls.cpu().numpy()

                    for box, score, cid in zip(boxes, scores, class_ids):
                        x1, y1, x2, y2 = map(int, box)
                        w, h = x2 - x1, y2 - y1
                        if w >= self.size_threshold and h >= self.size_threshold:
                            detections.append(
                                {'bbox': (x1, y1, x2, y2),
                                 'score': float(score),
                                 'class_id': int(cid)}
                            )

            # ---------- Logging ----------
            goal_cnt      = sum(1 for d in detections if d['class_id'] == 1)
            obstacle_cnt  = sum(1 for d in detections if d['class_id'] == 0)
            self.get_logger().info(
                f"DetectionNode: {goal_cnt} goals, {obstacle_cnt} obstacles (from {len(detections)} total boxes)"
            )

            # ---------- Visualisation ----------
            vis = frame.copy()
            for d in detections:
                (x1, y1, x2, y2), score, cid = d['bbox'], d['score'], d['class_id']
                if cid == 1:  # Goal / trash
                    color, label = (0, 255, 0), "Goal"
                else:         # Obstacle
                    color, label = (0, 0, 255), "Obstacle"

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    vis, f"{label}:{score:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
                )

            vis_msg = self.bridge.cv2_to_imgmsg(vis, "bgr8")
            vis_msg.header = msg.header
            self.detection_pub.publish(vis_msg)

            # ---------- PoseArray publishing ----------
            pose_array = PoseArray()
            pose_array.header = msg.header   # always set header

            for d in detections:
                x1, y1, x2, y2 = d['bbox']
                pose = Pose()
                pose.position.x = float(x1)
                pose.position.y = float(y1)
                pose.position.z = float(x2 - x1)           # width
                pose.orientation.x = float(y2 - y1)        # height
                pose.orientation.y = float(d['score'])     # confidence
                pose.orientation.z = float(d['class_id'])  # class id
                pose_array.poses.append(pose)

            if not pose_array.poses:
                self.get_logger().debug("DetectionNode: no valid detections – publishing EMPTY PoseArray")

            self.detections_pub.publish(pose_array)

        except Exception as exc:
            self.get_logger().error(f"DetectionNode: exception in image_callback → {exc}", exc_info=True)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down detection node')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
