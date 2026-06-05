import os
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from pathlib import Path
import av
import logging
import warnings
import time
from datetime import datetime
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from contextlib import nullcontext
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('root').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)


################################################################################
# MASK-TO-BOX AND BOX-TO-YOLO HELPERS
################################################################################
def mask_to_bboxes(mask, min_area=50):
    """
    mask : (H,W) binary numpy array
    returns list[[x1,y1,x2,y2]]  (absolute pixels)
    """
    import cv2, numpy as np
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bbs = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h >= min_area:
            bbs.append([x, y, x + w, y + h])
    return bbs

def bboxes_to_yolo(bbs, w, h, cls_ids=0):
    """
    bbs : [[x1,y1,x2,y2], ...]  absolute
    cls_ids : int or list of class IDs
    returns list[str] 'cls cx cy bw bh'
    """
    yolo = []
    if isinstance(cls_ids, (list, tuple, np.ndarray)):
        pairs = zip(bbs, cls_ids)
    else:
        pairs = ((bb, cls_ids) for bb in bbs)
    for (x1, y1, x2, y2), cid in pairs:
        cx = (x1 + x2) / 2 / w
        cy = (y1 + y2) / 2 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        yolo.append(f"{int(cid)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return yolo


################################################################################
# TRACKERS FOR OBJECT TRACKING
################################################################################
class CSRTObjectTracker:
    def __init__(self):
        self.trackers = {}  # Dictionary to store {track_id: tracker}
        self.boxes = {}     # Dictionary to store {track_id: box}
        self.classes = {}   # Dictionary to store {track_id: class_id}
        self.next_id = 0    # Counter for assigning unique track IDs
        self.tracked_frame = None  # Keep reference to the last tracked frame
        
    def create_tracker(self):
        """Create a CSRT tracker"""
        return cv2.legacy.TrackerCSRT_create()
    
    def initialize_trackers(self, frame, boxes, class_ids):
        """Initialize trackers with bounding boxes from the current frame"""
        if not boxes:
            logger.warning("No boxes to track")
            return
            
        # Convert PIL image to OpenCV format
        if isinstance(frame, Image.Image):
            frame_np = np.array(frame)
            frame_cv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_cv = frame
            
        self.tracked_frame = frame_cv.copy()
        
        # Clear existing trackers
        self.trackers = {}
        self.boxes = {}
        self.classes = {}
        self.next_id = 0
        
        # Initialize new trackers for each box
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            tracker = self.create_tracker()
            x1, y1, x2, y2 = box
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            
            # Ensure box dimensions are valid
            if w <= 0 or h <= 0:
                logger.warning(f"Invalid box dimensions: w={w}, h={h}, skipping")
                continue
                
            track_id = self.next_id
            
            try:
                success = tracker.init(frame_cv, (x, y, w, h))
                
                if success:
                    self.trackers[track_id] = tracker
                    self.boxes[track_id] = box
                    self.classes[track_id] = class_id
                    self.next_id += 1
                else:
                    logger.warning(f"Failed to initialize tracker for box {i}")
            except Exception as e:
                logger.error(f"Error initializing tracker: {e}")
        
        logger.info(f"Initialized {len(self.trackers)} trackers")
    
    def update_trackers(self, frame):
        """Update all trackers with the new frame and return the tracked boxes and classes"""
        if not self.trackers:
            logger.warning("No trackers to update")
            return [], []
            
        # Convert PIL image to OpenCV format
        if isinstance(frame, Image.Image):
            frame_np = np.array(frame)
            frame_cv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            frame_cv = frame
            
        self.tracked_frame = frame_cv.copy()
        frame_height, frame_width = frame_cv.shape[:2]
        
        updated_boxes = []
        updated_classes = []
        failed_track_ids = []
        
        # Update each tracker
        for track_id, tracker in self.trackers.items():
            try:
                success, bbox = tracker.update(frame_cv)
                
                if success:
                    x, y, w, h = [int(v) for v in bbox]
                    
                    # Add stricter criteria for what constitutes a "lost" object
                    # 1. Box dimensions must be reasonable
                    # 2. Box must be mostly inside the frame
                    # 3. Box must not have moved too far from previous position
                    
                    # Check if box dimensions are reasonable (not too small or too large)
                    if w < 3 or h < 3 or w > frame_width*0.9 or h > frame_height*0.9:
                        logger.warning(f"Tracker {track_id} returned unreasonable box size: {w}x{h}, marking as failed")
                        failed_track_ids.append(track_id)
                        continue
                    
                    # Check if box is mostly inside the frame
                    if x < -w/2 or y < -h/2 or x+w/2 > frame_width or y+h/2 > frame_height:
                        logger.warning(f"Tracker {track_id} returned box mostly outside frame, marking as failed")
                        failed_track_ids.append(track_id)
                        continue
                    
                    # Check if box has moved too far from previous position (if we know it)
                    if track_id in self.boxes:
                        prev_x1, prev_y1, prev_x2, prev_y2 = self.boxes[track_id]
                        prev_cx = (prev_x1 + prev_x2) / 2
                        prev_cy = (prev_y1 + prev_y2) / 2
                        new_cx = x + w/2
                        new_cy = y + h/2
                        
                        # Calculate distance moved as a percentage of frame dimensions
                        dx = abs(new_cx - prev_cx) / frame_width
                        dy = abs(new_cy - prev_cy) / frame_height
                        
                        # If box jumped by more than 20% of frame dimension, consider it lost
                        if dx > 0.2 or dy > 0.2:
                            logger.warning(f"Tracker {track_id} jumped too far (dx={dx:.2f}, dy={dy:.2f}), marking as failed")
                            failed_track_ids.append(track_id)
                            continue
                    
                    # If all checks pass, accept the tracked box
                    updated_box = [x, y, x + w, y + h]
                    self.boxes[track_id] = updated_box
                    updated_boxes.append(updated_box)
                    updated_classes.append(self.classes[track_id])
                else:
                    failed_track_ids.append(track_id)
                    logger.info(f"Lost track of object {track_id}")
            except Exception as e:
                logger.error(f"Error updating tracker {track_id}: {e}")
                failed_track_ids.append(track_id)
        
        # Remove failed trackers
        for track_id in failed_track_ids:
            if track_id in self.trackers:
                del self.trackers[track_id]
            if track_id in self.boxes:
                del self.boxes[track_id]
            if track_id in self.classes:
                del self.classes[track_id]
        
        if failed_track_ids:
            logger.info(f"Removed {len(failed_track_ids)} failed trackers, {len(self.trackers)} remaining")
            
        logger.info(f"Tracked {len(updated_boxes)} objects successfully")
        
        return updated_boxes, updated_classes
    
    def add_new_tracker(self, frame, box, class_id):
        """Add a new tracker for a new box"""
        # Convert PIL image to OpenCV format if needed
        if isinstance(frame, Image.Image):
            frame_np = np.array(frame)
            frame_cv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
            self.tracked_frame = frame_cv.copy()
        elif self.tracked_frame is not None:
            frame_cv = self.tracked_frame
        else:
            logger.error("No valid frame for tracking")
            return False
            
        tracker = self.create_tracker()
        x1, y1, x2, y2 = box
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        
        # Ensure box dimensions are valid
        if w <= 0 or h <= 0:
            logger.warning(f"Invalid box dimensions: w={w}, h={h}, skipping")
            return False
            
        track_id = self.next_id
        
        try:
            success = tracker.init(frame_cv, (x, y, w, h))
            
            if success:
                self.trackers[track_id] = tracker
                self.boxes[track_id] = box
                self.classes[track_id] = class_id
                self.next_id += 1
                return True
            else:
                logger.warning(f"Failed to initialize new tracker")
                return False
        except Exception as e:
            logger.error(f"Error initializing new tracker: {e}")
            return False
    
    def get_tracked_objects(self):
        """Return all currently tracked objects"""
        boxes = list(self.boxes.values())
        classes = list(self.classes.values())
        return boxes, classes


class SAMBoxTracker:
    """
    Box-to-mask-to-box tracker using SAM
    
    This tracker uses SAM to:
    1. Take a bounding box from previous frame
    2. Generate a mask using that box as a prompt
    3. Convert the refined mask back to a bounding box
    4. Use that new box for tracking
    
    This approach is more robust for non-rigid objects than pure box trackers.
    """
    def __init__(self, sam_predictor, device):
        self.boxes = {}     # Dictionary to store {track_id: box}
        self.classes = {}   # Dictionary to store {track_id: class_id}
        self.logits = {}    # Dictionary to store {track_id: logits} for temporal consistency
        self.next_id = 0    # Counter for assigning unique track IDs
        self.sam_predictor = sam_predictor
        self.device = device
        
    def initialize_trackers(self, frame, boxes, class_ids):
        """Initialize trackers with bounding boxes from the current frame"""
        if not boxes:
            logger.warning("No boxes to track with SAM")
            return
            
        # Clear existing trackers
        self.boxes = {}
        self.classes = {}
        self.logits = {}
        self.next_id = 0
        
        # Set image for SAM
        self.sam_predictor.set_image(frame)
        
        # Initialize tracking for each box
        ctx = torch.autocast(self.device, dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
        for i, (box, class_id) in enumerate(zip(boxes, class_ids)):
            track_id = self.next_id
            
            try:
                with torch.inference_mode(), ctx:
                    # Get mask from box
                    masks, scores, logits = self.sam_predictor.predict(
                        box=np.array(box)[None],  # shape (1,4)
                        multimask_output=False
                    )
                
                # Store results
                self.boxes[track_id] = box
                self.classes[track_id] = class_id
                if isinstance(logits[0], torch.Tensor):
                    self.logits[track_id] = logits[0].cpu()
                else:
                    self.logits[track_id] = logits[0]
                self.next_id += 1
                
            except Exception as e:
                logger.error(f"Error initializing SAM tracker for box {i}: {e}")
        
        logger.info(f"Initialized {len(self.boxes)} SAM trackers")

    def update_trackers(self, frame):
        """Update all trackers with the new frame and return the tracked boxes and classes"""
        if not self.boxes:
            logger.warning("No SAM trackers to update")
            return [], []
            
        # Set image for SAM
        self.sam_predictor.set_image(frame)
        
        updated_boxes = []
        updated_classes = []
        failed_track_ids = []
        
        # Process the frame dimensions for boundary checking
        if isinstance(frame, Image.Image):
            frame_height, frame_width = frame.height, frame.width
        else:
            frame_height, frame_width = frame.shape[:2]
        
        ctx = torch.autocast(self.device, dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
        
        # Update each tracker
        for track_id in list(self.boxes.keys()):
            try:
                prev_box = self.boxes[track_id]
                prev_logits = self.logits.get(track_id)
                
                with torch.inference_mode(), ctx:
                    # If we have previous logits, use them for temporal consistency
                    if prev_logits is not None:
                        masks, scores, logits = self.sam_predictor.predict(
                            box=np.array(prev_box)[None],  # shape (1,4)
                            mask_input=prev_logits[None] if isinstance(prev_logits, np.ndarray) else prev_logits.unsqueeze(0),
                            multimask_output=False
                        )
                    else:
                        # Otherwise just use the box
                        masks, scores, logits = self.sam_predictor.predict(
                            box=np.array(prev_box)[None],  # shape (1,4)
                            multimask_output=False
                        )
                
                # Convert mask to bounding boxes
                mask = (masks[0] > 0.5).astype(np.uint8)
                bbs = mask_to_bboxes(mask)
                
                if not bbs:
                    # No valid boxes found, use previous box as fallback
                    updated_box = prev_box
                else:
                    # Choose the box with largest area
                    areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in bbs]
                    best_idx = int(np.argmax(areas))
                    updated_box = bbs[best_idx]
                    
                    # Check if box is reasonable and mostly inside the frame
                    x1, y1, x2, y2 = updated_box
                    w, h = x2 - x1, y2 - y1
                    
                    # Check if box dimensions are reasonable (not too small or too large)
                    if w < 3 or h < 3 or w > frame_width*0.9 or h > frame_height*0.9:
                        logger.warning(f"SAM tracker {track_id} returned unreasonable box size: {w}x{h}, using previous box")
                        updated_box = prev_box
                    
                    # Check if box is mostly inside the frame
                    elif x1 < -w/2 or y1 < -h/2 or x2 > frame_width + w/2 or y2 > frame_height + h/2:
                        logger.warning(f"SAM tracker {track_id} returned box mostly outside frame, using previous box")
                        updated_box = prev_box
                    
                    # Check if box has moved too far from previous position
                    else:
                        prev_x1, prev_y1, prev_x2, prev_y2 = prev_box
                        prev_cx = (prev_x1 + prev_x2) / 2
                        prev_cy = (prev_y1 + prev_y2) / 2
                        new_cx = (x1 + x2) / 2
                        new_cy = (y1 + y2) / 2
                        
                        # Calculate distance moved as a percentage of frame dimensions
                        dx = abs(new_cx - prev_cx) / frame_width
                        dy = abs(new_cy - prev_cy) / frame_height
                        
                        # If box jumped by more than 25% of frame dimension, consider it suspicious
                        if dx > 0.25 or dy > 0.25:
                            logger.warning(f"SAM tracker {track_id} jumped too far (dx={dx:.2f}, dy={dy:.2f}), using previous box")
                            updated_box = prev_box
                
                # Update tracker with new box and logits
                self.boxes[track_id] = updated_box
                if isinstance(logits[0], torch.Tensor):
                    self.logits[track_id] = logits[0].cpu()
                else:
                    self.logits[track_id] = logits[0]
                    
                # Add to results
                updated_boxes.append(updated_box)
                updated_classes.append(self.classes[track_id])
                
            except Exception as e:
                logger.error(f"Error updating SAM tracker {track_id}: {e}")
                failed_track_ids.append(track_id)
        
        # Remove failed trackers
        for track_id in failed_track_ids:
            if track_id in self.boxes:
                del self.boxes[track_id]
            if track_id in self.classes:
                del self.classes[track_id]
            if track_id in self.logits:
                del self.logits[track_id]
        
        if failed_track_ids:
            logger.info(f"Removed {len(failed_track_ids)} failed SAM trackers, {len(self.boxes)} remaining")
            
        logger.info(f"SAM tracked {len(updated_boxes)} objects successfully")
        
        return updated_boxes, updated_classes
    
    def add_new_tracker(self, frame, box, class_id):
        """Add a new tracker for a new box"""
        # Set image for SAM
        self.sam_predictor.set_image(frame)
        
        track_id = self.next_id
        
        try:
            ctx = torch.autocast(self.device, dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
            with torch.inference_mode(), ctx:
                # Get mask from box
                masks, scores, logits = self.sam_predictor.predict(
                    box=np.array(box)[None],  # shape (1,4)
                    multimask_output=False
                )
            
            # Store results
            self.boxes[track_id] = box
            self.classes[track_id] = class_id
            if isinstance(logits[0], torch.Tensor):
                self.logits[track_id] = logits[0].cpu()
            else:
                self.logits[track_id] = logits[0]
            self.next_id += 1
            return True
                
        except Exception as e:
            logger.error(f"Error initializing new SAM tracker: {e}")
            return False
    
    def get_tracked_objects(self):
        """Return all currently tracked objects"""
        boxes = list(self.boxes.values())
        classes = list(self.classes.values())
        return boxes, classes


################################################################################
# MANUAL ANNOTATION INTERFACE
################################################################################
class ManualAnnotationInterface:
    def __init__(self):
        self.root = None
        self.canvas = None
        self.photo = None
        self.frame = None
        self.boxes = []
        self.class_indices = []
        self.start_x = None
        self.start_y = None
        self.drawing = False
        self.current_box_id = None
        self.resizing = False
        self.moving = False
        self.resize_handle = None
        self.resize_handle_size = 3  # Reduced from 8 to 3 pixels
        self.last_mouse_x = None
        self.last_mouse_y = None
        self.master = None  # Reference to parent BBoxAnnotationSystem

    def manual_correction_interface(self, frames, current_frame_idx, boxes_by_frame, classes_by_frame):
        """
        Interface for manual correction with frame navigation within the same interface
        
        Args:
            frames: List of all frames
            current_frame_idx: Current frame index
            boxes_by_frame: Dictionary of boxes for each frame index
            classes_by_frame: Dictionary of class indices for each frame index
        
        Returns:
            Tuple of (all_boxes_by_frame, all_classes_by_frame, final_frame_idx)
        """
        self.current_frame_idx = current_frame_idx
        self.frames = frames
        self.boxes_by_frame = boxes_by_frame.copy() if boxes_by_frame else {}
        self.classes_by_frame = classes_by_frame.copy() if classes_by_frame else {}
        self.last_tracked_frame = current_frame_idx  # Track which frame was last processed
        self.playing = False  # Flag for auto-playback
        self.play_speed = 0.1  # Seconds between frames during playback
        self.tracker_type = self.master.tracker_type if hasattr(self.master, "tracker_type") else "CSRT"
        
        # Initialize trackers for this session
        self.csrt_tracker = CSRTObjectTracker()
        
        # Initialize SAM tracker if using SAM
        if self.tracker_type == "SAM":
            if hasattr(self.master, "sam_predictor") and self.master.sam_predictor is not None:
                self.sam_tracker = SAMBoxTracker(self.master.sam_predictor, self.master.device)
            else:
                logger.warning("SAM predictor not available, falling back to CSRT tracker")
                self.tracker_type = "CSRT"
        
        # Initialize current frame's boxes and classes
        if self.current_frame_idx in self.boxes_by_frame:
            self.boxes = self.boxes_by_frame[self.current_frame_idx].copy()
        else:
            self.boxes = []
            
        if self.current_frame_idx in self.classes_by_frame:
            self.class_indices = self.classes_by_frame[self.current_frame_idx].copy()
        else:
            self.class_indices = []
        
        # Set current frame
        self.frame = frames[self.current_frame_idx]
        
        # Initialize trackers with current boxes if they exist
        if self.boxes:
            try:
                if self.tracker_type == "SAM":
                    self.sam_tracker.initialize_trackers(self.frame, self.boxes, self.class_indices)
                    logger.info(f"Initialized SAM tracking with {len(self.boxes)} boxes")
                else:
                    self.csrt_tracker.initialize_trackers(self.frame, self.boxes, self.class_indices)
                    logger.info(f"Initialized CSRT tracking with {len(self.boxes)} boxes")
            except Exception as e:
                logger.error(f"Error initializing trackers: {e}")
        
        # Create main window if not exists
        if self.root is None:
            self.root = tk.Tk()
            self.root.title("Object Detection Annotation")
            
            # Create the auto-clear variable AFTER the root window
            self.auto_clear_future = tk.BooleanVar(value=False)  # Option to auto-clear future frames
            
            main_frame = ttk.Frame(self.root)
            main_frame.pack(fill=tk.BOTH, expand=True)
            
            self.canvas = tk.Canvas(main_frame, width=self.frame.width, height=self.frame.height)
            self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            control_panel = ttk.Frame(main_frame, padding=10)
            control_panel.pack(side=tk.RIGHT, fill=tk.Y)
            
            ttk.Label(control_panel, text="Object Class:").pack(pady=5)
            self.class_var = tk.IntVar(value=0)
            ttk.Radiobutton(control_panel, text="Obstacle (Class 0)", variable=self.class_var, value=0).pack(anchor=tk.W)
            ttk.Radiobutton(control_panel, text="Goal (Class 1)", variable=self.class_var, value=1).pack(anchor=tk.W)
            ttk.Radiobutton(control_panel, text="Other (Class 2)", variable=self.class_var, value=2).pack(anchor=tk.W)
            
            ttk.Label(control_panel, text="Actions:").pack(pady=5, anchor=tk.W)
            ttk.Button(control_panel, text="Delete Selected", command=self.delete_selected_box).pack(pady=2, fill=tk.X)
            ttk.Button(control_panel, text="Clear All", command=self.clear_all_boxes).pack(pady=2, fill=tk.X)
            
            # Tracking control
            tracking_frame = ttk.Frame(control_panel)
            tracking_frame.pack(pady=5, fill=tk.X)
            
            self.tracking_var = tk.BooleanVar(value=True)
            ttk.Checkbutton(tracking_frame, text="Enable Tracking", variable=self.tracking_var).pack(side=tk.LEFT)
            
            # Add tracker type selection
            tracker_type_frame = ttk.Frame(control_panel)
            tracker_type_frame.pack(pady=2, fill=tk.X)
            
            ttk.Label(tracker_type_frame, text="Tracker Type:").pack(side=tk.LEFT)
            self.tracker_type_var = tk.StringVar(value=self.tracker_type)
            tracker_combo = ttk.Combobox(tracker_type_frame, 
                                         textvariable=self.tracker_type_var, 
                                         values=["CSRT", "SAM"], 
                                         width=8,
                                         state="readonly")
            tracker_combo.pack(side=tk.RIGHT, padx=5)
            
            def on_tracker_change(event):
                new_type = self.tracker_type_var.get()
                if new_type != self.tracker_type:
                    self.tracker_type = new_type
                    # Initialize appropriate tracker
                    if self.tracker_type == "SAM":
                        if hasattr(self.master, "sam_predictor") and self.master.sam_predictor is not None:
                            self.sam_tracker = SAMBoxTracker(self.master.sam_predictor, self.master.device)
                            messagebox.showinfo("Tracker Changed", "Switched to SAM box-to-mask-to-box tracker")
                        else:
                            messagebox.showerror("Error", "SAM predictor not available")
                            self.tracker_type = "CSRT"
                            self.tracker_type_var.set("CSRT")
                    else:
                        self.csrt_tracker = CSRTObjectTracker()
                        messagebox.showinfo("Tracker Changed", "Switched to CSRT tracker")
                    
                    # Reset with current boxes
                    if self.boxes:
                        reset_tracker()
                    
            tracker_combo.bind("<<ComboboxSelected>>", on_tracker_change)
            
            def reset_tracker():
                if self.boxes:
                    try:
                        if self.tracker_type == "SAM":
                            if hasattr(self, 'sam_tracker') and self.sam_tracker is not None:
                                self.sam_tracker.initialize_trackers(self.frame, self.boxes, self.class_indices)
                                messagebox.showinfo("Tracker Reset", f"Reinitialized SAM tracking with {len(self.boxes)} boxes")
                            else:
                                messagebox.showerror("Error", "SAM tracker not available")
                        else:
                            self.csrt_tracker.initialize_trackers(self.frame, self.boxes, self.class_indices)
                            messagebox.showinfo("Tracker Reset", f"Reinitialized CSRT tracking with {len(self.boxes)} boxes")
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to reset tracker: {e}")
                else:
                    messagebox.showinfo("No Boxes", "No boxes to track. Create some annotations first.")
            
            ttk.Button(tracking_frame, text="Reset Tracker", command=reset_tracker).pack(side=tk.RIGHT)
            
            # Auto-clear future frames option
            clear_frame = ttk.Frame(control_panel)
            clear_frame.pack(pady=2, fill=tk.X)
            ttk.Checkbutton(clear_frame, text="Auto-clear future frames", variable=self.auto_clear_future).pack(anchor=tk.W)
            
            # Clear all future frames button
            def clear_all_future_frames():
                future_frames = [idx for idx in self.boxes_by_frame.keys() if idx > self.current_frame_idx]
                if future_frames:
                    if messagebox.askyesno("Clear Future Frames", 
                                          f"Clear annotations for {len(future_frames)} frames after this point?"):
                        for idx in future_frames:
                            if idx in self.boxes_by_frame:
                                del self.boxes_by_frame[idx]
                            if idx in self.classes_by_frame:
                                del self.classes_by_frame[idx]
                        logger.info(f"Cleared annotations for {len(future_frames)} future frames")
                        messagebox.showinfo("Cleared", f"Cleared annotations for {len(future_frames)} future frames")
                else:
                    messagebox.showinfo("No Future Frames", "No annotated frames after this point.")
                
            ttk.Button(clear_frame, text="Clear All Future Frames", command=clear_all_future_frames).pack(pady=2, fill=tk.X)
            
            # Frame info display
            self.frame_info = ttk.Label(control_panel, text=f"Frame: {self.current_frame_idx+1}/{len(frames)}")
            self.frame_info.pack(pady=5, fill=tk.X)
            
            # Create frame to hold multiple navigation rows
            nav_container = ttk.Frame(control_panel)
            nav_container.pack(pady=5, fill=tk.X)
            
            # Basic navigation buttons
            nav_frame = ttk.Frame(nav_container)
            nav_frame.pack(pady=3, fill=tk.X)
            
            ttk.Button(nav_frame, text="← Previous", command=self.go_to_previous_frame).pack(side=tk.LEFT, fill=tk.X, expand=True)
            ttk.Button(nav_frame, text="Next →", command=self.go_to_next_frame).pack(side=tk.RIGHT, fill=tk.X, expand=True)
            
            # Skip multiple frames buttons
            skip_frame = ttk.Frame(nav_container)
            skip_frame.pack(pady=3, fill=tk.X)
            
            ttk.Label(skip_frame, text="Skip:").pack(side=tk.LEFT)
            self.skip_var = tk.StringVar(value="10")
            skip_entry = ttk.Entry(skip_frame, textvariable=self.skip_var, width=5)
            skip_entry.pack(side=tk.LEFT, padx=5)
            
            def skip_forward():
                try:
                    num_frames = int(self.skip_var.get())
                    self.skip_frames(num_frames)
                except ValueError:
                    messagebox.showwarning("Invalid Input", "Please enter a valid number of frames")
                    
            def skip_backward():
                try:
                    num_frames = -int(self.skip_var.get())
                    self.skip_frames(num_frames)
                except ValueError:
                    messagebox.showwarning("Invalid Input", "Please enter a valid number of frames")
            
            ttk.Button(skip_frame, text="← Back", command=skip_backward).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
            ttk.Button(skip_frame, text="Forward →", command=skip_forward).pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)
            
            # Jump to frame entry
            jump_frame = ttk.Frame(nav_container)
            jump_frame.pack(pady=3, fill=tk.X)
            
            ttk.Label(jump_frame, text="Jump to:").pack(side=tk.LEFT)
            self.jump_var = tk.StringVar()
            jump_entry = ttk.Entry(jump_frame, textvariable=self.jump_var, width=8)
            jump_entry.pack(side=tk.LEFT, padx=5)
            
            def jump_to_frame():
                try:
                    frame_num = int(self.jump_var.get())
                    if 1 <= frame_num <= len(self.frames):
                        self.jump_to_frame(frame_num - 1)  # Convert to 0-based index
                    else:
                        messagebox.showwarning("Invalid Frame", f"Please enter a frame number between 1 and {len(self.frames)}")
                except ValueError:
                    messagebox.showwarning("Invalid Input", "Please enter a valid frame number")
            
            ttk.Button(jump_frame, text="Go", command=jump_to_frame).pack(side=tk.LEFT)
            
            # Playback controls
            play_frame = ttk.Frame(nav_container)
            play_frame.pack(pady=3, fill=tk.X)
            
            self.play_button = ttk.Button(play_frame, text="▶ Play", command=self.toggle_playback)
            self.play_button.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            # Playback speed control
            speed_frame = ttk.Frame(play_frame)
            speed_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True)
            
            ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
            speed_values = ["Slow", "Normal", "Fast"]
            self.speed_var = tk.StringVar(value="Normal")
            
            speed_combo = ttk.Combobox(speed_frame, textvariable=self.speed_var, values=speed_values, width=8)
            speed_combo.pack(side=tk.LEFT, padx=5)
            
            def on_speed_change(event):
                speed = self.speed_var.get()
                if speed == "Slow":
                    self.play_speed = 0.2  # 5 FPS
                elif speed == "Normal":
                    self.play_speed = 0.1  # 10 FPS
                elif speed == "Fast":
                    self.play_speed = 0.033  # 30 FPS
                logger.info(f"Set playback speed to {speed} ({1/self.play_speed:.1f} FPS)")
            
            speed_combo.bind("<<ComboboxSelected>>", on_speed_change)
            
            # Save and quit button
            ttk.Button(control_panel, text="Save and Quit", command=self.save_and_quit).pack(pady=10, fill=tk.X)
            
            # Add a tracking status display
            self.tracking_status = ttk.Label(control_panel, text="")
            self.tracking_status.pack(pady=5, fill=tk.X)
            
            ttk.Separator(control_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=5)
            ttk.Label(control_panel, text="Instructions:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=5)
            ttk.Label(control_panel, text="• Click to select a box").pack(anchor=tk.W)
            ttk.Label(control_panel, text="• Drag to create a new box").pack(anchor=tk.W)
            ttk.Label(control_panel, text="• Click a box corner to resize").pack(anchor=tk.W)
            ttk.Label(control_panel, text="• Click center to move box").pack(anchor=tk.W)
            ttk.Label(control_panel, text="• Use nav buttons to change frames").pack(anchor=tk.W)
            ttk.Label(control_panel, text="• Enter number of frames to skip").pack(anchor=tk.W)
            ttk.Label(control_panel, text="• Press Space to play/pause").pack(anchor=tk.W)
            
            self.canvas.bind("<ButtonPress-1>", self.on_box_press)
            self.canvas.bind("<B1-Motion>", self.on_box_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_box_release)
            
            self.root.bind("<Delete>", lambda e: self.delete_selected_box())
            self.root.bind("<Left>", lambda e: self.go_to_previous_frame())
            self.root.bind("<Right>", lambda e: self.go_to_next_frame())
            self.root.bind("<Escape>", lambda e: self.save_and_quit())
            self.root.bind("<space>", lambda e: self.toggle_playback())
            
            # Flag to indicate when to quit the main loop
            self.should_quit = False
        else:
            # Update frame info if the interface already exists
            self.frame_info.config(text=f"Frame: {self.current_frame_idx+1}/{len(frames)}")
        
        self.update_box_display()
        
        # Get the current time for playback
        last_frame_time = time.time()
        
        # Main interface loop
        while not self.should_quit:
            current_time = time.time()
            
            # Handle automatic playback
            if self.playing and current_time - last_frame_time > self.play_speed:
                # Time to advance to next frame
                if self.current_frame_idx < len(self.frames) - 1:
                    self.go_to_next_frame()
                    last_frame_time = current_time
                else:
                    # End of frames, stop playing
                    self.playing = False
                    self.play_button.config(text="▶ Play")
            
            self.root.update()
            time.sleep(0.01)  # Small delay to prevent CPU hogging
            
        # Clean up
        self.root.destroy()
        self.root = None
        
        # Return updated data
        return self.boxes_by_frame, self.classes_by_frame, self.current_frame_idx
    
    def toggle_playback(self):
        """Toggle playback state between playing and paused"""
        self.playing = not self.playing
        
        if self.playing:
            self.play_button.config(text="⏸ Pause")
            logger.info(f"Started playback at {1/self.play_speed:.1f} FPS")
        else:
            self.play_button.config(text="▶ Play")
            logger.info("Paused playback")
    
    def skip_frames(self, num_frames):
        """Skip multiple frames forward or backward"""
        if num_frames == 0:
            return
            
        target_frame = self.current_frame_idx + num_frames
        
        # Clamp to valid frame range
        target_frame = max(0, min(target_frame, len(self.frames) - 1))
        
        # If we're not moving, do nothing
        if target_frame == self.current_frame_idx:
            return
        
        # Save current frame
        self.save_current_frame()
        
        # Figure out if we're going forward or backward
        going_forward = target_frame > self.current_frame_idx
        
        # If going backward, clear future frames if auto-clear is enabled
        if not going_forward and hasattr(self, 'auto_clear_future') and self.auto_clear_future.get():
            future_frames = [idx for idx in self.boxes_by_frame.keys() if idx > target_frame]
            if future_frames:
                for idx in future_frames:
                    if idx in self.boxes_by_frame:
                        del self.boxes_by_frame[idx]
                    if idx in self.classes_by_frame:
                        del self.classes_by_frame[idx]
                logger.info(f"Auto-cleared annotations for {len(future_frames)} future frames")
                self.tracking_status.config(text=f"Cleared {len(future_frames)} future frames")
                self.root.update()
        
        # Update status
        self.tracking_status.config(text=f"Tracking frames...")
        self.root.update()  # Force UI update to show status
        
        if going_forward:
            logger.info(f"Skipping forward {target_frame - self.current_frame_idx} frames to frame {target_frame}")
            
            # Process each frame sequentially to ensure proper tracking
            current = self.current_frame_idx
            
            # Initialize tracking with current frame's boxes if tracking is enabled
            if self.tracking_var.get() and self.boxes:
                try:
                    self.csrt_tracker.initialize_trackers(self.frame, self.boxes, self.class_indices)
                    logger.info(f"Initialized tracking with {len(self.boxes)} boxes")
                except Exception as e:
                    logger.error(f"Error initializing trackers: {e}")
            
            # Track progress for status updates
            total_frames = target_frame - current
            frames_processed = 0
            
            # Process each intermediate frame in sequence
            while current < target_frame:
                current += 1
                frames_processed += 1
                
                # Update status every 5 frames
                if frames_processed % 5 == 0 or frames_processed == total_frames:
                    progress = int((frames_processed / total_frames) * 100)
                    self.tracking_status.config(text=f"Tracking... {progress}% ({frames_processed}/{total_frames})")
                    self.root.update()  # Force UI update to show progress
                
                # Get the next frame
                next_frame = self.frames[current]
                
                # If this frame already has saved annotations, use them
                if current in self.boxes_by_frame:
                    boxes = self.boxes_by_frame[current].copy()
                    classes = self.classes_by_frame[current].copy()
                    
                    # Update trackers with these saved annotations
                    if self.tracking_var.get():
                        try:
                            self.csrt_tracker.initialize_trackers(next_frame, boxes, classes)
                        except Exception as e:
                            logger.error(f"Error reinitializing trackers on frame {current}: {e}")
                else:
                    # No saved annotations, use tracking to predict
                    if self.tracking_var.get() and hasattr(self, 'csrt_tracker') and self.csrt_tracker is not None:
                        try:
                            boxes, classes = self.csrt_tracker.update_trackers(next_frame)
                            
                            # If tracking succeeded, save these results
                            if boxes:
                                self.boxes_by_frame[current] = boxes.copy()
                                self.classes_by_frame[current] = classes.copy()
                                logger.info(f"Tracked and saved {len(boxes)} boxes for frame {current}")
                            else:
                                # Tracking failed, try to use previous frame's boxes
                                if current - 1 in self.boxes_by_frame:
                                    prev_boxes = self.boxes_by_frame[current - 1].copy()
                                    prev_classes = self.classes_by_frame[current - 1].copy()
                                    self.boxes_by_frame[current] = prev_boxes
                                    self.classes_by_frame[current] = prev_classes
                                    logger.info(f"Using previous frame's {len(prev_boxes)} boxes for frame {current}")
                                    
                                    # Try to reinitialize tracking with these boxes
                                    try:
                                        self.csrt_tracker.initialize_trackers(next_frame, prev_boxes, prev_classes)
                                    except Exception as e:
                                        logger.error(f"Error reinitializing trackers on frame {current}: {e}")
                        except Exception as e:
                            logger.error(f"Error tracking on frame {current}: {e}")
            
            # Now set to the final target frame
            self.current_frame_idx = target_frame
            self.frame = self.frames[self.current_frame_idx]
            
            # Load the annotations for the target frame
            if self.current_frame_idx in self.boxes_by_frame:
                self.boxes = self.boxes_by_frame[self.current_frame_idx].copy()
                self.class_indices = self.classes_by_frame[self.current_frame_idx].copy()
                logger.info(f"Loaded saved annotations for target frame {self.current_frame_idx}")
            else:
                # This should rarely happen as we should have tracked to this frame
                logger.warning(f"No annotations found for target frame {self.current_frame_idx}")
                self.boxes = []
                self.class_indices = []
        else:
            # Going backward
            logger.info(f"Skipping backward {self.current_frame_idx - target_frame} frames to frame {target_frame}")
            
            # Track progress for status updates
            total_frames = self.current_frame_idx - target_frame
            frames_processed = 0
            
            # Process each frame in reverse to ensure proper tracking if needed
            current = self.current_frame_idx
            
            while current > target_frame:
                current -= 1
                frames_processed += 1
                
                # Update status every 5 frames
                if frames_processed % 5 == 0 or frames_processed == total_frames:
                    progress = int((frames_processed / total_frames) * 100)
                    self.tracking_status.config(text=f"Tracking... {progress}% ({frames_processed}/{total_frames})")
                    self.root.update()  # Force UI update to show progress
                
                # When going backward, we prioritize using saved annotations
                if current in self.boxes_by_frame:
                    # Already have annotations, no need to track
                    continue
                
                # If no saved annotations and tracking enabled, generate them
                if self.tracking_var.get():
                    # Find nearest prior frame with annotations
                    best_prior_frame = -1
                    for frame_idx in range(current-1, max(0, current-10), -1):
                        if frame_idx in self.boxes_by_frame:
                            best_prior_frame = frame_idx
                            break
                    
                    # If found a prior frame, track forward to current
                    if best_prior_frame >= 0:
                        # Track sequentially from the prior frame to this one
                        boxes = self.boxes_by_frame[best_prior_frame].copy()
                        classes = self.classes_by_frame[best_prior_frame].copy()
                        
                        # Initialize tracker with the prior frame
                        try:
                            self.csrt_tracker.initialize_trackers(
                                self.frames[best_prior_frame], 
                                boxes, 
                                classes
                            )
                            
                            # Track through each frame
                            for idx in range(best_prior_frame + 1, current + 1):
                                frame_img = self.frames[idx]
                                tracked_boxes, tracked_classes = self.csrt_tracker.update_trackers(frame_img)
                                
                                if tracked_boxes:
                                    self.boxes_by_frame[idx] = tracked_boxes.copy()
                                    self.classes_by_frame[idx] = tracked_classes.copy()
                                    
                                    # Continue tracking with these results
                                    self.csrt_tracker.initialize_trackers(
                                        frame_img, 
                                        tracked_boxes, 
                                        tracked_classes
                                    )
                                else:
                                    # Tracking failed, use the last known boxes
                                    self.boxes_by_frame[idx] = boxes.copy()
                                    self.classes_by_frame[idx] = classes.copy()
                        except Exception as e:
                            logger.error(f"Error tracking backward to frame {current}: {e}")
            
            # Now set to the final target frame
            self.current_frame_idx = target_frame
            self.frame = self.frames[self.current_frame_idx]
            
            # Load the annotations for the target frame
            if self.current_frame_idx in self.boxes_by_frame:
                self.boxes = self.boxes_by_frame[self.current_frame_idx].copy()
                self.class_indices = self.classes_by_frame[self.current_frame_idx].copy()
                logger.info(f"Loaded saved annotations for target frame {self.current_frame_idx}")
                
                # Reinitialize trackers with these saved boxes for future tracking
                if self.tracking_var.get() and hasattr(self, 'csrt_tracker'):
                    try:
                        self.csrt_tracker.initialize_trackers(self.frame, self.boxes, self.class_indices)
                    except Exception as e:
                        logger.error(f"Error reinitializing trackers: {e}")
            else:
                self.boxes = []
                self.class_indices = []
        
        # Update UI and clear status
        self.tracking_status.config(text="")
        self.frame_info.config(text=f"Frame: {self.current_frame_idx+1}/{len(self.frames)}")
        self.current_box_id = None
        self.update_box_display()
        self.last_tracked_frame = self.current_frame_idx
    
    def jump_to_frame(self, frame_idx):
        """Jump directly to a specific frame"""
        self.skip_frames(frame_idx - self.current_frame_idx)
    
    def save_current_frame(self):
        """Save the current frame's annotations"""
        self.boxes_by_frame[self.current_frame_idx] = self.boxes.copy()
        self.classes_by_frame[self.current_frame_idx] = self.class_indices.copy()
        logger.info(f"Saved annotations for frame {self.current_frame_idx}")
    
    def go_to_next_frame(self):
        """Navigate to the next frame"""
        if self.current_frame_idx < len(self.frames) - 1:
            # Save current frame
            self.save_current_frame()
            
            # Reinitialize tracker with ONLY the current boxes
            # This ensures deleted objects don't come back
            if self.tracking_var.get() and self.boxes:
                try:
                    # Create a new tracker to ensure clean state
                    self.csrt_tracker = CSRTObjectTracker()
                    
                    # Initialize with current boxes only
                    self.csrt_tracker.initialize_trackers(self.frame, self.boxes, self.class_indices)
                    logger.info(f"Reinitialized clean tracker with {len(self.boxes)} current boxes")
                except Exception as e:
                    logger.error(f"Error initializing trackers: {e}")
            
            # Move to next frame
            self.current_frame_idx += 1
            
            # Load next frame data
            self.frame = self.frames[self.current_frame_idx]
            
            # ---- SAM2 mask tracking first ----
            if hasattr(self.master, "sam_logits") and self.tracking_var.get():
                self.master.sam_predictor.set_image(self.frame)
                # Use nullcontext on CPU since autocast only works with CUDA
                ctx = torch.autocast(self.master.device, dtype=torch.bfloat16) if self.master.device == "cuda" else nullcontext()
                with torch.inference_mode(), ctx:
                    m, s, log = self.master.sam_predictor.predict(
                        mask_input=self.master.sam_logits[None], multimask_output=True
                    )
                if m is not None and len(m) > 0:
                    best = int(np.argmax(s))
                    # Check if log is a torch tensor or numpy array
                    if isinstance(log[best], torch.Tensor):
                        self.master.sam_logits = log[best].cpu()
                    else:
                        # Already a numpy array
                        self.master.sam_logits = log[best]
                    self.master.sam_conf = float(s[best])
                    sam_mask = (m[best] > 0.5).astype(np.uint8)
                    bbs = mask_to_bboxes(sam_mask)
                    if bbs:
                        self.boxes = bbs
                        self.class_indices = [self.class_var.get()] * len(bbs)
                        logger.info(f"SAM2 tracking found {len(bbs)} boxes")
            
            # If this frame already has saved annotations, use them
            if self.current_frame_idx in self.boxes_by_frame:
                self.boxes = self.boxes_by_frame[self.current_frame_idx].copy()
                self.class_indices = self.classes_by_frame[self.current_frame_idx].copy()
                logger.info(f"Loaded saved annotations for frame {self.current_frame_idx}")
            # Otherwise try tracking if enabled
            elif self.tracking_var.get():
                # Use the appropriate tracker based on selected type
                if self.tracker_type == "SAM" and hasattr(self, 'sam_tracker') and self.sam_tracker is not None:
                    try:
                        tracked_boxes, tracked_classes = self.sam_tracker.update_trackers(self.frame)
                        if tracked_boxes:
                            self.boxes = tracked_boxes
                            self.class_indices = tracked_classes
                            logger.info(f"Used SAM tracking for next frame {self.current_frame_idx} - found {len(tracked_boxes)} boxes")
                        else:
                            # If tracking completely failed, use previous frame's boxes instead of empty
                            prev_boxes = self.boxes_by_frame.get(self.current_frame_idx - 1, [])
                            prev_classes = self.classes_by_frame.get(self.current_frame_idx - 1, [])
                            
                            if prev_boxes:
                                logger.info(f"SAM tracking failed, using previous frame's {len(prev_boxes)} boxes as fallback")
                                self.boxes = prev_boxes.copy()
                                self.class_indices = prev_classes.copy()
                                
                                # Reinitialize trackers with the previous frame's boxes
                                try:
                                    self.sam_tracker.initialize_trackers(self.frame, self.boxes, self.class_indices)
                                    logger.info(f"Reinitialized SAM tracking with previous frame's boxes")
                                except Exception as e:
                                    logger.error(f"Error reinitializing SAM trackers: {e}")
                            else:
                                # Fall back to empty if no previous boxes
                                self.boxes = []
                                self.class_indices = []
                                logger.info(f"No tracked boxes or previous boxes for frame {self.current_frame_idx}")
                    except Exception as e:
                        logger.error(f"Error updating SAM trackers: {e}")
                        # Fall back to previous frame's boxes
                        prev_boxes = self.boxes_by_frame.get(self.current_frame_idx - 1, [])
                        prev_classes = self.classes_by_frame.get(self.current_frame_idx - 1, [])
                        
                        if prev_boxes:
                            logger.info(f"SAM tracking error, using previous frame's {len(prev_boxes)} boxes as fallback")
                            self.boxes = prev_boxes.copy()
                            self.class_indices = prev_classes.copy()
                        else:
                            self.boxes = []
                            self.class_indices = []
                
                # Fall back to CSRT tracker if SAM is not available or if CSRT is selected
                elif hasattr(self, 'csrt_tracker') and self.csrt_tracker is not None:
                    try:
                        tracked_boxes, tracked_classes = self.csrt_tracker.update_trackers(self.frame)
                        if tracked_boxes:
                            self.boxes = tracked_boxes
                            self.class_indices = tracked_classes
                            logger.info(f"Used CSRT tracking for next frame {self.current_frame_idx} - found {len(tracked_boxes)} boxes")
                        else:
                            # If tracking completely failed, use previous frame's boxes instead of empty
                            # This gives better continuity when tracking fails
                            prev_boxes = self.boxes_by_frame.get(self.current_frame_idx - 1, [])
                            prev_classes = self.classes_by_frame.get(self.current_frame_idx - 1, [])
                            
                            if prev_boxes:
                                logger.info(f"CSRT tracking failed, using previous frame's {len(prev_boxes)} boxes as fallback")
                                self.boxes = prev_boxes.copy()
                                self.class_indices = prev_classes.copy()
                                
                                # Reinitialize trackers with the previous frame's boxes
                                try:
                                    self.csrt_tracker.initialize_trackers(self.frame, self.boxes, self.class_indices)
                                    logger.info(f"Reinitialized CSRT tracking with previous frame's boxes")
                                except Exception as e:
                                    logger.error(f"Error reinitializing CSRT trackers: {e}")
                            else:
                                # Fall back to empty if no previous boxes
                                self.boxes = []
                                self.class_indices = []
                                logger.info(f"No tracked boxes or previous boxes for frame {self.current_frame_idx}")
                    except Exception as e:
                        logger.error(f"Error updating CSRT trackers: {e}")
                        # Fall back to previous frame's boxes
                        prev_boxes = self.boxes_by_frame.get(self.current_frame_idx - 1, [])
                        prev_classes = self.classes_by_frame.get(self.current_frame_idx - 1, [])
                        
                        if prev_boxes:
                            logger.info(f"CSRT tracking error, using previous frame's {len(prev_boxes)} boxes as fallback")
                            self.boxes = prev_boxes.copy()
                            self.class_indices = prev_classes.copy()
                        else:
                            self.boxes = []
                            self.class_indices = []
            else:
                # No tracker available or tracking disabled, use previous frame's boxes as starting point
                prev_boxes = self.boxes_by_frame.get(self.current_frame_idx - 1, [])
                prev_classes = self.classes_by_frame.get(self.current_frame_idx - 1, [])
                
                if prev_boxes:
                    logger.info(f"Using previous frame's {len(prev_boxes)} boxes (tracking disabled)")
                    self.boxes = prev_boxes.copy()
                    self.class_indices = prev_classes.copy()
                else:
                    self.boxes = []
                    self.class_indices = []
            
            # Update UI
            self.frame_info.config(text=f"Frame: {self.current_frame_idx+1}/{len(self.frames)}")
            self.current_box_id = None
            self.update_box_display()
            self.last_tracked_frame = self.current_frame_idx
            logger.info(f"Moved to next frame {self.current_frame_idx}")
    
    def go_to_previous_frame(self):
        """Navigate to the previous frame"""
        if self.current_frame_idx > 0:
            # Save current frame
            self.save_current_frame()
            
            # Move to previous frame
            self.current_frame_idx -= 1
            
            # Clear all future frames' annotations if auto-clear is enabled
            if hasattr(self, 'auto_clear_future') and self.auto_clear_future.get():
                future_frames = [idx for idx in self.boxes_by_frame.keys() if idx > self.current_frame_idx]
                if future_frames:
                    for idx in future_frames:
                        if idx in self.boxes_by_frame:
                            del self.boxes_by_frame[idx]
                        if idx in self.classes_by_frame:
                            del self.classes_by_frame[idx]
                    logger.info(f"Auto-cleared annotations for {len(future_frames)} future frames")
                    self.tracking_status.config(text=f"Cleared {len(future_frames)} future frames")
                    self.root.update()
                    self.root.after(1500, lambda: self.tracking_status.config(text=""))  # Clear after 1.5 seconds
            
            # Load previous frame data
            self.frame = self.frames[self.current_frame_idx]
            
            # Always use saved annotations for previous frames if available
            if self.current_frame_idx in self.boxes_by_frame:
                self.boxes = self.boxes_by_frame[self.current_frame_idx].copy()
                self.class_indices = self.classes_by_frame[self.current_frame_idx].copy()
                logger.info(f"Loaded saved annotations for previous frame {self.current_frame_idx}")
                
                # Reinitialize trackers with these boxes for better tracking going forward
                if self.tracking_var.get() and hasattr(self, 'csrt_tracker') and self.csrt_tracker is not None:
                    try:
                        self.csrt_tracker.initialize_trackers(self.frame, self.boxes, self.class_indices)
                        logger.info(f"Reinitialized tracking for previous frame {self.current_frame_idx}")
                    except Exception as e:
                        logger.error(f"Error reinitializing trackers: {e}")
            else:
                # No saved annotations, start with empty boxes
                self.boxes = []
                self.class_indices = []
                logger.info(f"No saved annotations for previous frame {self.current_frame_idx}")
            
            # Update UI
            self.frame_info.config(text=f"Frame: {self.current_frame_idx+1}/{len(self.frames)}")
            self.current_box_id = None
            self.update_box_display()
            self.last_tracked_frame = self.current_frame_idx
            logger.info(f"Moved to previous frame {self.current_frame_idx}")
    
    def save_and_quit(self):
        """Save current frame and exit the interface"""
        self.save_current_frame()
        self.should_quit = True
        if hasattr(self, 'csrt_tracker'):
            self.csrt_tracker = None  # Clean up tracker

    def is_near_corner(self, x, y, box):
        """Check if (x,y) is near any corner of the box"""
        x1, y1, x2, y2 = box
        corners = [
            (x1, y1, "topleft"),
            (x2, y1, "topright"),
            (x1, y2, "bottomleft"),
            (x2, y2, "bottomright")
        ]
        for cx, cy, corner_name in corners:
            if abs(x - cx) <= self.resize_handle_size and abs(y - cy) <= self.resize_handle_size:
                return True, corner_name
        return False, None

    def is_inside_box(self, x, y, box):
        """Check if (x,y) is inside the box"""
        x1, y1, x2, y2 = box
        return x1 <= x <= x2 and y1 <= y <= y2

    def on_box_press(self, event):
        x, y = event.x, event.y
        
        # First check if near a corner of the selected box for resizing
        if self.current_box_id is not None and self.current_box_id < len(self.boxes):
            is_corner, corner_type = self.is_near_corner(x, y, self.boxes[self.current_box_id])
            if is_corner:
                self.resizing = True
                self.resize_handle = corner_type
                self.last_mouse_x = x
                self.last_mouse_y = y
                return
        
        # Check if clicking inside a box for selection or movement
        for i, box in enumerate(self.boxes):
            if self.is_inside_box(x, y, box):
                # If clicked inside the box, select it or prepare to move it
                if self.current_box_id == i:
                    # Box already selected, prepare to move
                    self.moving = True
                    self.last_mouse_x = x
                    self.last_mouse_y = y
                else:
                    # Just select the box
                    self.current_box_id = i
                self.update_box_display()
                return
        
        # If not on a box or corner, start drawing a new box
        self.start_x = x
        self.start_y = y
        self.drawing = True
        self.current_box_id = None
        self.resizing = False
        self.moving = False

    def on_box_drag(self, event):
        x, y = event.x, event.y
        
        if self.resizing and self.current_box_id is not None:
            # Handle resizing a box
            box = self.boxes[self.current_box_id].copy()
            x1, y1, x2, y2 = box
            
            if self.resize_handle == "topleft":
                x1 = x
                y1 = y
            elif self.resize_handle == "topright":
                x2 = x
                y1 = y
            elif self.resize_handle == "bottomleft":
                x1 = x
                y2 = y
            elif self.resize_handle == "bottomright":
                x2 = x
                y2 = y
            
            # Ensure x1 < x2 and y1 < y2
            if x1 > x2:
                x1, x2 = x2, x1
                if self.resize_handle == "topleft":
                    self.resize_handle = "topright"
                elif self.resize_handle == "topright":
                    self.resize_handle = "topleft"
                elif self.resize_handle == "bottomleft":
                    self.resize_handle = "bottomright"
                elif self.resize_handle == "bottomright":
                    self.resize_handle = "bottomleft"
                
            if y1 > y2:
                y1, y2 = y2, y1
                if self.resize_handle == "topleft":
                    self.resize_handle = "bottomleft"
                elif self.resize_handle == "topright":
                    self.resize_handle = "bottomright"
                elif self.resize_handle == "bottomleft":
                    self.resize_handle = "topleft"
                elif self.resize_handle == "bottomright":
                    self.resize_handle = "topright"
            
            self.boxes[self.current_box_id] = [x1, y1, x2, y2]
            self.update_box_display()
            
        elif self.moving and self.current_box_id is not None:
            # Handle moving a box
            dx = x - self.last_mouse_x
            dy = y - self.last_mouse_y
            
            box = self.boxes[self.current_box_id].copy()
            x1, y1, x2, y2 = box
            
            # Move the box
            x1 += dx
            y1 += dy
            x2 += dx
            y2 += dy
            
            # Keep the box within frame boundaries
            if x1 < 0:
                x2 -= x1
                x1 = 0
            if y1 < 0:
                y2 -= y1
                y1 = 0
            if x2 > self.frame.width:
                x1 -= (x2 - self.frame.width)
                x2 = self.frame.width
            if y2 > self.frame.height:
                y1 -= (y2 - self.frame.height)
                y2 = self.frame.height
                
            self.boxes[self.current_box_id] = [x1, y1, x2, y2]
            self.last_mouse_x = x
            self.last_mouse_y = y
            self.update_box_display()
            
        elif self.drawing:
            # Drawing a new box
            self.update_box_display(temp_box=[self.start_x, self.start_y, event.x, event.y])

    def on_box_release(self, event):
        if self.resizing or self.moving:
            # Reset resizing and moving flags
            self.resizing = False
            self.moving = False
            self.resize_handle = None
            self.last_mouse_x = None
            self.last_mouse_y = None
            
        elif self.drawing:
            # User released mouse – let SAM-2 segment around that point
            sam_mask = self.master.sam_segment_from_point(event.x, event.y, self.frame)
            if sam_mask is None:
                # Fallback: keep the hand-drawn rectangle you already had
                sam_mask = np.zeros((self.frame.height, self.frame.width), np.uint8)
                cv2.rectangle(sam_mask, (self.start_x, self.start_y),
                            (event.x, event.y), 1, -1)

            bbs = mask_to_bboxes(sam_mask)
            for bb in bbs:
                self.boxes.append(bb)
                self.class_indices.append(self.class_var.get())

            self.current_box_id = len(self.boxes) - 1
            self.update_box_display()
            self.drawing = False
            self.start_x = None
            self.start_y = None

    def delete_selected_box(self):
        """Delete the selected box and remove it from tracking"""
        if self.current_box_id is not None and 0 <= self.current_box_id < len(self.boxes):
            # Get the box that's being deleted
            deleted_box = self.boxes[self.current_box_id]
            
            # Remove it from current boxes
            del self.boxes[self.current_box_id]
            del self.class_indices[self.current_box_id]
            
            # After deleting, need to completely reset tracker with remaining boxes
            # This ensures the deleted box is no longer tracked
            if hasattr(self, 'csrt_tracker') and self.csrt_tracker is not None and self.boxes:
                try:
                    # Reinitialize tracker with only the remaining boxes
                    self.csrt_tracker.initialize_trackers(self.frame, self.boxes, self.class_indices)
                    logger.info(f"Reinitialized tracker after deleting box")
                except Exception as e:
                    logger.error(f"Error reinitializing tracker after deletion: {e}")
            
            # Reset current box selection
            self.current_box_id = None
            self.update_box_display()
            
            # Show feedback
            self.tracking_status.config(text=f"Deleted box and reset tracking")
            self.root.update()
            self.root.after(1500, lambda: self.tracking_status.config(text=""))  # Clear after 1.5 seconds

    def clear_all_boxes(self):
        """Clear all boxes and reset tracking"""
        if messagebox.askyesno("Clear All", "Are you sure you want to clear all boxes?"):
            self.boxes = []
            self.class_indices = []
            self.current_box_id = None
            
            # Reset tracker since all boxes are gone
            if self.tracker_type == "SAM" and hasattr(self, 'sam_tracker'):
                self.sam_tracker = SAMBoxTracker(self.master.sam_predictor, self.master.device)
                logger.info("Reset SAM tracker after clearing all boxes")
            elif hasattr(self, 'csrt_tracker'):
                self.csrt_tracker = CSRTObjectTracker()
                logger.info("Reset CSRT tracker after clearing all boxes")
            
            self.update_box_display()
            
            # Show feedback
            self.tracking_status.config(text=f"Cleared all boxes and reset tracking")
            self.root.update()
            self.root.after(1500, lambda: self.tracking_status.config(text=""))  # Clear after 1.5 seconds

    # Helper function for drawing dashed rectangle
    def draw_dashed_rectangle(self, draw, box, color, width=2, dash_length=5, gap_length=5):
        x1, y1, x2, y2 = box
        # Draw top line
        for i in range(x1, x2, dash_length + gap_length):
            end = min(i + dash_length, x2)
            draw.line([(i, y1), (end, y1)], fill=color, width=width)
        # Draw right line
        for i in range(y1, y2, dash_length + gap_length):
            end = min(i + dash_length, y2)
            draw.line([(x2, i), (x2, end)], fill=color, width=width)
        # Draw bottom line
        for i in range(x1, x2, dash_length + gap_length):
            end = min(i + dash_length, x2)
            draw.line([(i, y2), (end, y2)], fill=color, width=width)
        # Draw left line
        for i in range(y1, y2, dash_length + gap_length):
            end = min(i + dash_length, y2)
            draw.line([(x1, i), (x1, end)], fill=color, width=width)
    
    def draw_resize_handle(self, draw, x, y, color):
        """Draw a small square handle for resizing"""
        size = self.resize_handle_size
        # Draw outline instead of filled square for less obstruction
        # Add light transparent fill with outline for better visibility but less obstruction
        draw.rectangle([x-size, y-size, x+size, y+size], 
                      fill=(color[0], color[1], color[2], 80),  # Semi-transparent fill
                      outline=color, width=1)

    def update_box_display(self, temp_box=None):
        display = self.frame.copy()
        draw = ImageDraw.Draw(display)
        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
        ]
        for i, (box, cls_idx) in enumerate(zip(self.boxes, self.class_indices)):
            x1, y1, x2, y2 = box
            color = colors[cls_idx % len(colors)]
            width = 3 if i == self.current_box_id else 2
            draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=width)
            
            # Draw resize handles for selected box
            if i == self.current_box_id:
                self.draw_resize_handle(draw, x1, y1, color)  # Top-left
                self.draw_resize_handle(draw, x2, y1, color)  # Top-right
                self.draw_resize_handle(draw, x1, y2, color)  # Bottom-left
                self.draw_resize_handle(draw, x2, y2, color)  # Bottom-right
            
            try:
                font = ImageFont.truetype("arial.ttf", 15)
            except:
                font = ImageFont.load_default()
            class_names = ["Obstacle", "Goal", "Other"]
            label_text = f"{i}: {class_names[cls_idx]}"
            text_size = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_size[2] - text_size[0]
            text_height = text_size[3] - text_size[1]
            draw.rectangle([(x1, y1-text_height-2), (x1+text_width+4, y1)], fill=(255, 255, 255, 180))
            draw.text((x1+2, y1-text_height-2), label_text, fill=color, font=font)
        
        if temp_box:
            x1, y1, x2, y2 = temp_box
            # Use custom dashed rectangle function
            self.draw_dashed_rectangle(draw, [x1, y1, x2, y2], colors[self.class_var.get()], width=2)
        
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def convert_boxes_to_yolo_format(self, boxes, classes, img_width, img_height):
        """Convert bounding boxes from [x1, y1, x2, y2] format to YOLO format [cls, cx, cy, w, h]"""
        yolo_boxes = []
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = box
            x_center = (x1 + x2) / (2 * img_width)
            y_center = (y1 + y2) / (2 * img_height)
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            yolo_boxes.append([cls, x_center, y_center, width, height])
        return yolo_boxes


################################################################################
# VIDEO SPLITTING HELPERS
################################################################################
def split_video_into_segments(video_path, max_frames_per_segment=7000):
    """
    Split a video into temporary segments with max_frames_per_segment frames each.
    
    Args:
        video_path: Path to the input video
        max_frames_per_segment: Maximum number of frames per segment
        
    Returns:
        List of temporary segment file paths
    """
    video_path = Path(video_path)
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="video_segments_")
    logger.info(f"Created temporary directory for video segments: {temp_dir}")
    
    # Open input video
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]
    
    # Get video properties
    total_frames = int(video_stream.frames) if video_stream.frames else 1000
    fps = video_stream.average_rate
    width = video_stream.width
    height = video_stream.height
    
    logger.info(f"Splitting video with {total_frames} frames into segments of {max_frames_per_segment} frames")
    
    # Calculate number of segments
    num_segments = (total_frames + max_frames_per_segment - 1) // max_frames_per_segment
    
    segment_paths = []
    
    # Process each segment
    for segment_idx in range(num_segments):
        start_frame = segment_idx * max_frames_per_segment
        end_frame = min((segment_idx + 1) * max_frames_per_segment, total_frames)
        
        # Create output video file
        segment_path = Path(temp_dir) / f"segment_{segment_idx+1}_{start_frame}_{end_frame}.mp4"
        segment_paths.append(segment_path)
        
        # Reset input container position
        container.close()
        container = av.open(str(video_path))
        
        # Create output container
        output_container = av.open(str(segment_path), mode='w')
        output_stream = output_container.add_stream('h264', rate=fps)
        output_stream.width = width
        output_stream.height = height
        output_stream.pix_fmt = 'yuv420p'
        
        # Skip to approximate start position
        if start_frame > 0:
            target_ts = int(start_frame * 1000000 / fps)  # Convert to microseconds
            container.seek(target_ts, stream=video_stream)
        
        # Copy frames to output segment
        current_frame = 0
        frames_written = 0
        
        for frame in container.decode(video=0):
            # Skip frames until we reach the start frame
            if current_frame < start_frame:
                current_frame += 1
                continue
                
            # Stop if we've reached the end frame for this segment
            if current_frame >= end_frame:
                break
                
            # Encode and write the frame
            for packet in output_stream.encode(frame):
                output_container.mux(packet)
                
            frames_written += 1
            current_frame += 1
            
            # Log progress
            if frames_written % 500 == 0:
                logger.info(f"Segment {segment_idx+1}/{num_segments}: Wrote {frames_written}/{end_frame-start_frame} frames")
        
        # Flush the stream
        for packet in output_stream.encode(None):
            output_container.mux(packet)
            
        output_container.close()
        
        logger.info(f"Created segment {segment_idx+1}/{num_segments} with {frames_written} frames: {segment_path}")
    
    # Close input container
    container.close()
    
    return segment_paths, temp_dir


################################################################################
# BOUNDING BOX ANNOTATION SYSTEM
################################################################################
class BBoxAnnotationSystem:
    def __init__(self):
        self.manual_annotator = ManualAnnotationInterface()
        self.manual_annotator.master = self  # Give GUI access to SAM-2 helpers
        self.csrt_tracker = CSRTObjectTracker()  # Add CSRT tracker
        self.tracking_enabled = True  # Enable tracking by default
        self.tracking_reset_interval = 30  # Reset trackers every N frames to prevent drift
        self.tracker_type = "CSRT"  # Default tracker type: "CSRT" or "SAM"
        
        # Initialize SAM-2 teacher
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam_predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-large"
        )
        # Explicitly move the model to GPU if available
        if self.device == "cuda":
            self.sam_predictor.model.cuda()
        # placeholder buffers for future use
        self.sam_logits = None
        self.sam_conf = None
        
        # Initialize SAM box tracker (only when needed)
        self.sam_tracker = None
        
        self.coco_data = {
            "info": {
                "description": "Object Detection Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "BBox Annotation System",
                "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "categories": [
                {
                    "id": 0,
                    "name": "obstacle",
                    "supercategory": "object"
                },
                {
                    "id": 1,
                    "name": "goal",
                    "supercategory": "object"
                },
                {
                    "id": 2,
                    "name": "other",
                    "supercategory": "object"
                }
            ],
            "images": [],
            "annotations": []
        }
        self.annotation_id = 0
        self.image_id = 0
        self.annotations_by_frame = {}  # Store annotations by frame
    
    def sam_segment_from_point(self, x, y, pil_frame):
        """Returns binary mask, also stores logits & confidence for future use"""
        pt = torch.tensor([[x, y]], dtype=torch.float32, device=self.device)
        lbl = torch.tensor([1],  dtype=torch.int64,  device=self.device)
        # Use nullcontext on CPU since autocast only works with CUDA
        ctx = torch.autocast(self.device, dtype=torch.bfloat16) if self.device == "cuda" else nullcontext()
        with torch.inference_mode(), ctx:
            self.sam_predictor.set_image(pil_frame)
            m, s, logits = self.sam_predictor.predict(
                point_coords=pt, point_labels=lbl, multimask_output=True
            )
        if m is None or len(m) == 0:
            return None
        best = int(np.argmax(s))
        # Check if logits is a torch tensor or numpy array
        if isinstance(logits[best], torch.Tensor):
            self.sam_logits = logits[best].cpu()
        else:
            # Already a numpy array
            self.sam_logits = logits[best]
        self.sam_conf = float(s[best])
        return (m[best] > 0.5).astype(np.uint8)

    def get_keyframe_indices(self, total_frames, interval_percent=5):
        interval = max(1, int(total_frames * interval_percent / 100))
        return list(range(0, total_frames, interval))

    def process_video_segment(self, video_path, output_dir, segment_info=None):
        """
        Process a single video segment
        
        Args:
            video_path: Path to the video segment
            output_dir: Base output directory
            segment_info: Optional dictionary with segment information
                          (used for tracking progress in multi-segment processing)
        
        Returns:
            Dictionary with frame annotations
        """
        logger.info(f"Processing video segment: {video_path}")
        
        # Initialize dictionaries to store annotations
        boxes_by_frame = {}
        classes_by_frame = {}
        all_frames = []  # Store frame objects for visualization
        
        try:
            # Open the video file
            container = av.open(str(video_path))
            video_stream = container.streams.video[0]
            total_frames = int(video_stream.frames) if video_stream.frames else 1000
            fps = video_stream.average_rate
            width = video_stream.width
            height = video_stream.height
            
            # Initialize CSRT tracker
            self.csrt_tracker = CSRTObjectTracker()
            
            # Load all frames (for this segment)
            frames = []
            frame_count = 0
            
            for frame in container.decode(video=0):
                # Convert to PIL image
                pil_frame = frame.to_image()
                frames.append(pil_frame)
                all_frames.append((frame_count, pil_frame))
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Loaded {frame_count} frames...")
            
            # If segment_info is provided, include it in the message
            if segment_info:
                segment_msg = f" (Segment {segment_info['current']}/{segment_info['total']})"
            else:
                segment_msg = ""
            
            logger.info(f"Loaded {len(frames)} frames{segment_msg}")
            
            # Enter manual annotation interface
            if frames:
                logger.info(f"Starting annotation interface for segment{segment_msg}")
                current_frame_idx = 0  # Start at beginning
                
                boxes_by_frame, classes_by_frame, last_frame_idx = self.manual_annotator.manual_correction_interface(
                    frames, current_frame_idx, boxes_by_frame, classes_by_frame
                )
            
            container.close()
            
            # Return the annotations for this segment
            return {
                "boxes_by_frame": boxes_by_frame,
                "classes_by_frame": classes_by_frame,
                "all_frames": all_frames,
                "width": width,
                "height": height,
                "fps": fps
            }
            
        except Exception as e:
            logger.error(f"Error processing video segment {video_path}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_videos(self, video_files, output_dir, max_frames_per_segment=7000):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        for subdir in ['visualizations', 'labels', 'coco']:
            (output_path / subdir).mkdir(exist_ok=True)
        
        for video_idx, video_path in enumerate(video_files):
            logger.info(f"Processing video {video_idx+1}/{len(video_files)}: {video_path}")
            
            try:
                # Open the video file to get total frames
                container = av.open(str(video_path))
                video_stream = container.streams.video[0]
                total_frames = int(video_stream.frames) if video_stream.frames else 1000
                fps = video_stream.average_rate
                width = video_stream.width
                height = video_stream.height
                container.close()
                
                logger.info(f"Video has approximately {total_frames} frames")
                
                # Determine if we need to split the video
                if total_frames > max_frames_per_segment:
                    # Split video into segments
                    logger.info(f"Video exceeds {max_frames_per_segment} frames, splitting into segments")
                    segment_paths, temp_dir = split_video_into_segments(video_path, max_frames_per_segment)
                    
                    # Process each segment
                    all_frames = []
                    boxes_by_frame = {}
                    classes_by_frame = {}
                    
                    for i, segment_path in enumerate(segment_paths):
                        segment_info = {
                            "current": i + 1,
                            "total": len(segment_paths)
                        }
                        
                        # Process this segment
                        segment_result = self.process_video_segment(
                            segment_path, 
                            output_dir,
                            segment_info
                        )
                        
                        if segment_result:
                            # Extract frame offset from segment filename
                            segment_name = segment_path.name
                            start_frame = int(segment_name.split('_')[2])
                            
                            # Adjust frame indices to match original video
                            segment_boxes = segment_result["boxes_by_frame"]
                            segment_classes = segment_result["classes_by_frame"]
                            
                            # Add segment annotations to global dictionaries with adjusted indices
                            for frame_idx, boxes in segment_boxes.items():
                                global_idx = start_frame + frame_idx
                                boxes_by_frame[global_idx] = boxes
                                classes_by_frame[global_idx] = segment_classes[frame_idx]
                            
                            # Add frames to global list with adjusted indices
                            for rel_idx, frame in segment_result["all_frames"]:
                                global_idx = start_frame + rel_idx
                                all_frames.append((global_idx, frame))
                    
                    # Clean up temporary directory
                    try:
                        shutil.rmtree(temp_dir)
                        logger.info(f"Removed temporary directory: {temp_dir}")
                    except Exception as e:
                        logger.error(f"Error removing temp directory: {e}")
                        
                else:
                    # Process the video directly (no splitting needed)
                    logger.info(f"Processing video directly (under {max_frames_per_segment} frames)")
                    result = self.process_video_segment(video_path, output_dir)
                    
                    if result:
                        boxes_by_frame = result["boxes_by_frame"]
                        classes_by_frame = result["classes_by_frame"]
                        all_frames = result["all_frames"]
                    else:
                        logger.error(f"Failed to process video: {video_path}")
                        continue
                
                # Sort all frames by index to ensure they're in the right order
                all_frames.sort(key=lambda x: x[0])
                
                # Second pass: Generate output files
                logger.info("Creating output files...")
                
                # Create visualization video
                output_video_path = output_path / 'visualizations' / f"processed_video_{video_idx}.mp4"
                output_container = av.open(str(output_video_path), mode='w')
                output_stream = output_container.add_stream('h264', rate=fps)
                output_stream.width = width
                output_stream.height = height
                output_stream.pix_fmt = 'yuv420p'
                
                # Process each frame
                for frame_info in all_frames:
                    frame_idx, frame = frame_info
                    frame_id = f"video_{video_idx}_frame_{frame_idx}"
                    
                    # Add frame to COCO dataset
                    coco_image = {
                        "id": self.image_id,
                        "width": width,
                        "height": height,
                        "file_name": f"{frame_id}.jpg",
                        "license": 1,
                        "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    self.coco_data["images"].append(coco_image)
                    
                    # Get annotations for this frame
                    if frame_idx in boxes_by_frame:
                        boxes = boxes_by_frame[frame_idx]
                        classes = classes_by_frame[frame_idx]
                    else:
                        boxes = []
                        classes = []
                    
                    # Save the annotations to a YOLO format text file
                    txt_lines = bboxes_to_yolo(boxes, width, height, classes)
                    (output_path / 'labels' / f"{frame_id}.txt").write_text("\n".join(txt_lines))
                    
                    # Add object annotations to COCO
                    for box, cls in zip(boxes, classes):
                        x1, y1, x2, y2 = box
                        w, h = x2 - x1, y2 - y1
                        object_annotation = {
                            "id": self.annotation_id,
                            "image_id": self.image_id,
                            "category_id": cls,
                            "bbox": [float(x1), float(y1), float(w), float(h)],
                            "area": float(w * h),
                            "iscrowd": 0
                        }
                        self.coco_data["annotations"].append(object_annotation)
                        self.annotation_id += 1
                    
                    # Create visualization frame
                    vis_img = frame.copy()
                    draw = ImageDraw.Draw(vis_img)
                    
                    # Add object boxes to visualization
                    colors = [
                        (255, 0, 0),
                        (0, 255, 0),
                        (0, 0, 255),
                    ]
                    for box, cls in zip(boxes, classes):
                        x1, y1, x2, y2 = box
                        color = colors[cls % len(colors)]
                        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=2)
                        try:
                            font = ImageFont.truetype("arial.ttf", 12)
                        except:
                            font = ImageFont.load_default()
                        class_names = ["Obstacle", "Goal", "Other"]
                        draw.text((x1, y1-15), class_names[cls], fill=color, font=font)
                    
                    # Add frame status to visualization
                    try:
                        status_font = ImageFont.truetype("arial.ttf", 14)
                    except:
                        status_font = ImageFont.load_default()
                    
                    draw.text((10, 10), f"FRAME {frame_idx+1}/{total_frames}", fill=(255, 255, 0), font=status_font)
                    
                    # Write visualization frame to video
                    output_frame = av.VideoFrame.from_image(vis_img)
                    for packet in output_stream.encode(output_frame):
                        output_container.mux(packet)
                    
                    self.image_id += 1
                    
                    if frame_idx % 10 == 0:
                        logger.info(f"Processed {frame_idx+1}/{total_frames} frames...")
                
                # Finalize video
                for packet in output_stream.encode(None):
                    output_container.mux(packet)
                output_container.close()
                logger.info(f"Completed video {video_idx+1} - Processed {len(all_frames)} frames")
                
            except Exception as e:
                logger.error(f"Error processing video {video_path}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save COCO annotations
        coco_json_path = output_path / 'coco' / 'annotations.json'
        with open(coco_json_path, 'w') as f:
            json.dump(self.coco_data, f)
        logger.info(f"Saved COCO annotations to {coco_json_path}")
        logger.info(f"Total images: {self.image_id}")
        logger.info(f"Total annotations: {self.annotation_id}")
        return self.image_id, self.annotation_id

    def main(self):
        import argparse
        parser = argparse.ArgumentParser(description="Bounding Box Annotation with SAM2 and CSRT Tracking")
        parser.add_argument("--interval", type=int, default=5, help="Interval percentage for keyframes (default: 5%%)")
        parser.add_argument("--tracking_reset", type=int, default=30, 
                           help="Number of frames before resetting trackers (default: 30)")
        parser.add_argument("--no_tracking", action="store_true", 
                           help="Disable tracking and use manual annotation for all frames")
        parser.add_argument("--max_frames", type=int, default=7000,
                           help="Maximum number of frames per video segment (default: 7000)")
        parser.add_argument("--tracker", type=str, choices=["CSRT", "SAM"], default="CSRT",
                           help="Tracking method to use: CSRT (faster) or SAM (more accurate)")
        
        args = parser.parse_args()
        
        # Apply settings
        self.tracking_reset_interval = args.tracking_reset
        self.tracking_enabled = not args.no_tracking
        self.tracker_type = args.tracker
        max_frames_per_segment = args.max_frames
        
        # Process videos
        dataset_dir = Path("dataset")
        video_files = list(dataset_dir.glob("*.mp4")) + list(dataset_dir.glob("*.avi"))
        if not video_files:
            logger.error("No videos found in dataset/ directory.")
            return
            
        output_dir = Path("bbox_annotation_data")
        output_dir.mkdir(exist_ok=True)
        
        # Log configuration
        logger.info(f"Starting bounding box annotation with:")
        logger.info(f"  SAM2 model: facebook/sam2-hiera-large")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Tracking enabled: {self.tracking_enabled}")
        logger.info(f"  Tracker type: {self.tracker_type}")
        logger.info(f"  Tracker reset interval: {self.tracking_reset_interval} frames")
        logger.info(f"  Keyframe interval: {args.interval}%")
        logger.info(f"  Maximum frames per segment: {max_frames_per_segment}")
        
        # Process videos
        num_images, num_annotations = self.process_videos(
            video_files, 
            output_dir, 
            max_frames_per_segment=max_frames_per_segment
        )
        
        logger.info("Bounding box annotation complete")
        logger.info(f"Total images: {num_images}")
        logger.info(f"Total annotations: {num_annotations}")
        logger.info(f"Output saved to {output_dir}")


################################################################################
# MAIN FUNCTION
################################################################################
if __name__ == "__main__":
    annotation_system = BBoxAnnotationSystem()
    annotation_system.main()
