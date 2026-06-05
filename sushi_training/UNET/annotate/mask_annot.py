import os
import numpy as np
import torch
import cv2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor
from PIL import Image, ImageTk, ImageDraw, ImageFont
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from pathlib import Path
import av
import logging
import warnings
import time
from scipy import ndimage
import argparse
from pycocotools import mask as mask_util
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.getLogger('root').setLevel(logging.WARNING)
warnings.filterwarnings('ignore', category=UserWarning)

################################################################################
# SAM2 MASK ANNOTATOR
################################################################################
class SAM2Annotator:
    def __init__(self):
        logger.info("Initializing SAM2 model...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large")
        logger.info("Model initialization complete")
        
        # Annotation states
        self.frame = None
        self.water_mask = None
        self.water_mask_dict = {}  # Dictionary to store masks for keyframes {frame_id: mask}
        self.keyframe_ids = []     # List to track keyframe IDs in order
        
        # Drawing states
        self.start_x = None
        self.start_y = None
        self.current_box = None
        self.current_mask = None
        self.bbox_history = []
        self.WATER_COLOR = (0, 0, 255)  # Blue for water

        # Locked mask: Once fill holes is applied, that mask remains permanent.
        self.locked_mask = None

        # UI elements
        self.root = None
        self.canvas = None
        self.photo = None
        self.frame_label = None
        self.keyframe_listbox = None
        self.annotation_mode = "bbox"  # Default to bounding box annotation

    def clamp_bbox(self, bbox, img_width, img_height):
        """Clamp bbox coordinates to be within the image dimensions."""
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        return [x1, y1, x2, y2]

    def get_mask_from_bbox(self, frame_pil, bbox):
        """Get a mask from a bounding box using SAM2"""
        if not bbox:
            return None
        
        bbox = self.clamp_bbox(bbox, frame_pil.width, frame_pil.height)
        x1, y1, x2, y2 = bbox
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.predictor.set_image(frame_pil)
            sam_bbox = np.array([x1, y1, x2 - x1, y2 - y1])
            
            masks, scores, embeddings = self.predictor.predict(
                box=torch.tensor(sam_bbox, device=self.device).unsqueeze(0),
                multimask_output=True
            )
            
            if masks is not None and len(masks) > 0:
                best_mask_idx = np.argmax(scores) if isinstance(scores, (list, np.ndarray)) else 0
                mask = masks[best_mask_idx]
                # If a locked mask exists, merge it so that it remains permanent.
                if self.locked_mask is not None:
                    mask = np.logical_or(self.locked_mask, mask).astype(np.uint8)
                return mask
                
        return None

    def get_mask_from_points(self, frame_pil, points, point_labels=None):
        """Get a mask from a set of points using SAM2"""
        if not points:
            return None
            
        pts = np.array(points)
        point_coords = torch.tensor(pts, dtype=torch.float32, device=self.device)
        if point_labels is None:
            point_labels = np.ones(len(pts), dtype=np.int64)
        point_labels = torch.tensor(point_labels, dtype=torch.int64, device=self.device)
        
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.predictor.set_image(frame_pil)
            
            masks, scores, embeddings = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True
            )
            
            if masks is not None and len(masks) > 0:
                best_mask_idx = np.argmax(scores) if isinstance(scores, (list, np.ndarray)) else 0
                mask = masks[best_mask_idx]
                # Merge with locked mask if it exists
                if self.locked_mask is not None:
                    mask = np.logical_or(self.locked_mask, mask).astype(np.uint8)
                return mask
                
        return None

    def compute_iou(self, mask1, mask2):
        """Compute IoU between two binary masks"""
        if mask1 is None or mask2 is None:
            return 0.0
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        return intersection / (union + 1e-6)

    def postprocess_mask(self, mask, min_area=500):
        """
        Postprocess the mask by removing small isolated regions and filling holes.
        This method applies both operations in a single step.
        """
        mask = (mask > 0).astype(np.uint8)
        mask = self.remove_small_blobs(mask, min_area)
        mask = self.fill_holes(mask)
        return mask

    def remove_small_blobs(self, mask, min_area=500):
        """
        Remove small disconnected regions (blobs) from the binary mask.
        """
        # Convert mask to uint8 to ensure compatibility with OpenCV
        mask_uint8 = (mask > 0).astype(np.uint8)
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
        cleaned_mask = np.zeros_like(mask_uint8)
        for i in range(1, num_labels):  # Skip background
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_mask[labels == i] = 1
        return cleaned_mask

    def fill_holes(self, mask):
        """
        Fill holes in the binary mask.
        """
        filled = ndimage.binary_fill_holes(mask).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        processed = cv2.morphologyEx(filled, cv2.MORPH_OPEN, kernel)
        return processed

    def apply_remove_blobs(self):
        """Callback to remove small blobs from the current mask and update the display."""
        if self.current_mask is not None:
            self.current_mask = self.remove_small_blobs(self.current_mask)
            self.update_display()
        else:
            messagebox.showwarning("No Mask", "No mask is available to process.")

    def apply_fill_holes(self):
        """Callback to fill holes in the current mask, lock that region permanently, and update the display."""
        if self.current_mask is not None:
            filled = self.fill_holes(self.current_mask)
            # Lock the filled region so that further annotations preserve it.
            self.locked_mask = filled.copy()
            self.current_mask = filled.copy()
            self.update_display()
        else:
            messagebox.showwarning("No Mask", "No mask is available to process.")

    def annotate_keyframe(self, frame_pil, frame_id):
        """Interactive annotation interface for a keyframe"""
        self.frame = frame_pil
        self.frame_id = frame_id
        self.current_box = None
        self.current_mask = None
        self.bbox_history = []
        self.points = []
        self.point_labels = []
        
        # Create annotation window
        self.root = tk.Tk()
        self.root.title(f"SAM Annotation - Frame {frame_id}")
        
        # Create main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create canvas for drawing
        self.canvas = tk.Canvas(main_frame, width=frame_pil.width, height=frame_pil.height)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Create side panel
        panel = ttk.Frame(main_frame, padding=10)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Annotation mode selection
        ttk.Label(panel, text="Annotation Mode:").pack(pady=5, anchor=tk.W)
        self.mode_var = tk.StringVar(value="bbox")
        ttk.Radiobutton(panel, text="Bounding Box", variable=self.mode_var, 
                        value="bbox", command=self.update_mode).pack(anchor=tk.W)
        ttk.Radiobutton(panel, text="Point Prompts", variable=self.mode_var, 
                        value="points", command=self.update_mode).pack(anchor=tk.W)
        
        # Add point type selection for point mode
        self.point_frame = ttk.LabelFrame(panel, text="Point Type")
        self.point_frame.pack(pady=10, fill=tk.X, anchor=tk.W)
        self.point_type = tk.IntVar(value=1)
        ttk.Radiobutton(self.point_frame, text="Include (Water)", variable=self.point_type, value=1).pack(anchor=tk.W)
        ttk.Radiobutton(self.point_frame, text="Exclude (Not Water)", variable=self.point_type, value=0).pack(anchor=tk.W)
        
        # Show/hide point frame based on mode
        if self.mode_var.get() != "points":
            self.point_frame.pack_forget()
        
        # Action buttons
        button_frame = ttk.Frame(panel)
        button_frame.pack(pady=20, fill=tk.X)
        ttk.Button(button_frame, text="Undo Last", command=self.undo_last).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Clear All", command=self.clear_all).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Save", command=self.save_mask).pack(side=tk.RIGHT, padx=2)
        
        # --- New Post-Process Buttons ---
        ttk.Label(panel, text="Post-process Mask:").pack(pady=(10, 5), anchor=tk.W)
        ttk.Button(panel, text="Remove Small Blobs", command=self.apply_remove_blobs).pack(fill=tk.X, pady=2)
        ttk.Button(panel, text="Fill Holes (Lock Region)", command=self.apply_fill_holes).pack(fill=tk.X, pady=2)
        # --- End of New Buttons ---
        
        # Completion buttons
        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Button(panel, text="Done", command=self.finalize_annotation).pack(fill=tk.X, pady=5)
        ttk.Button(panel, text="Cancel", command=self.cancel_annotation).pack(fill=tk.X, pady=5)
        
        # Previously annotated frames
        if self.keyframe_ids:
            ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
            ttk.Label(panel, text="Previously Annotated Frames:").pack(pady=5, anchor=tk.W)
            self.keyframe_listbox = tk.Listbox(panel, height=5)
            self.keyframe_listbox.pack(fill=tk.X, pady=5)
            for kf_id in self.keyframe_ids:
                self.keyframe_listbox.insert(tk.END, kf_id)
        
        # Canvas events
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        
        # Key bindings
        self.root.bind("<Escape>", lambda e: self.cancel_annotation())
        self.root.bind("<Return>", lambda e: self.finalize_annotation())
        
        # Update display
        self.update_display()
        
        # Start main loop
        self.root.mainloop()
        
        # Return the final mask or None if canceled
        if self.root:
            self.root.destroy()
        
        return self.current_mask

    def update_mode(self):
        """Update the annotation mode"""
        new_mode = self.mode_var.get()
        if new_mode != self.annotation_mode:
            self.annotation_mode = new_mode
            if self.annotation_mode == "points":
                self.point_frame.pack(pady=10, fill=tk.X, anchor=tk.W)
            else:
                self.point_frame.pack_forget()
            self.clear_all()

    def on_mouse_down(self, event):
        """Handle mouse button press"""
        x, y = event.x, event.y
        if self.annotation_mode == "bbox":
            self.start_x = x
            self.start_y = y
            self.current_box = [x, y, x, y]
        elif self.annotation_mode == "points":
            self.points.append((x, y))
            self.point_labels.append(self.point_type.get())
            # Generate the mask from points and then merge with the locked region (if exists)
            new_mask = self.get_mask_from_points(self.frame, self.points, self.point_labels)
            if self.locked_mask is not None and new_mask is not None:
                self.current_mask = np.logical_or(self.locked_mask, new_mask).astype(np.uint8)
            else:
                self.current_mask = new_mask
        self.update_display()

    def on_mouse_move(self, event):
        """Handle mouse movement while button pressed"""
        if self.annotation_mode == "bbox" and self.start_x is not None:
            self.current_box = [self.start_x, self.start_y, event.x, event.y]
            self.update_display()

    def on_mouse_up(self, event):
        """Handle mouse button release"""
        if self.annotation_mode == "bbox" and self.start_x is not None:
            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                self.current_box = [x1, y1, x2, y2]
                self.bbox_history.append(self.current_box)
                new_mask = self.get_mask_from_bbox(self.frame, self.current_box)
                # If there's a locked mask, merge it with the new mask.
                if self.locked_mask is not None and new_mask is not None:
                    self.current_mask = np.logical_or(self.locked_mask, new_mask).astype(np.uint8)
                else:
                    self.current_mask = new_mask
            self.start_x = None
            self.start_y = None
            self.update_display()

    def undo_last(self):
        """Remove the last annotation"""
        if self.annotation_mode == "bbox" and self.bbox_history:
            self.bbox_history.pop()
            if self.bbox_history:
                self.current_box = self.bbox_history[-1]
                new_mask = self.get_mask_from_bbox(self.frame, self.current_box)
                if self.locked_mask is not None and new_mask is not None:
                    self.current_mask = np.logical_or(self.locked_mask, new_mask).astype(np.uint8)
                else:
                    self.current_mask = new_mask
            else:
                self.current_box = None
                self.current_mask = None
        elif self.annotation_mode == "points" and self.points:
            self.points.pop()
            self.point_labels.pop()
            if self.points:
                new_mask = self.get_mask_from_points(self.frame, self.points, self.point_labels)
                if self.locked_mask is not None and new_mask is not None:
                    self.current_mask = np.logical_or(self.locked_mask, new_mask).astype(np.uint8)
                else:
                    self.current_mask = new_mask
            else:
                self.current_mask = None
        self.update_display()

    def clear_all(self):
        """Clear all non-locked annotations but preserve the locked region if it exists."""
        if self.annotation_mode == "bbox":
            self.bbox_history = []
            self.current_box = None
        elif self.annotation_mode == "points":
            self.points = []
            self.point_labels = []
        # Do not clear self.locked_mask to preserve the region.
        # Clear only the non-locked part.
        if self.locked_mask is not None:
            self.current_mask = self.locked_mask.copy()
        else:
            self.current_mask = None
        self.update_display()

    def save_mask(self):
        """Save the current mask to the water mask dictionary after postprocessing."""
        if self.current_mask is not None:
            processed_mask = self.postprocess_mask(self.current_mask)
            self.water_mask_dict[self.frame_id] = processed_mask.copy()
            if self.frame_id not in self.keyframe_ids:
                self.keyframe_ids.append(self.frame_id)
                if self.keyframe_listbox:
                    self.keyframe_listbox.insert(tk.END, self.frame_id)
            messagebox.showinfo("Saved", f"Mask for frame {self.frame_id} saved successfully")
        else:
            messagebox.showwarning("No Mask", "No valid mask to save. Please create a mask first.")

    def finalize_annotation(self):
        """Accept the current mask (after postprocessing) and return it."""
        if self.current_mask is not None:
            processed_mask = self.postprocess_mask(self.current_mask)
            self.water_mask = processed_mask.copy()
            self.water_mask_dict[self.frame_id] = processed_mask.copy()
            if self.frame_id not in self.keyframe_ids:
                self.keyframe_ids.append(self.frame_id)
            self.root.quit()
        else:
            messagebox.showwarning("No Mask", "No valid mask to finalize. Please create a mask first.")

    def cancel_annotation(self):
        """Cancel annotation without saving."""
        if messagebox.askyesno("Cancel", "Discard current annotations?"):
            self.current_mask = None
            self.water_mask = None
            self.root.quit()

    def update_display(self):
        """Update the annotation display."""
        display = self.frame.copy()
        draw = ImageDraw.Draw(display)
        if self.current_mask is not None:
            overlay = Image.new('RGBA', display.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            mask_np = self.current_mask.astype(np.uint8) * 255
            mask_img = Image.fromarray(mask_np, 'L')
            overlay_draw.bitmap((0, 0), mask_img, fill=(*self.WATER_COLOR, 76))
            display = Image.alpha_composite(display.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(display)
        if self.annotation_mode == "bbox" and self.current_box:
            x1, y1, x2, y2 = self.current_box
            draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)
        if self.annotation_mode == "points":
            for (x, y), label in zip(self.points, self.point_labels):
                color = self.WATER_COLOR if label == 1 else (255, 0, 0)
                draw.ellipse([x-5, y-5, x+5, y+5], fill=color)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            font = ImageFont.load_default()
        if self.annotation_mode == "bbox":
            draw.text((10, 10), "Drag to create bounding box for water", fill=(255, 255, 255), font=font)
        else:
            draw.text((10, 10), "Click to add points (Blue=Water, Red=Not Water)", fill=(255, 255, 255), font=font)
        self.photo = ImageTk.PhotoImage(display)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)


################################################################################
# SAM2 VIDEO PROPAGATOR (New)
################################################################################
class SAM2VideoPropagator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing SAM2 Video Predictor on {self.device}...")
        
        # These paths should point to your SAM2 model files
        self.checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
        self.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        
        # Check if model files exist
        if not Path(self.checkpoint).exists() or not Path(self.model_cfg).exists():
            logger.warning(f"SAM2 model files not found at {self.checkpoint} or {self.model_cfg}")
            logger.warning("Will use fallback tracking method")
            self.video_predictor = None
        else:
            # Initialize the video predictor
            try:
                self.video_predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint)
                logger.info("SAM2 Video Predictor initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize SAM2 Video Predictor: {e}")
                self.video_predictor = None
    
    def propagate_masks(self, video_frames, keyframe_indices, keyframe_masks):
        """
        Propagate masks through video frames using SAM2 video predictor
        
        Args:
            video_frames: List of PIL images for each frame
            keyframe_indices: List of frame indices with annotated masks
            keyframe_masks: Dictionary mapping keyframe indices to masks
            
        Returns:
            Dictionary mapping all frame indices to propagated masks
        """
        if self.video_predictor is None:
            logger.warning("SAM2 Video Predictor not available. Cannot propagate masks.")
            return keyframe_masks  # Return only keyframe masks if predictor not available
            
        all_frame_masks = {}
        
        try:
            logger.info("Starting mask propagation through video...")
            with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
                # Convert video frames to tensor format expected by SAM2
                video_tensor = self._frames_to_tensor(video_frames)
                
                # Initialize the state with video
                state = self.video_predictor.init_state(video_tensor)
                
                # For each keyframe, add its prompts (using box or mask)
                for kf_idx in keyframe_indices:
                    if kf_idx not in keyframe_masks:
                        continue
                        
                    mask = keyframe_masks[kf_idx]
                    
                    # If mask is just a binary mask, generate box prompt from it
                    bbox = self._mask_to_bbox(mask)
                    
                    # Add the box prompt for this keyframe
                    logger.info(f"Adding prompt for keyframe {kf_idx}")
                    frame_idx, object_ids, masks = self.video_predictor.add_new_box(
                        state, 
                        frame_idx=kf_idx,
                        boxes=torch.tensor([bbox], device=self.device)
                    )
                    
                    # Store the keyframe mask (use the original annotated mask)
                    all_frame_masks[kf_idx] = keyframe_masks[kf_idx]
                
                # Propagate prompts through the entire video
                logger.info("Propagating masks through video...")
                for frame_idx, object_ids, masks in self.video_predictor.propagate_in_video(state):
                    if frame_idx in keyframe_indices:
                        # Skip keyframes since we already have annotated masks for them
                        continue
                        
                    # masks is shape [N, H, W] where N is number of objects
                    # For water segmentation, we assume just one object (water)
                    if masks.shape[0] > 0:
                        # Convert mask from tensor to numpy
                        mask = masks[0].cpu().numpy()
                        all_frame_masks[frame_idx] = mask
            
            logger.info(f"Successfully propagated masks to {len(all_frame_masks)} frames")
            return all_frame_masks
            
        except Exception as e:
            logger.error(f"Error during mask propagation: {e}")
            import traceback
            traceback.print_exc()
            return keyframe_masks  # Return only the keyframe masks in case of error
    
    def _frames_to_tensor(self, frames):
        """Convert list of PIL images to tensor format expected by SAM2"""
        # Convert frames to format expected by SAM2
        # This might need adjustment based on exact requirements
        tensors = []
        for frame in frames:
            # Convert PIL image to numpy
            frame_np = np.array(frame)
            # Convert to tensor
            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1).float().to(self.device)
            tensors.append(frame_tensor)
        
        # Stack frames into a single tensor [T, C, H, W]
        return torch.stack(tensors)
    
    def _mask_to_bbox(self, mask):
        """Convert binary mask to bounding box in format expected by SAM2"""
        # Find non-zero locations
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            # Empty mask, return a dummy box
            return [0, 0, 10, 10]
            
        # Get bounding box coordinates
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        # Return as [x, y, w, h]
        return [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]


################################################################################
# MASK ANNOTATION SYSTEM
################################################################################
class MaskAnnotationSystem:
    def __init__(self):
        self.sam_annotator = SAM2Annotator()
        self.sam_video_propagator = SAM2VideoPropagator()
        self.key_masks = {}  # {frame_idx: mask}
        self.coco_data = {
            "info": {
                "description": "Water Detection Dataset",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Mask Annotation System",
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
                    "id": 3,
                    "name": "water",
                    "supercategory": "region"
                }
            ],
            "images": [],
            "annotations": []
        }
        self.annotation_id = 0
        self.image_id = 0

    def get_keyframe_indices(self, total_frames, interval_percent=5):
        interval = max(1, int(total_frames * interval_percent / 100))
        return list(range(0, total_frames, interval))

    def mask_to_bbox(self, binary_mask):
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            return [0, 0, 0, 0]
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return [float(cmin), float(rmin), float(cmax - cmin + 1), float(rmax - rmin + 1)]

    def process_videos(self, video_files, output_dir):
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        for subdir in ['visualizations', 'masks', 'coco']:
            (output_path / subdir).mkdir(exist_ok=True)
            
        for video_idx, video_path in enumerate(video_files):
            logger.info(f"Processing video {video_idx+1}/{len(video_files)}: {video_path}")
            try:
                container = av.open(str(video_path))
                video_stream = container.streams.video[0]
                total_frames = int(video_stream.frames) if video_stream.frames else 1000
                fps = video_stream.average_rate
                width = video_stream.width
                height = video_stream.height
                keyframe_indices = self.get_keyframe_indices(total_frames)
                logger.info(f"Will annotate {len(keyframe_indices)} keyframes out of {total_frames} total frames")
                
                output_video_path = output_path / 'visualizations' / f"processed_video_{video_idx}.mp4"
                output_container = av.open(str(output_video_path), mode='w')
                output_stream = output_container.add_stream('h264', rate=fps)
                output_stream.width = width
                output_stream.height = height
                output_stream.pix_fmt = 'yuv420p'
                
                keyframes = {}
                keyframe_masks = {}
                keyframe_save_path = output_path / 'keyframes' / f"video_{video_idx}_keyframes.json"
                (output_path / 'keyframes').mkdir(exist_ok=True)
                
                # Initialize all_frame_masks to contain at least keyframe masks
                all_frame_masks = {}
                
                # Load previously saved keyframes if available
                if keyframe_save_path.exists():
                    try:
                        logger.info(f"Loading saved keyframes from {keyframe_save_path}")
                        keyframe_data = json.loads(keyframe_save_path.read_text())
                        for k, v in keyframe_data.items():
                            frame_idx = int(k)
                            mask_path = Path(v)
                            if mask_path.exists():
                                mask = np.array(Image.open(mask_path)) > 128
                                keyframe_masks[frame_idx] = mask
                                # Also add to all_frame_masks
                                all_frame_masks[frame_idx] = mask
                                logger.info(f"Loaded mask for keyframe {frame_idx}")
                        
                        # If we're using saved keyframes, we need to propagate masks for all frames
                        if keyframe_masks and self.sam_video_propagator.video_predictor is not None:
                            logger.info("Collecting all frames for video propagation...")
                            container.seek(0)
                            all_frames = []
                            for frame in container.decode(video=0):
                                all_frames.append(frame.to_image())
                                
                            # Convert frame indices to int
                            keyframe_idx_list = sorted(list(keyframe_masks.keys()))
                            
                            # Propagate masks using saved keyframes
                            logger.info("Propagating masks using saved keyframes...")
                            all_frame_masks = self.sam_video_propagator.propagate_masks(
                                all_frames, 
                                keyframe_idx_list, 
                                keyframe_masks
                            )
                            
                            # Reopen container for the final processing pass
                            container.seek(0)
                    except Exception as e:
                        logger.error(f"Error loading keyframes: {e}")
                        keyframe_masks = {}
                        all_frame_masks = {}
                
                # If no keyframes loaded, manually annotate them
                if not keyframe_masks:
                    logger.info("Annotating keyframes manually")
                    # First, collect all keyframes
                    frame_count = 0
                    all_frames = []  # Store all frames for video propagation
                    
                    for frame in container.decode(video=0):
                        pil_frame = frame.to_image()
                        all_frames.append(pil_frame)
                        
                        if frame_count in keyframe_indices:
                            keyframes[frame_count] = pil_frame
                        
                        frame_count += 1
                        if frame_count % 100 == 0:
                            logger.info(f"Collected {len(keyframes)}/{len(keyframe_indices)} keyframes...")
                    
                    # Now annotate each keyframe
                    keyframe_save_data = {}
                    for idx, (frame_idx, pil_frame) in enumerate(keyframes.items()):
                        logger.info(f"Annotating keyframe {idx+1}/{len(keyframes)} (frame {frame_idx})...")
                        water_mask = self.sam_annotator.annotate_keyframe(pil_frame, f"{video_idx}_{frame_idx}")
                        if water_mask is not None:
                            processed_mask = self.sam_annotator.postprocess_mask(water_mask)
                            keyframe_masks[frame_idx] = processed_mask
                            mask_path = output_path / 'keyframes' / f"video_{video_idx}_frame_{frame_idx}_mask.png"
                            Image.fromarray((processed_mask * 255).astype(np.uint8)).save(mask_path)
                            
                            keyframe_save_data[str(frame_idx)] = str(mask_path)
                            with open(keyframe_save_path, 'w') as f:
                                json.dump(keyframe_save_data, f)
                    
                    # Now propagate masks using SAM2 video predictor
                    logger.info("Propagating masks to all frames using SAM2 video predictor...")
                    if self.sam_video_propagator.video_predictor is not None:
                        # Convert frame indices to int
                        keyframe_idx_list = sorted([int(idx) for idx in keyframe_masks.keys()])
                        
                        # Propagate masks
                        all_frame_masks = self.sam_video_propagator.propagate_masks(
                            all_frames, 
                            keyframe_idx_list, 
                            keyframe_masks
                        )
                    else:
                        logger.warning("SAM2 Video Predictor not available. Only keyframe masks will be used.")
                        all_frame_masks = keyframe_masks
                    
                    # Reopen container for processing all frames with propagated masks
                    container.close()
                    container = av.open(str(video_path))
                
                # Process all frames with the propagated masks
                frame_count = 0
                
                for frame in container.decode(video=0):
                    try:
                        pil_frame = frame.to_image()
                        frame_id = f"video_{video_idx}_frame_{frame_count}"
                        coco_image = {
                            "id": self.image_id,
                            "width": width,
                            "height": height,
                            "file_name": f"{frame_id}.jpg",
                            "license": 1,
                            "date_captured": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        self.coco_data["images"].append(coco_image)
                        
                        # Get mask for current frame
                        if frame_count in all_frame_masks:
                            current_mask = all_frame_masks[frame_count]
                        elif frame_count in keyframe_masks:
                            current_mask = keyframe_masks[frame_count]
                        else:
                            logger.warning(f"No mask available for frame {frame_count}")
                            frame_count += 1
                            self.image_id += 1
                            continue
                        
                        # Process and save the mask
                        mask_path = output_path / 'masks' / f"{frame_id}_mask.png"
                        processed_mask = self.sam_annotator.postprocess_mask(current_mask)
                        Image.fromarray((processed_mask * 255).astype(np.uint8)).save(mask_path)
                        
                        # Create COCO annotation
                        rle = mask_util.encode(np.asfortranarray(processed_mask.astype(np.uint8)))
                        rle['counts'] = rle['counts'].decode('utf-8')
                        water_annotation = {
                            "id": self.annotation_id,
                            "image_id": self.image_id,
                            "category_id": 3,  # water
                            "segmentation": rle,
                            "area": float(np.sum(processed_mask)),
                            "bbox": self.mask_to_bbox(processed_mask),
                            "iscrowd": 0
                        }
                        self.coco_data["annotations"].append(water_annotation)
                        self.annotation_id += 1
                        
                        # Create visualization
                        vis_img = pil_frame.copy()
                        
                        # Add water mask to visualization
                        overlay = Image.new('RGBA', vis_img.size, (0, 0, 0, 0))
                        overlay_draw = ImageDraw.Draw(overlay)
                        mask_np = processed_mask.astype(np.uint8) * 255
                        mask_img = Image.fromarray(mask_np, 'L')
                        overlay_draw.bitmap((0, 0), mask_img, fill=(0, 0, 255, 76))  # Blue for water
                        vis_img = Image.alpha_composite(vis_img.convert('RGBA'), overlay).convert('RGB')
                        draw = ImageDraw.Draw(vis_img)
                        
                        # Add frame info to visualization
                        try:
                            status_font = ImageFont.truetype("arial.ttf", 14)
                        except:
                            status_font = ImageFont.load_default()
                            
                        if frame_count in keyframe_masks:
                            draw.text((10, 10), "KEYFRAME", fill=(255, 255, 0), font=status_font)
                        else:
                            draw.text((10, 10), "PROPAGATED MASK", fill=(100, 255, 100), font=status_font)
                        
                        # Write visualization frame to video
                        output_frame = av.VideoFrame.from_image(vis_img)
                        for packet in output_stream.encode(output_frame):
                            output_container.mux(packet)
                        
                        frame_count += 1
                        self.image_id += 1
                        
                        if frame_count % 10 == 0:
                            logger.info(f"Processed {frame_count}/{total_frames} frames...")
                            
                    except Exception as e:
                        logger.error(f"Error processing frame {frame_count}: {e}")
                        import traceback
                        traceback.print_exc()
                        frame_count += 1
                        self.image_id += 1
                
                # Finalize video
                for packet in output_stream.encode(None):
                    output_container.mux(packet)
                output_container.close()
                container.close()
                logger.info(f"Completed video {video_idx+1} - Processed {frame_count} frames")
                
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
        parser = argparse.ArgumentParser(description="Mask Annotation with SAM2 Video Propagation")
        parser.add_argument("--interval", type=int, default=5, help="Interval percentage for keyframes (default: 5%%)")
        parser.add_argument("--checkpoint", type=str, default="./checkpoints/sam2.1_hiera_large.pt", 
                           help="Path to SAM2 checkpoint file")
        parser.add_argument("--config", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                           help="Path to SAM2 config file")
        
        args = parser.parse_args()
        
        # Apply settings
        self.sam_video_propagator.checkpoint = args.checkpoint
        self.sam_video_propagator.model_cfg = args.config
        
        # Process videos
        dataset_dir = Path("dataset")
        video_files = list(dataset_dir.glob("*.mp4")) + list(dataset_dir.glob("*.avi"))
        if not video_files:
            logger.error("No videos found in dataset/ directory.")
            return
            
        output_dir = Path("mask_annotation_data")
        output_dir.mkdir(exist_ok=True)
        
        # Log configuration
        logger.info(f"Starting mask annotation with:")
        logger.info(f"  SAM2 Video Propagation enabled")
        logger.info(f"  Keyframe interval: {args.interval}%")
        logger.info(f"  SAM2 checkpoint: {self.sam_video_propagator.checkpoint}")
        logger.info(f"  SAM2 config: {self.sam_video_propagator.model_cfg}")
        
        # Process videos
        num_images, num_annotations = self.process_videos(video_files, output_dir)
        
        logger.info("Mask annotation complete")
        logger.info(f"Total images: {num_images}")
        logger.info(f"Total annotations: {num_annotations}")
        logger.info(f"Output saved to {output_dir}")


################################################################################
# MAIN FUNCTION
################################################################################
if __name__ == "__main__":
    annotation_system = MaskAnnotationSystem()
    annotation_system.main()
