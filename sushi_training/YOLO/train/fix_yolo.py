#!/usr/bin/env python3
import os
import sys
import glob
from pathlib import Path
from tqdm import tqdm
import argparse

def fix_bounding_boxes(label_dir, margin=0.01):
    """
    Aggressively fix all bounding boxes to ensure they are safely within bounds.
    
    Args:
        label_dir: Directory containing YOLO format label files
        margin: Safety margin to keep boxes away from the boundaries (0-1)
    """
    print(f"Fixing bounding boxes in {label_dir} with margin {margin}")
    
    # Find all label files
    label_files = list(Path(label_dir).glob("*.txt"))
    print(f"Found {len(label_files)} label files")
    
    files_modified = 0
    boxes_fixed = 0
    
    for label_path in tqdm(label_files, desc="Fixing bounding boxes"):
        file_modified = False
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Ensure width and height are positive and not too small
                width = max(0.01, min(1.0 - 2*margin, width))
                height = max(0.01, min(1.0 - 2*margin, height))
                
                # Calculate max allowed center coordinates based on width/height
                max_x_center = 1.0 - margin - width/2
                min_x_center = margin + width/2
                max_y_center = 1.0 - margin - height/2
                min_y_center = margin + height/2
                
                # Bound the center coordinates
                old_x, old_y = x_center, y_center
                x_center = max(min_x_center, min(max_x_center, x_center))
                y_center = max(min_y_center, min(max_y_center, y_center))
                
                # Check if we modified the box
                if (x_center != old_x or y_center != old_y or
                    width != float(parts[3]) or height != float(parts[4])):
                    boxes_fixed += 1
                    file_modified = True
                
                new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            except Exception as e:
                print(f"Error processing line in {label_path}: {e}")
                continue
        
        # Write back if modified
        if file_modified:
            files_modified += 1
            with open(label_path, 'w') as f:
                f.writelines(new_lines)
    
    print(f"Fixed {boxes_fixed} boxes in {files_modified} files")
    return files_modified, boxes_fixed

def main():
    parser = argparse.ArgumentParser(description="Fix bounding boxes in YOLO format labels")
    parser.add_argument("--data_dir", type=str, default="trained_models/yolov8",
                      help="Directory containing train/val/test splits with labels subdirectories")
    parser.add_argument("--margin", type=float, default=0.03,
                      help="Safety margin to keep boxes away from boundaries (default: 0.03)")
    args = parser.parse_args()
    
    # Process each split
    total_files = 0
    total_boxes = 0
    
    for split in ['train', 'val', 'test']:
        labels_dir = os.path.join(args.data_dir, split, 'labels')
        if os.path.exists(labels_dir):
            files, boxes = fix_bounding_boxes(labels_dir, args.margin)
            total_files += files
            total_boxes += boxes
    
    print(f"Summary: Fixed {total_boxes} boxes in {total_files} files")

if __name__ == "__main__":
    main()
