import os
import sys
import shutil
import yaml
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import random
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def clear_cache(data_dir):
    """Remove any existing labels.cache files to force a cache refresh."""
    data_dir = Path(data_dir)
    for cache_file in data_dir.glob('**/labels.cache'):
        try:
            cache_file.unlink()
            logger.info(f"Removed cache file: {cache_file}")
        except Exception as e:
            logger.warning(f"Could not remove cache file {cache_file}: {e}")

def validate_yolo_label(label_path, margin=0.05, num_classes=2):
    """
    Aggressively validate and fix a YOLO format label file.
    
    Rather than resetting invalid class IDs, this function removes any annotation
    that has a class id outside the valid range (0 to num_classes-1).

    Args:
        label_path: Path to the label file.
        margin: Safety margin for bounding boxes (between 0 and 1).
        num_classes: Number of valid classes (annotations must have class IDs in 0 to num_classes-1).
        
    Returns:
        A tuple (valid, lines_fixed) where 'valid' is True if any annotations remain,
        and 'lines_fixed' is the total number of lines where bounding box adjustments were performed.
    """
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        valid_lines = []
        modified = False
        lines_fixed = 0

        for i, line in enumerate(lines):
            try:
                parts = line.strip().split()
                if len(parts) != 5:
                    # Skip line if it doesn't have exactly 5 values.
                    modified = True
                    logger.warning(f"Line {i+1} in {label_path} has {len(parts)} values; expected 5. Skipping.")
                    continue

                # Parse annotation values.
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # If the class ID is not within the allowed range, remove the annotation.
                if class_id < 0 or class_id >= num_classes:
                    logger.warning(
                        f"Line {i+1} in {label_path} has invalid class id {class_id} "
                        f"(allowed range is 0 to {num_classes - 1}). Removing this annotation."
                    )
                    modified = True
                    continue

                # Check for out-of-bound values for the box parameters.
                if (x_center < 0 or x_center > 1 or
                    y_center < 0 or y_center > 1 or
                    width <= 0 or width > 1 or
                    height <= 0 or height > 1):
                    logger.warning(f"Line {i+1} in {label_path} has out-of-bounds values. Fixing.")
                    modified = True

                # Aggressive fixes:
                old_width, old_height = width, height
                width = max(0.01, min(0.90, width))
                height = max(0.01, min(0.90, height))

                max_x_center = 1.0 - margin - width / 2
                min_x_center = margin + width / 2
                max_y_center = 1.0 - margin - height / 2
                min_y_center = margin + height / 2

                old_x, old_y = x_center, y_center
                x_center = max(min_x_center, min(max_x_center, x_center))
                y_center = max(min_y_center, min(max_y_center, y_center))

                # If the box is still out of bounds after recentering, shrink it further.
                if (x_center - width / 2 < 0 or x_center + width / 2 > 1 or
                    y_center - height / 2 < 0 or y_center + height / 2 > 1):
                    logger.warning(f"Line {i+1} in {label_path} still out of bounds. Applying more aggressive fix.")
                    width *= 0.9
                    height *= 0.9
                    max_x_center = 1.0 - margin - width / 2
                    min_x_center = margin + width / 2
                    max_y_center = 1.0 - margin - height / 2
                    min_y_center = margin + height / 2
                    x_center = max(min_x_center, min(max_x_center, x_center))
                    y_center = max(min_y_center, min(max_y_center, y_center))

                if (x_center != old_x or y_center != old_y or width != old_width or height != old_height):
                    lines_fixed += 1
                    modified = True

                valid_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

            except Exception as e:
                logger.warning(f"Error in line {i+1} of {label_path}: {e}. Skipping.")
                modified = True

        # Write back the file only if modifications occurred.
        if modified:
            with open(label_path, 'w') as f:
                f.writelines(valid_lines)

        return len(valid_lines) > 0, lines_fixed

    except Exception as e:
        logger.error(f"Error processing label file {label_path}: {e}")
        return False, 0

def apply_random_crop(image_path, label_path, output_image_path, output_label_path, min_crop_percent=0.6, max_crop_percent=0.9):
    """
    Create a randomly cropped version of an image and adjust its labels accordingly.
    
    Args:
        image_path: Path to the original image
        label_path: Path to the original YOLO format label file
        output_image_path: Path to save the cropped image
        output_label_path: Path to save the adjusted labels
        min_crop_percent: Minimum percentage of original image to keep
        max_crop_percent: Maximum percentage of original image to keep
        
    Returns:
        True if the crop was successful, False otherwise
    """
    try:
        # Read the image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return False
            
        h, w, _ = img.shape
        
        # Determine random crop size between min and max percent
        crop_percent = random.uniform(min_crop_percent, max_crop_percent)
        new_w = int(w * crop_percent)
        new_h = int(h * crop_percent)
        
        # Calculate random crop coordinates (anywhere in the image)
        x1 = random.randint(0, w - new_w)
        y1 = random.randint(0, h - new_h)
        x2 = x1 + new_w
        y2 = y1 + new_h
        
        # Create the cropped image
        cropped_img = img[y1:y2, x1:x2]
        
        # Read the labels
        if not label_path.exists():
            logger.warning(f"Label file not found: {label_path}")
            return False
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        new_lines = []
        valid_boxes = 0
        
        # Normalize crop coordinates to 0-1 range for YOLO format
        x1_norm = x1 / w
        y1_norm = y1 / h
        crop_w_norm = (x2 - x1) / w
        crop_h_norm = (y2 - y1) / h
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            # Parse original bbox (YOLO format: class_id, x_center, y_center, width, height)
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert to absolute coordinates (still in 0-1 range)
            x1_box = x_center - width / 2
            y1_box = y_center - height / 2
            x2_box = x_center + width / 2
            y2_box = y_center + height / 2
            
            # Check if box intersects with crop area
            if (x2_box > x1_norm and x1_box < x1_norm + crop_w_norm and
                y2_box > y1_norm and y1_box < y1_norm + crop_h_norm):
                
                # Calculate intersection
                ix1 = max(x1_box, x1_norm)
                iy1 = max(y1_box, y1_norm)
                ix2 = min(x2_box, x1_norm + crop_w_norm)
                iy2 = min(y2_box, y1_norm + crop_h_norm)
                
                # Calculate intersection area
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                box_area = width * height
                
                # Skip if intersection is too small (less than 30% of original box)
                if intersection_area < 0.3 * box_area:
                    continue
                
                # Adjust coordinates to new crop (0-1 range in cropped image)
                new_x1 = (ix1 - x1_norm) / crop_w_norm
                new_y1 = (iy1 - y1_norm) / crop_h_norm
                new_x2 = (ix2 - x1_norm) / crop_w_norm
                new_y2 = (iy2 - y1_norm) / crop_h_norm
                
                # Convert back to YOLO format
                new_x_center = (new_x1 + new_x2) / 2
                new_y_center = (new_y1 + new_y2) / 2
                new_width = new_x2 - new_x1
                new_height = new_y2 - new_y1
                
                # Ensure values are valid
                if (new_x_center < 0 or new_x_center > 1 or
                    new_y_center < 0 or new_y_center > 1 or
                    new_width <= 0 or new_width > 1 or
                    new_height <= 0 or new_height > 1):
                    continue
                
                new_lines.append(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")
                valid_boxes += 1
        
        # Only save if there's at least one valid box
        if valid_boxes > 0:
            # Save the cropped image
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_image_path), cropped_img)
            
            # Save the adjusted labels
            output_label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_label_path, 'w') as f:
                f.writelines(new_lines)
            
            return True
        else:
            return False
            
    except Exception as e:
        logger.error(f"Error creating random crop: {str(e)}")
        return False
        
def apply_quadrant_crop(image_path, label_path, output_image_path, output_label_path, quadrant, min_box_overlap=0.5):
    """
    Create a crop focused on one quadrant of the image, ensuring at least one bounding box is 
    at least 50% visible, and removing boxes with less than min_box_overlap visibility.
    
    Args:
        image_path: Path to the original image
        label_path: Path to the original YOLO format label file
        output_image_path: Path to save the cropped image
        output_label_path: Path to save the adjusted labels
        quadrant: Which quadrant to crop (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
        min_box_overlap: Minimum overlap required to keep a box (0.5 = 50%)
        
    Returns:
        True if the crop was successful, False otherwise
    """
    try:
        # Read the image
        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return False
            
        h, w, _ = img.shape
        
        # Read the labels first to find if there are valid boxes
        if not label_path.exists():
            logger.warning(f"Label file not found: {label_path}")
            return False
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            logger.warning(f"No labels found in {label_path}")
            return False
        
        # Define crop coordinates based on quadrant
        # Each quadrant will be slightly larger than 1/4 of the image (about 60%)
        # to ensure we include objects that might be near quadrant boundaries
        crop_size_w = int(w * 0.6)
        crop_size_h = int(h * 0.6)
        
        if quadrant == 0:  # Top-left
            x1, y1 = 0, 0
        elif quadrant == 1:  # Top-right
            x1, y1 = w - crop_size_w, 0
        elif quadrant == 2:  # Bottom-left
            x1, y1 = 0, h - crop_size_h
        else:  # Bottom-right (quadrant == 3)
            x1, y1 = w - crop_size_w, h - crop_size_h
            
        x2 = x1 + crop_size_w
        y2 = y1 + crop_size_h
        
        # Ensure we don't go out of bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        # Create the cropped image
        cropped_img = img[y1:y2, x1:x2]
        
        # Process label coordinates
        new_lines = []
        valid_boxes = 0
        has_significant_box = False
        
        # Normalize crop coordinates to 0-1 range for YOLO format
        x1_norm = x1 / w
        y1_norm = y1 / h
        crop_w_norm = (x2 - x1) / w
        crop_h_norm = (y2 - y1) / h
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
                
            # Parse original bbox
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            
            # Convert to absolute coordinates (still in 0-1 range)
            x1_box = x_center - width / 2
            y1_box = y_center - height / 2
            x2_box = x_center + width / 2
            y2_box = y_center + height / 2
            
            # Check if box intersects with crop area
            if (x2_box > x1_norm and x1_box < x1_norm + crop_w_norm and
                y2_box > y1_norm and y1_box < y1_norm + crop_h_norm):
                
                # Calculate intersection
                ix1 = max(x1_box, x1_norm)
                iy1 = max(y1_box, y1_norm)
                ix2 = min(x2_box, x1_norm + crop_w_norm)
                iy2 = min(y2_box, y1_norm + crop_h_norm)
                
                # Calculate intersection area
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                box_area = width * height
                
                # Calculate overlap percentage
                overlap_percent = intersection_area / box_area
                
                # Skip boxes with less than minimum overlap
                if overlap_percent < min_box_overlap:
                    continue
                
                # Mark if we found at least one box with significant overlap
                if overlap_percent >= min_box_overlap:
                    has_significant_box = True
                
                # Adjust coordinates to new crop (0-1 range in cropped image)
                new_x1 = (ix1 - x1_norm) / crop_w_norm
                new_y1 = (iy1 - y1_norm) / crop_h_norm
                new_x2 = (ix2 - x1_norm) / crop_w_norm
                new_y2 = (iy2 - y1_norm) / crop_h_norm
                
                # Convert back to YOLO format
                new_x_center = (new_x1 + new_x2) / 2
                new_y_center = (new_y1 + new_y2) / 2
                new_width = new_x2 - new_x1
                new_height = new_y2 - new_y1
                
                # Ensure values are valid
                if (new_x_center < 0 or new_x_center > 1 or
                    new_y_center < 0 or new_y_center > 1 or
                    new_width <= 0 or new_width > 1 or
                    new_height <= 0 or new_height > 1):
                    continue
                
                new_lines.append(f"{class_id} {new_x_center:.6f} {new_y_center:.6f} {new_width:.6f} {new_height:.6f}\n")
                valid_boxes += 1
        
        # Only proceed if there's at least one box with significant overlap
        if has_significant_box and valid_boxes > 0:
            # Save the cropped image
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_image_path), cropped_img)
            
            # Save the adjusted labels
            output_label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_label_path, 'w') as f:
                f.writelines(new_lines)
            
            return True
        else:
            return False
            
    except Exception as e:
        logger.error(f"Error creating quadrant crop: {str(e)}")
        return False

def create_selective_augmentations(data_dir, aug_percent=0.25):
    """
    Create selective augmentations for a subset of images - much more controlled than before.
    
    Args:
        data_dir: Path to the dataset directory
        aug_percent: Percentage of original images to augment (0.25 = 25%)
        
    Returns:
        Number of augmented images created
    """
    data_dir = Path(data_dir)
    augmented_count = 0
    
    for split in ['train', 'val']:  # Don't augment test set
        images_dir = data_dir / split / 'images'
        labels_dir = data_dir / split / 'labels'
        
        if not images_dir.exists() or not labels_dir.exists():
            logger.warning(f"Directory not found: {images_dir} or {labels_dir}")
            continue
            
        augmented_images_dir = data_dir / split / 'images'
        augmented_labels_dir = data_dir / split / 'labels'
        
        # Create directories
        augmented_images_dir.mkdir(parents=True, exist_ok=True)
        augmented_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # List all image files
        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        total_images = len(image_files)
        logger.info(f"Found {total_images} images in {split} split")
        
        # Determine how many images to augment (percentage of total)
        images_to_augment = int(total_images * aug_percent)
        logger.info(f"Will augment {images_to_augment} images ({aug_percent*100:.1f}% of total)")
        
        # Randomly select images to augment
        if images_to_augment > 0:
            selected_images = random.sample(image_files, min(images_to_augment, total_images))
            
            for image_path in tqdm(selected_images, desc=f"Creating augmentations for {split}"):
                # Corresponding label file
                label_path = labels_dir / f"{image_path.stem}.txt"
                
                if not label_path.exists():
                    continue
                
                # For each image, pick ONE type of augmentation
                # 0 = random crop, 1-4 = quadrant crops
                aug_type = random.randint(0, 4)
                
                if aug_type == 0:
                    # Random crop
                    output_image_path = augmented_images_dir / f"{image_path.stem}_random_crop{image_path.suffix}"
                    output_label_path = augmented_labels_dir / f"{image_path.stem}_random_crop.txt"
                    
                    if apply_random_crop(image_path, label_path, output_image_path, output_label_path):
                        augmented_count += 1
                else:
                    # Quadrant crop (0-3)
                    quadrant = aug_type - 1
                    quadrant_name = ["tl", "tr", "bl", "br"][quadrant]
                    output_image_path = augmented_images_dir / f"{image_path.stem}_quad_{quadrant_name}{image_path.suffix}"
                    output_label_path = augmented_labels_dir / f"{image_path.stem}_quad_{quadrant_name}.txt"
                    
                    if apply_quadrant_crop(image_path, label_path, output_image_path, output_label_path, quadrant):
                        augmented_count += 1
                
        logger.info(f"Created {augmented_count} augmented images for {split} split")
                
    logger.info(f"Created {augmented_count} augmented images total across all splits")
    return augmented_count

def validate_dataset(data_dir, margin=0.05, num_classes=2):
    data_dir = Path(data_dir)
    validated_count = 0
    invalid_count = 0
    total_boxes_fixed = 0
    
    if not data_dir.exists():
        logger.error(f"Dataset directory {data_dir} does not exist")
        return False
    
    logger.info(f"Applying AGGRESSIVE box validation with {margin*100:.1f}% safety margin")
    
    for split in ['train', 'val', 'test']:
        labels_dir = data_dir / split / 'labels'
        if not labels_dir.exists():
            logger.warning(f"Labels directory {labels_dir} does not exist")
            continue
        
        label_files = list(labels_dir.glob('*.txt'))
        logger.info(f"Validating {len(label_files)} label files in {labels_dir}")
        split_boxes_fixed = 0
        
        for label_path in tqdm(label_files, desc=f"Aggressively fixing {split} labels"):
            valid, boxes_fixed = validate_yolo_label(label_path, margin=margin, num_classes=num_classes)
            split_boxes_fixed += boxes_fixed
            if valid:
                validated_count += 1
            else:
                invalid_count += 1
                try:
                    image_path = data_dir / split / 'images' / f"{label_path.stem}.jpg"
                    if image_path.exists():
                        logger.warning(f"Removing invalid label and its image: {label_path.name}")
                        label_path.unlink()
                        image_path.unlink()
                except Exception as e:
                    logger.error(f"Error removing files: {e}")
        
        logger.info(f"Fixed {split_boxes_fixed} boxes in {split} split")
        total_boxes_fixed += split_boxes_fixed
    
    logger.info(f"Validated {validated_count} label files. Removed {invalid_count} invalid files.")
    logger.info(f"TOTAL: Fixed {total_boxes_fixed} boxes across all splits")
    return validated_count > 0

def fix_yaml_config(yaml_path):
    if not Path(yaml_path).exists():
        logger.error(f"YAML file {yaml_path} does not exist")
        return False
    
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        
        base_dir = Path(yaml_path).parent
        data['path'] = str(base_dir.absolute())
        
        for key in ['train', 'val', 'test']:
            if key in data:
                data[key] = f'./{key}'
        
        if 'nc' not in data or 'names' not in data:
            logger.warning(f"Missing 'nc' or 'names' in {yaml_path}. Adding default values.")
            data['nc'] = 2
            data['names'] = ["obstacle", "goal"]
        
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        
        logger.info(f"Fixed YAML configuration in {yaml_path}")
        return True
    except Exception as e:
        logger.error(f"Error fixing YAML config {yaml_path}: {e}")
        return False

def train_yolov8(data_yaml, output_dir, batch_size=16, epochs=200, model_size='s', device=None, margin=0.05, 
              apply_augmentation=True, aug_percent=0.25):
    try:
        from ultralytics import YOLO
        
        data_yaml = Path(data_yaml)
        output_dir = Path(output_dir)
        
        # Clear cache before starting training
        clear_cache(data_yaml.parent)
        
        if device is None:
            device = 0 if torch.cuda.is_available() else 'cpu'
        
        logger.info("Step 1: Fixing YAML configuration...")
        if not fix_yaml_config(data_yaml):
            logger.error("Failed to fix YAML configuration. Aborting training.")
            return False

        # Load dataset YAML to determine number of classes
        with open(data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        num_classes = data_cfg.get('nc', 2)
        
        logger.info("Step 2: Aggressively fixing bounding boxes...")
        data_dir = data_yaml.parent
        if not validate_dataset(data_dir, margin=margin, num_classes=num_classes):
            logger.error("Dataset validation failed. Aborting training.")
            return False
        
        # Apply selective augmentation if enabled
        if apply_augmentation:
            try:
                logger.info(f"Step 3: Creating selective augmentations for {aug_percent*100:.1f}% of images...")
                augmented_count = create_selective_augmentations(data_dir, aug_percent)
                logger.info(f"Created {augmented_count} augmented images")
            except Exception as e:
                logger.error(f"Error creating augmentations: {e}")
                logger.info("Continuing without augmentation...")
        
        logger.info("Step 4: Initializing YOLOv8 model...")
        model_path = f"yolov8{model_size}.pt"
        logger.info(f"Using model: {model_path}")
        model = YOLO(model_path)
        
        if device != 'cpu':
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"PyTorch version: {torch.__version__}")
            if torch.cuda.is_available():
                logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
                logger.info(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
                try:
                    logger.info(f"Memory cached: {torch.cuda.memory_cached(0) / 1024**2:.2f} MB")
                except:
                    logger.info(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
        
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        image_size = 640
        workers = 0
        
        run_dir = output_dir / 'yolo_training'
        run_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Step 5: Starting training with batch_size={batch_size}, image_size={image_size}")
        use_amp = device != 'cpu'
        
        # Configure for Jetson Orin Nano performance
        if device != 'cpu':
            # Set cache memory limit for better memory management on Jetson
            torch.cuda.set_per_process_memory_fraction(0.8)
            
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            patience=20,
            device=device,
            workers=workers,
            project=str(output_dir),
            name='yolo_training',
            verbose=True,
            exist_ok=True,
            amp=use_amp,
            rect=True,
            single_cls=False,
            cache=False,
        )
        
        try:
            weight_files = list((output_dir / 'yolo_training').glob('*.pt'))
            if weight_files:
                best_model = sorted(weight_files, key=lambda x: x.stat().st_mtime)[-1]
                output_path = output_dir / 'best_yolov8_model.pt'
                shutil.copy(best_model, output_path)
                logger.info(f"Copied best model to {output_path}")
                
                logger.info("Exporting model to ONNX format for Jetson deployment...")
                model = YOLO(str(output_path))
                
                # Configure export specifically for Jetson Orin Nano
                model.export(format='onnx', 
                             imgsz=image_size, 
                             simplify=True,  # Simplify the model for better inference
                             opset=13,       # Compatible ONNX opset for Jetson
                             dynamic=True)   # Dynamic axes for flexible inference
                
                logger.info(f"Exported to {output_dir / 'best_yolov8_model.onnx'}")
        except Exception as e:
            logger.error(f"Error copying/exporting best model: {e}")
                
        return True

    except Exception as e:
        logger.error(f"Error in YOLOv8 training: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Robust YOLOv8 Training with Selective Augmentation")
    parser.add_argument("--data_yaml", type=str, default="trained_models/yolov8/dataset.yaml",
                        help="Path to the dataset YAML configuration file")
    parser.add_argument("--output_dir", type=str, default="trained_models",
                        help="Directory for saving trained models")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training (default: 16)")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--model_size", type=str, default="s", choices=['n', 's', 'm', 'l', 'x'],
                        help="YOLOv8 model size (default: s)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (default: auto; use 'cpu' to force CPU training)")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="Safety margin for bounding boxes (default: 0.05 = 5%)")
    parser.add_argument("--validate_only", action="store_true",
                        help="Only validate dataset without training")
    parser.add_argument("--augmentation", action="store_true", default=True,
                        help="Apply selective data augmentation")
    parser.add_argument("--aug_percent", type=float, default=0.25,
                        help="Percentage of original images to augment (default: 0.25 = 25%)")
    args = parser.parse_args()
    
    if args.validate_only:
        data_dir = Path(args.data_yaml).parent
        with open(args.data_yaml, 'r') as f:
            data_cfg = yaml.safe_load(f)
        num_classes = data_cfg.get('nc', 2)
        clear_cache(data_dir)
        validate_dataset(data_dir, margin=args.margin, num_classes=num_classes)
        fix_yaml_config(args.data_yaml)
        
        if args.augmentation:
            create_selective_augmentations(data_dir, args.aug_percent)
    else:
        train_yolov8(
            args.data_yaml,
            args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            model_size=args.model_size,
            device=args.device,
            margin=args.margin,
            apply_augmentation=args.augmentation,
            aug_percent=args.aug_percent
        )

if __name__ == "__main__":
    main()
