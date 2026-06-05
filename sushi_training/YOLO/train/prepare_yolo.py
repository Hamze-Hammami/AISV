import os
import json
import argparse
import logging
import shutil
import random
import av
import yaml
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def extract_frames_from_videos(video_dir, output_dir, sample_rate=1):
    """
    Extract frames from all videos in the specified directory
    
    Args:
        video_dir: Directory containing videos
        output_dir: Directory to save extracted frames
        sample_rate: Extract 1 frame every N frames (default: 1 = extract all frames)
        
    Returns:
        Dictionary mapping frame filenames to their dimensions (width, height)
    """
    video_dir = Path(video_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Store frame dimensions for later use
    frame_dimensions = {}
    
    # Find all video files
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(list(video_dir.glob(f'*{ext}')))
    
    logger.info(f"Found {len(video_files)} video files in {video_dir}")
    
    if not video_files:
        logger.error(f"No video files found in {video_dir}")
        return frame_dimensions
    
    # Extract frames from each video
    for video_idx, video_path in enumerate(video_files):
        logger.info(f"Processing video {video_idx+1}/{len(video_files)}: {video_path.name}")
        
        try:
            # Open video with PyAV
            container = av.open(str(video_path))
            video_stream = next(s for s in container.streams if s.type == 'video')
            
            # Get video information
            total_frames = video_stream.frames or 0
            fps = video_stream.average_rate
            width = video_stream.width
            height = video_stream.height
            
            logger.info(f"Video has {total_frames} frames at {fps} fps, {width}x{height} resolution")
            
            # Extract frames
            frame_count = 0
            extracted_count = 0
            
            with tqdm(total=total_frames or None, desc=f"Extracting frames from {video_path.name}") as pbar:
                for frame in container.decode(video=0):
                    if frame_count % sample_rate == 0:
                        # Convert to PIL image
                        pil_frame = frame.to_image()
                        
                        # Generate frame filename
                        frame_filename = f"video_{video_idx}_frame_{frame_count}.jpg"
                        frame_path = output_dir / frame_filename
                        
                        # Save frame
                        pil_frame.save(frame_path)
                        extracted_count += 1
                        
                        # Store dimensions
                        frame_dimensions[frame_filename] = (width, height)
                        
                        # Log progress occasionally
                        if extracted_count % 100 == 0:
                            logger.info(f"Extracted {extracted_count} frames from {video_path.name}")
                    
                    frame_count += 1
                    pbar.update(1)
            
            logger.info(f"Completed extracting {extracted_count} frames from {video_path.name}")
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
    
    logger.info(f"Completed frame extraction. Total frames extracted: {len(list(output_dir.glob('*.jpg')))}")
    return frame_dimensions

def convert_coco_to_yolo(coco_json_path, output_dir, frame_dir=None, merge_with_existing=False):
    """
    Convert COCO format annotations to YOLO format and organize into train/val/test splits
    
    Args:
        coco_json_path: Path to COCO format JSON annotations
        output_dir: Base directory for YOLO dataset
        frame_dir: Directory containing image frames (optional)
        merge_with_existing: Whether to merge with existing annotations (if any)
        
    Returns:
        Path to the YAML configuration file for the dataset
    """
    coco_json_path = Path(coco_json_path)
    output_dir = Path(output_dir)
    
    # Load COCO annotations
    logger.info(f"Loading COCO annotations from {coco_json_path}")
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image ID to file name mapping
    image_id_to_file = {}
    image_id_to_size = {}
    for image in coco_data['images']:
        image_id_to_file[image['id']] = image['file_name']
        image_id_to_size[image['id']] = (image['width'], image['height'])
    
    # Create annotations by image ID
    annotations_by_image = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)
    
    # Create category ID mapping
    category_id_to_name = {}
    for category in coco_data['categories']:
        category_id_to_name[category['id']] = category['name']
    
    # Create output directories
    yolo_dataset_dir = output_dir / 'yolo_dataset'
    yolo_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    splits = ['train', 'val', 'test']
    split_dirs = {}
    for split in splits:
        for subdir in ['images', 'labels']:
            split_dir = yolo_dataset_dir / split / subdir
            split_dir.mkdir(parents=True, exist_ok=True)
            if subdir == 'images':
                split_dirs[split] = split_dir
    
    # Get all image IDs and randomize them for splitting
    all_image_ids = list(image_id_to_file.keys())
    random.shuffle(all_image_ids)
    
    # Split images into train/val/test (80/10/10)
    num_train = int(len(all_image_ids) * 0.8)
    num_val = int(len(all_image_ids) * 0.1)
    
    train_ids = all_image_ids[:num_train]
    val_ids = all_image_ids[num_train:num_train+num_val]
    test_ids = all_image_ids[num_train+num_val:]
    
    split_image_ids = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    logger.info(f"Split {len(all_image_ids)} images into: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")
    
    # Process each split
    total_processed = 0
    total_annotations = 0
    frames_not_found = 0
    
    # Check if frame_dir is provided
    if frame_dir:
        frame_dir = Path(frame_dir)
        if not frame_dir.exists():
            logger.warning(f"Frame directory {frame_dir} does not exist")
            frame_dir = None
    
    # Process each split
    for split, image_ids in split_image_ids.items():
        logger.info(f"Processing {len(image_ids)} images for {split} split")
        processed_count = 0
        annotation_count = 0
        class2_images = 0
        
        for image_id in tqdm(image_ids, desc=f"Converting {split} annotations"):
            # Get image filename and check if it has annotations
            image_filename = image_id_to_file[image_id]
            image_annotations = annotations_by_image.get(image_id, [])
            
            # Filter out class 2 annotations and keep track of images that only had class 2
            has_class2_only = False
            if image_annotations:
                # Check if image only has class 2 annotations
                class2_only_annotations = [ann for ann in image_annotations if ann['category_id'] == 2]
                if len(class2_only_annotations) == len(image_annotations) and len(class2_only_annotations) > 0:
                    has_class2_only = True
                    class2_images += 1
                
                # Remove all class 2 annotations
                image_annotations = [ann for ann in image_annotations if ann['category_id'] != 2]
            
            # Skip images without annotations if not merging, and not class2-only images
            # (we want to include class2-only images as background)
            if not image_annotations and not merge_with_existing and not has_class2_only:
                continue
            
            # Prepare image copy and label creation
            source_image_path = None
            if frame_dir:
                source_image_path = frame_dir / image_filename
                if not source_image_path.exists():
                    frames_not_found += 1
                    if frames_not_found <= 5:  # Only show first few warnings to avoid log spam
                        logger.warning(f"Image not found: {source_image_path}")
                    if frames_not_found == 6:
                        logger.warning("Additional missing frames will not be logged individually")
                    continue
            
            # Get output paths
            dest_image_path = split_dirs[split] / image_filename
            label_path = yolo_dataset_dir / split / 'labels' / f"{Path(image_filename).stem}.txt"
            
            # Copy image if source exists and destination doesn't (or if forced)
            if source_image_path and (not dest_image_path.exists() or not merge_with_existing):
                try:
                    shutil.copy(source_image_path, dest_image_path)
                except Exception as e:
                    logger.error(f"Error copying image {source_image_path}: {e}")
                    continue
            
            # For class 2 only images, create an empty label file (keeping the image as background)
            if has_class2_only:
                with open(label_path, 'w') as f:
                    # Create an empty label file for images that only had class 2 annotations
                    pass
                processed_count += 1
                continue
            
            # Convert COCO annotations to YOLO format
            width, height = image_id_to_size[image_id]
            
            # Check if we should merge with existing annotations
            existing_annotations = []
            if merge_with_existing and label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        existing_annotations = f.readlines()
                        
                        # Filter out class 2 annotations from existing file if any
                        filtered_annotations = []
                        for line in existing_annotations:
                            parts = line.strip().split()
                            if len(parts) >= 5 and int(parts[0]) != 2:
                                # If class ID is not 2, keep it
                                filtered_annotations.append(line)
                        existing_annotations = filtered_annotations
                        
                except Exception as e:
                    logger.warning(f"Error reading existing annotations {label_path}: {e}")
            
            # Convert and write YOLO format annotations
            if image_annotations or existing_annotations:
                try:
                    with open(label_path, 'w') as f:
                        # Write existing annotations first if we're merging
                        if existing_annotations:
                            for line in existing_annotations:
                                f.write(line)
                                annotation_count += 1
                        
                        # Convert and write new COCO annotations
                        for annotation in image_annotations:
                            category_id = annotation['category_id']
                            
                            # Skip class 2 annotations
                            if category_id == 2:
                                continue
                                
                            # For classes with ID > 2, decrease the ID by 1 to account for removed class 2
                            if category_id > 2:
                                category_id -= 1
                                
                            bbox = annotation['bbox']  # [x, y, width, height] in COCO format
                            
                            # Convert to YOLO format: [class_id, x_center, y_center, width, height]
                            # All values normalized between 0 and 1
                            x, y, w, h = bbox
                            
                            # Normalize to 0-1 range
                            x_center = (x + w / 2) / width
                            y_center = (y + h / 2) / height
                            norm_width = w / width
                            norm_height = h / height
                            
                            # Skip invalid boxes
                            if norm_width <= 0 or norm_height <= 0:
                                logger.warning(f"Skipping invalid box with width={norm_width}, height={norm_height} in {image_filename}")
                                continue
                            
                            # Ensure values are within 0-1 range with a small buffer
                            margin = 0.005
                            x_center = max(min(x_center, 1.0 - margin), margin)
                            y_center = max(min(y_center, 1.0 - margin), margin)
                            norm_width = max(min(norm_width, 1.0 - margin), 0.01)  # Minimum width of 1%
                            norm_height = max(min(norm_height, 1.0 - margin), 0.01)  # Minimum height of 1%
                            
                            # Write YOLO annotation
                            yolo_line = f"{category_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n"
                            f.write(yolo_line)
                            annotation_count += 1
                    
                    processed_count += 1
                
                except Exception as e:
                    logger.error(f"Error writing annotations for {image_filename}: {e}")
        
        logger.info(f"Processed {processed_count} images with {annotation_count} annotations for {split} split")
        logger.info(f"Found {class2_images} images with only class 2 annotations (kept as background)")
        total_processed += processed_count
        total_annotations += annotation_count
    
    if frames_not_found > 0:
        logger.warning(f"Total of {frames_not_found} frames were not found in the frame directory")
    
    logger.info(f"Completed conversion: {total_processed} images with {total_annotations} annotations")
    
    # Create YAML configuration file - adjust class names to remove class 2
    # and shift higher class indices down by 1
    adjusted_class_names = {}
    for class_id, class_name in category_id_to_name.items():
        if class_id == 2:
            continue  # Skip class 2
        elif class_id > 2:
            adjusted_class_names[class_id - 1] = class_name  # Shift indices down
        else:
            adjusted_class_names[class_id] = class_name  # Keep as is
    
    yaml_content = {
        'path': str(yolo_dataset_dir.absolute()),
        'train': './train/images',
        'val': './val/images',
        'test': './test/images',
        'nc': len(adjusted_class_names),  # Adjusted class count
        'names': [adjusted_class_names[i] for i in range(len(adjusted_class_names))]  # Adjusted class names
    }
    
    yaml_path = yolo_dataset_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    logger.info(f"Created YAML configuration at {yaml_path}")
    logger.info(f"Removed class 2 from training and kept those images as background")
    
    return str(yaml_path)

def validate_yolo_dataset(dataset_yaml):
    """
    Perform basic validation of YOLO dataset
    
    Args:
        dataset_yaml: Path to YAML configuration file
    """
    dataset_yaml = Path(dataset_yaml)
    if not dataset_yaml.exists():
        logger.error(f"Dataset YAML {dataset_yaml} does not exist")
        return False
    
    try:
        with open(dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        base_dir = dataset_yaml.parent
        
        # Check splits
        for split in ['train', 'val', 'test']:
            split_images_dir = base_dir / config.get(split, f'./{split}/images')
            split_labels_dir = base_dir / f"{split}/labels"
            
            if not split_images_dir.exists():
                logger.warning(f"{split} images directory {split_images_dir} does not exist")
                continue
                
            if not split_labels_dir.exists():
                logger.warning(f"{split} labels directory {split_labels_dir} does not exist")
                continue
            
            # Count files
            image_files = list(split_images_dir.glob('*.jpg')) + list(split_images_dir.glob('*.png'))
            label_files = list(split_labels_dir.glob('*.txt'))
            
            logger.info(f"{split}: {len(image_files)} images, {len(label_files)} label files")
            
            # Check sample of labels
            num_samples = min(5, len(label_files))
            if num_samples > 0:
                logger.info(f"Checking {num_samples} random label files from {split}")
                samples = random.sample(label_files, num_samples)
                for label_path in samples:
                    try:
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                        image_name = f"{label_path.stem}.jpg"
                        image_path = split_images_dir / image_name
                        
                        if not image_path.exists():
                            image_path = split_images_dir / f"{label_path.stem}.png"
                        
                        if not image_path.exists():
                            logger.warning(f"Image not found for label: {label_path.name}")
                            continue
                            
                        logger.info(f"Label {label_path.name}: {len(lines)} annotations")
                        
                        # Verify no class 2 exists
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(parts[0])
                                if class_id == 2:
                                    logger.warning(f"Found class 2 in {label_path}, which should have been removed")
                        
                        # Check first line format
                        if lines:
                            parts = lines[0].strip().split()
                            if len(parts) != 5:
                                logger.warning(f"Unexpected format in {label_path}: expected 5 values, got {len(parts)}")
                            
                    except Exception as e:
                        logger.error(f"Error reading label {label_path}: {e}")
        
        logger.info(f"Dataset validation completed for {dataset_yaml}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating dataset: {e}")
        return False

def visualize_dataset_samples(dataset_yaml, output_dir, num_samples=5):
    """
    Visualize random samples from the dataset for verification
    
    Args:
        dataset_yaml: Path to YAML configuration file
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize per split
    """
    dataset_yaml = Path(dataset_yaml)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        base_dir = dataset_yaml.parent
        class_names = config.get('names', ['object'])
        
        for split in ['train', 'val', 'test']:
            split_images_dir = base_dir / config.get(split, f'./{split}/images')
            split_labels_dir = base_dir / f"{split}/labels"
            
            if not split_images_dir.exists() or not split_labels_dir.exists():
                logger.warning(f"Missing directories for {split}, skipping visualization")
                continue
            
            # Get image files with corresponding labels
            image_files = list(split_images_dir.glob('*.jpg')) + list(split_images_dir.glob('*.png'))
            valid_images = []
            
            for img_path in image_files:
                label_path = split_labels_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    valid_images.append((img_path, label_path))
            
            if not valid_images:
                logger.warning(f"No valid image-label pairs found for {split}")
                continue
            
            # Pick random samples
            num_to_sample = min(num_samples, len(valid_images))
            samples = random.sample(valid_images, num_to_sample)
            
            logger.info(f"Visualizing {num_to_sample} samples from {split}")
            
            for idx, (img_path, label_path) in enumerate(samples):
                try:
                    # Load image and annotations
                    image = Image.open(img_path)
                    width, height = image.size
                    
                    # Create drawing context
                    draw = ImageDraw.Draw(image)
                    
                    # Load labels
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    # Draw each annotation
                    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
                    
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        
                        class_id = int(parts[0])
                        x_center = float(parts[1]) * width
                        y_center = float(parts[2]) * height
                        box_width = float(parts[3]) * width
                        box_height = float(parts[4]) * height
                        
                        # Calculate box coordinates
                        x1 = int(x_center - box_width / 2)
                        y1 = int(y_center - box_height / 2)
                        x2 = int(x_center + box_width / 2)
                        y2 = int(y_center + box_height / 2)
                        
                        # Get color for this class
                        color = colors[class_id % len(colors)]
                        
                        # Draw rectangle and label
                        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                        
                        # Draw class name
                        class_name = class_names[class_id] if class_id < len(class_names) else f"class_{class_id}"
                        draw.text((x1, y1-10), class_name, fill=color)
                    
                    # Save visualization
                    output_path = output_dir / f"{split}_sample_{idx}.jpg"
                    image.save(output_path)
                    logger.info(f"Saved visualization to {output_path}")
                    
                except Exception as e:
                    logger.error(f"Error visualizing {img_path}: {e}")
            
        logger.info(f"Visualization completed. Output saved to {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Process videos and convert annotations to YOLO format")
    parser.add_argument("--video_dir", type=str, default="dataset",
                      help="Directory containing video files")
    parser.add_argument("--coco_json", type=str, default="bbox_annotation_data/coco/annotations.json",
                      help="Path to COCO format JSON annotations")
    parser.add_argument("--output_dir", type=str, default="trained_models/yolov82",
                      help="Directory to save output")
    parser.add_argument("--frames_dir", type=str, help="Directory to save extracted frames (default: output_dir/frames)")
    parser.add_argument("--sample_rate", type=int, default=1,
                      help="Extract 1 frame every N frames (default: 1 = extract all frames)")
    parser.add_argument("--skip_frame_extraction", action="store_true",
                      help="Skip frame extraction and use existing frames")
    parser.add_argument("--merge_existing", action="store_true",
                      help="Merge with existing YOLO annotations (if any)")
    parser.add_argument("--visualize", action="store_true",
                      help="Visualize sample annotations for verification")
    parser.add_argument("--train", action="store_true",
                      help="Run YOLOv8 training after conversion")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for training (default: 16)")
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of training epochs (default: 50)")
    parser.add_argument("--model_size", type=str, default="s", choices=['n', 's', 'm', 'l', 'x'],
                      help="YOLOv8 model size (default: s)")
    parser.add_argument("--margin", type=float, default=0.05,
                      help="Safety margin for bounding boxes (default: 0.05 = 5%)")
    
    args = parser.parse_args()
    
    # Set default frames directory if not provided
    if not args.frames_dir:
        args.frames_dir = os.path.join(args.output_dir, "frames")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Extract frames if needed
    if not args.skip_frame_extraction:
        logger.info("Step 1: Extracting frames from videos")
        frame_dimensions = extract_frames_from_videos(args.video_dir, args.frames_dir, args.sample_rate)
    else:
        logger.info("Skipping frame extraction as requested")
    
    # Step 2: Convert COCO annotations to YOLO format
    logger.info("Step 2: Converting COCO annotations to YOLO format")
    yaml_path = convert_coco_to_yolo(args.coco_json, args.output_dir, args.frames_dir, args.merge_existing)
    
    # Step 3: Validate the dataset
    logger.info("Step 3: Validating YOLO dataset")
    validate_yolo_dataset(yaml_path)
    
    # Step 4: Visualize samples if requested
    if args.visualize:
        logger.info("Step 4: Visualizing sample annotations")
        visualize_dataset_samples(yaml_path, os.path.join(args.output_dir, "visualizations"))
    
    # Step 5: Run YOLOv8 training if requested
    if args.train:
        # Import train function from the companion script
        # If not available, try to use the function directly
        try:
            logger.info("Step 5: Starting YOLOv8 training")
            
            # First try to import directly from the companion script
            try:
                sys.path.append(os.path.dirname(os.path.abspath(__file__)))
                from paste import train_yolov8
            except ImportError:
                # If that fails, try to load from paste-3.txt provided
                from importlib.util import spec_from_file_location, module_from_spec
                spec = spec_from_file_location("paste_module", "paste-3.txt")
                paste_module = module_from_spec(spec)
                spec.loader.exec_module(paste_module)
                train_yolov8 = paste_module.train_yolov8
            
            train_yolov8(
                yaml_path,
                Path(args.output_dir).parent,  # Use parent directory as output
                batch_size=args.batch_size,
                epochs=args.epochs,
                model_size=args.model_size,
                margin=args.margin
            )
        except Exception as e:
            logger.warning(f"Could not run training automatically: {e}")
            logger.info("You can run training separately using:")
            logger.info(f"python paste.py --data_yaml {yaml_path}")
    
    logger.info(f"Processing complete. YOLO dataset configuration: {yaml_path}")
    logger.info(f"To train with YOLOv8, run: python paste.py --data_yaml {yaml_path}")

if __name__ == "__main__":
    main()
