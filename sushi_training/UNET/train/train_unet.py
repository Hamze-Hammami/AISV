import os
import shutil
import random
import yaml
import json
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import av
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class DatasetPreparation:
    """Prepare annotated data for U-Net training (water segmentation)"""

    def __init__(self, annotation_dir, output_dir):
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.unet_dir = self.output_dir / 'unet'
        self.video_dir = Path('dataset')
        self.train_ratio = 0.8
        self.val_ratio = 0.1
        self.test_ratio = 0.1

    def setup_directories(self):
        """Create necessary directories for training"""
        # U-Net directories
        for split in ['train', 'val', 'test']:
            (self.unet_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.unet_dir / split / 'masks').mkdir(parents=True, exist_ok=True)

    def extract_frames(self):
        """Extract frames from videos if they don't exist already"""
        frames_dir = self.annotation_dir / 'frames'
        frames_dir.mkdir(exist_ok=True)

        # Check if frames have already been extracted
        if list(frames_dir.glob('*.jpg')):
            logger.info(f"Found existing frames in {frames_dir}, skipping extraction")
            return

        logger.info("Extracting frames from video files...")
        video_files = list(self.video_dir.glob('*.mp4')) + list(self.video_dir.glob('*.avi'))

        for video_path in tqdm(video_files, desc="Extracting video frames"):
            try:
                container = av.open(str(video_path))
                video_name = video_path.stem
                for i, frame in enumerate(container.decode(video=0)):
                    pil_frame = frame.to_image()
                    frame_path = frames_dir / f"video_{video_name}_frame_{i}.jpg"
                    pil_frame.save(frame_path)
                    if i % 100 == 0:
                        logger.info(f"Extracted {i} frames from {video_name}")
            except Exception as e:
                logger.error(f"Error extracting frames from {video_path}: {e}")

    def organize_data_for_unet(self):
        """Organize annotated data for U-Net training (water segmentation)"""
        logger.info("Organizing data for U-Net training...")

        # Get all mask files and corresponding frames
        masks_dir = self.annotation_dir / 'masks'
        frames_dir = self.annotation_dir / 'frames'
        if not frames_dir.exists():
            frames_dir = Path('manual_annotation_data/frames')
            if not frames_dir.exists():
                self.extract_frames()
                frames_dir = self.annotation_dir / 'frames'

        all_masks = list(masks_dir.glob('*_mask.png'))

        # Split data into train/val/test sets
        train_masks, temp_masks = train_test_split(all_masks, train_size=self.train_ratio, random_state=42)
        val_ratio_adjusted = self.val_ratio / (self.val_ratio + self.test_ratio)
        val_masks, test_masks = train_test_split(temp_masks, train_size=val_ratio_adjusted, random_state=42)

        # Copy files to appropriate directories
        self._copy_files_for_unet(train_masks, 'train', frames_dir)
        self._copy_files_for_unet(val_masks, 'val', frames_dir)
        self._copy_files_for_unet(test_masks, 'test', frames_dir)

        logger.info(f"U-Net dataset prepared: {len(train_masks)} train, {len(val_masks)} val, {len(test_masks)} test")

    def _copy_files_for_unet(self, mask_files, split, frames_dir):
        """Copy files to U-Net directory structure"""
        for mask_path in tqdm(mask_files, desc=f"Copying {split} files for U-Net"):
            # Copy mask file
            dest_mask = self.unet_dir / split / 'masks' / mask_path.name

            # Ensure mask is binary (0 or 255)
            mask = Image.open(mask_path)
            mask_array = np.array(mask)
            binary_mask = (mask_array > 128).astype(np.uint8) * 255
            Image.fromarray(binary_mask).save(dest_mask)

            # Find and copy corresponding image
            base_name = mask_path.stem.replace('_mask', '')
            image_path = frames_dir / f"{base_name}.jpg"
            if not image_path.exists():
                # Try finding the image in visualization directory
                viz_dir = self.annotation_dir / 'visualizations'
                if viz_dir.exists():
                    for ext in ['.jpg', '.png']:
                        alt_path = viz_dir / f"{base_name}{ext}"
                        if alt_path.exists():
                            image_path = alt_path
                            break

            if image_path.exists():
                dest_image = self.unet_dir / split / 'images' / f"{base_name}.jpg"
                # Convert to JPG if it's not already
                if image_path.suffix.lower() == '.jpg':
                    shutil.copy(image_path, dest_image)
                else:
                    img = Image.open(image_path).convert('RGB')
                    img.save(dest_image, 'JPEG')
            else:
                logger.warning(f"Image for {base_name} not found")

    def prepare_datasets(self):
        """Prepare U-Net dataset"""
        self.setup_directories()
        self.organize_data_for_unet()
        logger.info("Dataset preparation for U-Net completed!")


class WaterSegmentationDataset(Dataset):
    """Dataset for water segmentation with U-Net"""

    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform
        self.images = sorted(list(self.images_dir.glob('*.jpg')))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks_dir / f"{img_path.stem}_mask.png"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path.exists():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            # Ensure binary mask with proper values (0 for background, 1 for water)
            mask = (mask > 128).astype(np.uint8)  # Binary mask: 0 or 1
        else:
            logger.warning(f"Mask not found for {img_path.name}, using empty mask")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        return image, mask.unsqueeze(0)  # Add channel dimension to mask


class UNetTrainer:
    """Trainer for U-Net water segmentation model"""

    def __init__(self, data_dir, output_dir):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 4
        self.learning_rate = 1e-4
        self.num_epochs = 50
        self.image_size = (320, 320)

        logger.info(f"Using device: {self.device}")

    def get_transforms(self):
        """Get transformations for training and validation data"""
        train_transform = A.Compose([
            A.Resize(*self.image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.3, contrast_limit=0.3),
            A.RandomGamma(p=0.3, gamma_limit=(80, 120)),
            A.HueSaturationValue(p=0.3, hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=15),
            A.CLAHE(p=0.3, clip_limit=(1, 4), tile_grid_size=(8, 8)),
            A.OneOf([
                A.RandomShadow(p=1, num_shadows_lower=1, num_shadows_upper=3),
                A.RandomRain(p=1, blur_value=2),
                A.RandomSunFlare(p=1, flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1),
            ], p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])

        val_transform = A.Compose([
            A.Resize(*self.image_size),
            A.Normalize(),
            ToTensorV2(),
        ])

        return train_transform, val_transform

    def create_dataloaders(self):
        """Create dataloaders for training and validation"""
        train_transform, val_transform = self.get_transforms()

        train_dataset = WaterSegmentationDataset(
            self.data_dir / 'train' / 'images',
            self.data_dir / 'train' / 'masks',
            transform=train_transform
        )

        val_dataset = WaterSegmentationDataset(
            self.data_dir / 'val' / 'images',
            self.data_dir / 'val' / 'masks',
            transform=val_transform
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return train_loader, val_loader

    def train(self):
        """Train U-Net model for water segmentation"""
        # Create model using segmentation_models_pytorch with a ResNet34 encoder
        model = smp.Unet(
            encoder_name="resnet34",      # Use ResNet34 as encoder
            encoder_weights="imagenet",     # Pre-trained on ImageNet
            in_channels=3,                  # RGB input
            classes=1,                      # Binary segmentation
            activation='sigmoid'            # Sigmoid activation for binary segmentation
        )

        model = model.to(self.device)

        # Define loss function and optimizer
        criterion = smp.losses.DiceLoss(mode='binary')
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders()

        best_val_loss = float('inf')

        for epoch in range(1, self.num_epochs + 1):
            model.train()
            epoch_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{self.num_epochs}")

            for images, masks in progress_bar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            train_loss = epoch_loss / len(train_loader)

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(self.device)
                    masks = masks.to(self.device)

                    outputs = model(images)
                    loss = criterion(outputs, masks)
                    val_loss += loss.item()

            val_loss = val_loss / len(val_loader)
            scheduler.step(val_loss)

            logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), self.output_dir / 'best_unet_model.pth')
                logger.info(f"Saved new best model with Val Loss = {val_loss:.4f}")

        # Save final model and checkpoint
        torch.save(model.state_dict(), self.output_dir / 'final_unet_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': self.num_epochs,
            'val_loss': val_loss,
        }, self.output_dir / 'checkpoint.pth')

        logger.info("U-Net training completed!")

    def export_to_onnx(self):
        """Export U-Net model to ONNX format"""
        unet_model_path = self.output_dir / 'best_unet_model.pth'
        if unet_model_path.exists():
            try:
                logger.info("Exporting U-Net model to ONNX...")
                # Create a dummy input tensor
                dummy_input = torch.randn(1, 3, 320, 320, device='cuda' if torch.cuda.is_available() else 'cpu')

                # Initialize the model architecture and load the trained weights
                model = smp.Unet(
                    encoder_name="resnet34",
                    encoder_weights="imagenet",
                    in_channels=3,
                    classes=1,
                    activation='sigmoid'
                )
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.load_state_dict(torch.load(unet_model_path, map_location=device))
                model.to(device)
                model.eval()

                # Export to ONNX
                torch.onnx.export(
                    model,
                    dummy_input,
                    str(self.output_dir / 'best_unet_model.onnx'),
                    export_params=True,
                    opset_version=12,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                logger.info(f"U-Net ONNX model saved to {self.output_dir / 'best_unet_model.onnx'}")
            except Exception as e:
                logger.error(f"Error exporting U-Net model to ONNX: {e}")
        else:
            logger.error("U-Net model file not found, cannot export to ONNX.")

class ModelTrainer:
    """Trainer for U-Net model"""

    def __init__(self, annotation_dir, output_dir):
        self.annotation_dir = Path(annotation_dir)
        self.output_dir = Path(output_dir)
        self.unet_dir = self.output_dir / 'unet'

        if not self.annotation_dir.exists():
            raise ValueError(f"Annotation directory {self.annotation_dir} does not exist")

    def prepare_data(self):
        """Prepare the U-Net dataset"""
        data_prep = DatasetPreparation(self.annotation_dir, self.output_dir)
        data_prep.prepare_datasets()

    def train_unet(self):
        """Train U-Net water segmentation model"""
        logger.info("Starting U-Net training...")
        unet_trainer = UNetTrainer(self.unet_dir, self.output_dir)
        unet_trainer.train()

    def export_to_onnx(self):
        """Export U-Net model to ONNX format"""
        logger.info("Exporting U-Net model to ONNX format...")
        unet_trainer = UNetTrainer(self.unet_dir, self.output_dir)
        unet_trainer.export_to_onnx()

    def train_models(self):
        """Prepare dataset, train U-Net model, and export to ONNX"""
        self.prepare_data()
        self.train_unet()
        self.export_to_onnx()
        logger.info("U-Net model training and export completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Train U-Net model for water segmentation on annotated data")
    parser.add_argument("--annotation_dir", type=str, default="manual_annotation_data",
                        help="Path to directory containing annotations")
    parser.add_argument("--output_dir", type=str, default="trained_models",
                        help="Path to directory to save trained models")
    parser.add_argument("--prepare_only", action="store_true", 
                        help="Only prepare datasets without training")
    args = parser.parse_args()

    trainer = ModelTrainer(args.annotation_dir, args.output_dir)

    if args.prepare_only:
        trainer.prepare_data()
        logger.info("Dataset preparation completed. Skipping training.")
    else:
        trainer.train_models()


if __name__ == "__main__":
    main()

