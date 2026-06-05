#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
import time
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision import transforms

################################################################################
# UNet Model (same as in training)
################################################################################
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, rate, pooling=True):
        super(Encoder, self).__init__()
        self.rate = rate
        self.pooling = pooling
        self.bn = nn.BatchNorm2d(in_channels)
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.drop = nn.Dropout(rate)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, x):
        x = self.bn(x)
        x = nn.ReLU()(self.c1(x))
        x = self.drop(x)
        x = nn.ReLU()(self.c2(x))
        if self.pooling:
            y = self.pool(x)
            return y, x
        else:
            return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels, rate):
        super(Decoder, self).__init__()
        self.rate = rate
        self.bn = nn.BatchNorm2d(in_channels)
        self.cT = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.c1 = nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.drop = nn.Dropout(rate)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, skip_x):
        x = self.bn(x)
        x = nn.ReLU()(self.cT(x))
        x = torch.cat([x, skip_x], dim=1)
        x = nn.ReLU()(self.c1(x))
        x = self.drop(x)
        x = nn.ReLU()(self.c2(x))
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.initial = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc1 = Encoder(64, 64, 0.1)
        self.enc2 = Encoder(64, 128, 0.1)
        self.enc3 = Encoder(128, 256, 0.2)
        self.enc4 = Encoder(256, 512, 0.2)
        self.enc5 = Encoder(512, 512, 0.3, pooling=False)
        self.dec1 = Decoder(512, 512, 512, 0.2)
        self.dec2 = Decoder(512, 256, 256, 0.2)
        self.dec3 = Decoder(256, 128, 128, 0.1)
        self.dec4 = Decoder(128, 64, 64, 0.1)
        self.final = nn.Conv2d(64, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = nn.ReLU()(self.initial(x))
        p1, c1 = self.enc1(x)
        p2, c2 = self.enc2(p1)
        p3, c3 = self.enc3(p2)
        p4, c4 = self.enc4(p3)
        e = self.enc5(p4)
        d1 = self.dec1(e, c4)
        d2 = self.dec2(d1, c3)
        d3 = self.dec3(d2, c2)
        d4 = self.dec4(d3, c1)
        return torch.sigmoid(self.final(d4))

################################################################################
# Inference Functions
################################################################################
def load_model(model_path, device):
    """Load the trained UNet model"""
    model = UNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def preprocess_image(image, target_size=(256, 256)):
    """Preprocess image for model input"""
    if isinstance(image, np.ndarray):
        # Convert OpenCV BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        image_pil = image
    else:
        raise ValueError("Input must be numpy array or PIL Image")
    
    # Resize and convert to tensor
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(image_pil)
    return input_tensor.unsqueeze(0), image_pil  # Add batch dimension

def create_visualization(original, mask, alpha=0.5, simple=False):
    """Create a visualization with colored mask overlay"""
    print("Starting visualization creation")
    # Ensure mask is numpy array 
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Convert original to numpy array for OpenCV processing
    if isinstance(original, Image.Image):
        original_np = np.array(original)
    else:
        original_np = original.copy()
        
    # Convert BGR to RGB if needed (for OpenCV input)
    if len(original_np.shape) == 3 and original_np.shape[2] == 3:
        original_np = cv2.cvtColor(original_np, cv2.COLOR_BGR2RGB)
    
    # Create a blue overlay for water
    water_overlay = np.zeros_like(original_np)
    water_overlay[:, :, 0] = 255  # Blue channel
    
    # Get and resize mask to match original image dimensions
    mask_np = mask[0, 0]  # Remove batch and channel dimensions
    mask_resized = cv2.resize(mask_np, (original_np.shape[1], original_np.shape[0]), interpolation=cv2.INTER_NEAREST)
    binary_mask = (mask_resized > 0.5).astype(np.uint8)
    print("Mask processed")
    
    # Ensure binary mask is broadcastable to 3 channels
    binary_mask_3d = np.repeat(binary_mask[:, :, np.newaxis], 3, axis=2)
    
    # Create visualization
    vis = original_np.copy()
    blended = cv2.addWeighted(original_np, 1-alpha, water_overlay, alpha, 0)
    
    # Apply mask to all channels
    vis = np.where(binary_mask_3d, blended, vis)
    print("Visualization blended")
    
    # Convert back to BGR for OpenCV
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    
    # Draw contour around water (optional, skipped if simple=True)
    if not simple:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, (0, 255, 255), 2)  # Yellow contour
        print("Contours drawn")
    
    print("Visualization complete")
    return vis

def infer_image(model, image_path, device, target_size=(256, 256), visualize=True, output_path=None):
    """Run inference on a single image"""
    print("Starting image inference")
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    input_tensor, image_pil = preprocess_image(image, target_size)
    
    # Run inference
    with torch.no_grad():
        input_tensor = input_tensor.to(device)
        start_time = time.time()
        output = model(input_tensor)
        inference_time = time.time() - start_time
        
    print(f"Image inference time: {inference_time:.4f} seconds")
    
    # Create visualization
    if visualize or output_path:
        visualization = create_visualization(image_pil, output.cpu().numpy())
        
        if visualize:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB))
            plt.title(f"Water Segmentation (Inference time: {inference_time:.4f}s)")
            plt.axis('off')
            plt.show()
        
        if output_path:
            cv2.imwrite(output_path, visualization)
            print(f"Saved visualization to {output_path}")
    
    return output.cpu().numpy()

def infer_video(model, video_path, device, target_size=(256, 256), output_path=None, display=False, simple_vis=False):
    """Run inference on a video file or webcam"""
    print("Starting video inference")
    # Open video capture
    if video_path == 0 or video_path.lower() == 'webcam':
        cap = cv2.VideoCapture(0)
        print("Opened webcam for inference")
    else:
        cap = cv2.VideoCapture(video_path)
        print(f"Opened video: {video_path}")
    
    if not cap.isOpened():
        print("Error: Could not open video source")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video info: FPS={fps}, Width={width}, Height={height}")
    
    # Set up video writer if needed
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            print("Error: Failed to initialize video writer")
            cap.release()
            return
        print(f"Video writer initialized for {output_path}")
    
    try:
        frame_count = 0
        total_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                if video_path != 0 and video_path.lower() != 'webcam':
                    print("End of video")
                break
            
            print(f"Processing frame {frame_count + 1}")
            
            # Preprocess frame and get the resized image for visualization
            input_tensor, frame_pil = preprocess_image(frame, target_size)
            print("Frame preprocessed")
            
            # Run inference
            with torch.no_grad():
                input_tensor = input_tensor.to(device)
                start_time = time.time()
                output = model(input_tensor)
                inference_time = time.time() - start_time
                total_time += inference_time
            
            print(f"Frame {frame_count + 1} inference time: {inference_time:.4f}s")
            
            # Use the preprocessed (resized) image for visualization
            visualization = create_visualization(frame_pil, output.cpu().numpy(), simple=simple_vis)
            print("Visualization created")
            
            # Add FPS information
            cv2.putText(
                visualization, 
                f"Inference: {1/inference_time:.1f} FPS", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
            print("FPS text added")
            
            # Display frame in window if enabled
            if display:
                cv2.imshow("Water Segmentation", visualization)
                print("Frame displayed")
            
            # Save frame if writer exists
            if writer:
                writer.write(visualization)
                print(f"Frame saved to {output_path}")
            
            frame_count += 1
            
            # Clear CUDA cache periodically to prevent memory issues
            if device.type == "cuda" and frame_count % 5 == 0:
                torch.cuda.empty_cache()
                print("Cleared CUDA cache")
            
            # Exit if 'q' is pressed (only if display is enabled)
            if display and cv2.waitKey(1) & 0xFF == ord('q'):
                print("User interrupted with 'q'")
                break
        
        if frame_count > 0:
            print(f"Processed {frame_count} frames")
            print(f"Average inference time: {total_time/frame_count:.4f}s ({frame_count/total_time:.1f} FPS)")
        else:
            print("No frames processed")
    
    except Exception as e:
        print(f"Error during video inference: {e}")
    
    finally:
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print("Resources released")

################################################################################
# Main Function
################################################################################
def main():
    parser = argparse.ArgumentParser(description="Water Segmentation UNet Inference")
    parser.add_argument("--model", type=str, default="water_segmentation_model.pth", 
                        help="Path to trained model")
    parser.add_argument("--input", type=str, default=None, 
                        help="Path to input image or video (use 'webcam' for webcam)")
    parser.add_argument("--output", type=str, default=None, 
                        help="Path to save output visualization")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256], 
                        help="Target size for model input (width height)")
    parser.add_argument("--no-visualize", action="store_true", 
                        help="Disable visualization (for image input)")
    parser.add_argument("--display", action="store_true", 
                        help="Enable real-time display during video inference")
    parser.add_argument("--simple-vis", action="store_true", 
                        help="Use simplified visualization (no contours)")
    
    args = parser.parse_args()
    
    # Set device to CUDA if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(args.model, device)
    print(f"Model loaded from {args.model}")
    
    # Run inference based on input type
    if args.input is None or args.input.lower() == 'webcam':
        print("Starting webcam inference...")
        infer_video(model, 0, device, tuple(args.size), args.output, display=args.display, simple_vis=args.simple_vis)
    elif os.path.isfile(args.input):
        # Check if it's an image or video
        ext = os.path.splitext(args.input)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            print(f"Running inference on image: {args.input}")
            infer_image(model, args.input, device, tuple(args.size), not args.no_visualize, args.output)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            print(f"Running inference on video: {args.input}")
            infer_video(model, args.input, device, tuple(args.size), args.output, display=args.display, simple_vis=args.simple_vis)
        else:
            print(f"Unsupported file type: {ext}")
    else:
        print(f"Input not found: {args.input}")

if __name__ == "__main__":
    main()
