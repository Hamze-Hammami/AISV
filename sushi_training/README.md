## sushi_training
this directory is for the scripts used to train Yolo object detection and the Sam2 distilled U-NET water segmentaion models. 

For the depth estimation model, refer to the official Depth-Anything V2 GitHub: [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)

## Pre-Trained Models

The debris detection and water segmentation models are hosted on Hugging Face: [AISV_Models](https://huggingface.co/Hamze-Hammami/AISV_Models)

---

## Training From Scratch
 
### Requirements
 
```bash
pip install ultralytics torch torchvision segmentation-models-pytorch
pip install albumentations av pillow opencv-python scikit-learn
pip install huggingface_hub pycocotools pyyaml tqdm scipy
```
 
---
 
### YOLO — Obstacle & Goal Detection
 
#### Step 1 — Annotate
 
Place your `.mp4` or `.avi` videos in a `dataset/` folder, then run:
 
```bash
cd yolo/annotation
python annotate_yolo.py
```
 
This opens a GUI. Draw bounding boxes on frames and navigate with arrow keys. The tracker carries boxes forward automatically, however make sure to check it's output not alwasy optimal.
Output is saved to `bbox_annotation_data/`.
 
#### Step 2 — Prepare Dataset
 
```bash
cd yolo/training
python prepare_yolo.py --coco_json ../../bbox_annotation_data/coco/annotations.json
```
 
This extracts frames, converts COCO annotations to YOLO format, and creates an 80/10/10 train/val/test split.
Output is saved to `yolo_data/yolo_dataset/dataset.yaml`.
 
#### Step 3 — Fix Bounding Boxes
 
```bash
python fix_yolo.py --label_dir yolo_data/yolo_dataset/train/labels
python fix_yolo.py --label_dir yolo_data/yolo_dataset/val/labels
python fix_yolo.py --label_dir yolo_data/yolo_dataset/test/labels
```
 
#### Step 4 — Train
 
```bash
python train_yolo.py --data_yaml yolo_data/yolo_dataset/dataset.yaml
```
 
#### Step 5 — Test
 
```bash
python test_yolo.py --model trained_models/best_yolov8_model.pt --source your_video.mp4
```
 
---
 
### U-Net — Water Segmentation
 
#### Step 1 — Annotate
 
Place your `.mp4` or `.avi` videos in a `dataset/` folder, then run:
 
```bash
cd unet/annotation
python annotate_unet.py
```
 
This opens a GUI. Draw bounding boxes or click points (this is alot better BBox is meant for other annotaions like objects) on water regions — SAM2 generates the mask automatically. Annotated frames and masks are saved to `mask_annotation_data/`.
 
#### Step 2 — Train
 
```bash
cd unet/training
python train_unet.py --annotation_dir mask_annotation_data
```
 
#### Step 3 — Test
 
```bash
python test_unet.py --model trained_models/best_unet_model.pth --input your_video.mp4
```
 
---
 
## Repository Structure
 
```
AISV/
├── yolo/
│   ├── annotation/
│   │   └── annotate_yolo.py          # Bounding box annotation GUI (YOLOE + CV tracker)
│   └── training/
│       ├── prepare_yolo.py            # Extract frames from video + convert COCO → YOLO format
│       ├── fix_yolo.py           # Fix bounding box coordinates after conversion
│       ├── train_yolo.py         # YOLOv8 training + augmentation + ONNX export
│       └── test_yolo.py          # Quick inference test for trained YOLO model
│
└── unet/
    ├── annotation/
    │   └── annotate_unet.py         # Water mask annotation GUI (SAM2-powered)
    └── training/
        ├── train_unet.py    # U-Net training + dataset preparation + ONNX export
        └── test_unet.py           # Inference test for trained U-Net model
```
 
---
 
## Model Conversion to TensorRT
 
Converting models to TensorRT is required for deployment on the Jetson and model optmization in general.
 
### Depth Estimation
 
Use the following repository to convert Depth-Anything V2 to TensorRT:
 
**[Depth-Anythingv2-TensorRT-python](https://github.com/zhujiajian98/Depth-Anythingv2-TensorRT-python)**
  
### Segmentation & Detection
 
Conversion scripts for the YOLO and U-Net models are available in the `conversion scripts` folder on Hugging Face:
  
### Object Tracking (Optional)
 
For running YOLOv8 with TensorRT-accelerated tracking on Jetson:
 
**[YOLOv8 Object Tracking TensorRT](https://github.com/nabang1010/YOLOv8_Object_Tracking_TensorRT/tree/main/srcs)**
 
---
 
## What Each File Does
 
| File | Purpose |
|---|---|
| `yolo/annotation/annotate_yolo.py` | GUI tool for drawing bounding boxes on video frames. Uses YOLOE for assisted detection and OpenCV trackers to propagate boxes across frames. Outputs COCO JSON annotations. |
| `yolo/training/prepare_yolo.py` | Reads COCO annotations, extracts frames from video, converts to YOLO label format, and splits into train/val/test. |
| `yolo/training/fix_yolo.py` | Clamps bounding box coordinates to stay within valid image bounds after conversion. |
| `yolo/training/train_yolo.py` | Trains a YOLOv8 model with optional crop augmentation. Validates and fixes labels before training. Exports the best checkpoint to ONNX. |
| `yolo/training/test_yolo.py` | Runs a trained YOLO model on an image, video, or webcam for quick inference testing. |
| `unet/annotation/annotate_unet.py` | GUI tool for annotating water masks using SAM2. Supports bounding box and point-click prompts. Saves annotated frames and binary masks. |
| `unet/training/train_unet.py` | Organises annotated mask/frame pairs into splits, trains a U-Net (ResNet34 encoder), and exports the best model to ONNX. |
| `unet/training/test_unet.py` | Runs a trained U-Net on an image, video, or webcam and overlays the water segmentation mask. |
 
---
 
## Notes
 
- Only annotated keyframes are used for U-Net training — no full video extraction is needed.
- Class 2 ("Other") YOLO annotations are stripped during dataset preparation; those images are kept as background examples.
-  models export to ONNX with `opset=12–13` for Jetson compatibility via TensorRT.
