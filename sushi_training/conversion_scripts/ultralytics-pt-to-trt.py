import os
import torch
from ultralytics import YOLO

# Set environment variables for CUDA
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PATH'] = f"{os.environ['CUDA_HOME']}/bin:{os.environ['PATH']}"
os.environ['LD_LIBRARY_PATH'] = f"{os.environ['CUDA_HOME']}/lib64:{os.environ.get('LD_LIBRARY_PATH', '')}"

def verify_cuda_installation():
    print("Verifying CUDA installation...")
    print("CUDA_HOME:", os.environ['CUDA_HOME'])
    print("PATH:", os.environ['PATH'])
    print("LD_LIBRARY_PATH:", os.environ['LD_LIBRARY_PATH'])

def check_cuda_availability():
    print("Checking CUDA availability in PyTorch...")
    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    if cuda_available:
        device_count = torch.cuda.device_count()
        print("CUDA device count:", device_count)
        for i in range(device_count):
            print(f"CUDA device {i}: {torch.cuda.get_device_name(i)}")
    return cuda_available

def load_and_transform_model(cuda_available):
    print("Loading YOLOv8 model...")
    model = YOLO("best_model.pt")
    
    if cuda_available:
        print("CUDA is available. Exporting model to TensorRT format...")
        model.export(format="engine")  # creates 'debris-det.engine'
        tensorrt_model = YOLO("best.engine")
    else:
        print("CUDA is not available. Skipping TensorRT export.")
        tensorrt_model = model
    
    return tensorrt_model

def run_inference(tensorrt_model):
    print("Running inference...")
    results = tensorrt_model("https://ultralytics.com/images/bus.jpg")
    
    print("Inference results:")
    print(results)

if __name__ == "__main__":
    verify_cuda_installation()
    cuda_available = check_cuda_availability()
    tensorrt_model = load_and_transform_model(cuda_available)
    run_inference(tensorrt_model)

