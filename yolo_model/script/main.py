import os
import torch
from ultralytics import YOLO

# Set custom temp and torch cache directories
os.environ["TEMP"] = "F:/temp"
os.environ["TMP"] = "F:/temp"
os.environ["TORCH_HOME"] = "F:/temp"

# Create necessary directories if they don't exist
os.makedirs("F:/temp", exist_ok=True)
os.makedirs(r"F:\projects\epics\final\yolo_model\script\runs_4", exist_ok=True)

if __name__ == "__main__":
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the YOLO model (you can replace 'yolov8n.pt' with a custom model path if needed)
    model = YOLO("yolov8n.pt")

    # Start training with improved parameters to handle overfitting and increase epochs
    model.train(
        data=r"F:\projects\epics\final\yolo_model\dataset\data.yaml",  # Path to data.yaml
        epochs=50,             # Increased number of training epochs
        imgsz=640,              # Image size
        batch=16,               # Batch size
        device=device,          # GPU or CPU
        workers=0,              # Disable multiprocessing for Windows
        cache=False,            # Avoid caching for low memory
        project=r"F:\projects\epics\final\yolo_model\script\runs_4",  # Custom output directory
        name="exp",             # Folder name inside 'runs_3'
        exist_ok=True,          # Overwrite folder if it already exists
        augment=True,           # Enable data augmentation
        lr0=0.01,               # Initial learning rate
        weight_decay=0.0005,  
    )
