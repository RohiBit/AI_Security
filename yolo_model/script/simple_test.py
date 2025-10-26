from ultralytics import YOLO
import cv2
import os
from datetime import datetime

# Load YOLO model
model = YOLO(r"F:\projects\epics\final\yolo_model\script\runs_4\exp\weights\best.pt")

# Define labels and colors
labels = ["walking", "running", "standing", "falling", "snatching", "fighting", "siting", "driving"]

action_colors = {
    0: (255, 255, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 165, 255),
    4: (0, 0, 255),
    5: (0, 225, 225),
    6: (0, 225, 165),
    7: (225, 165, 0)
}

# Input and Output Video
input_video_path = r"F:\projects\epics\final\input\24.mp4"
output_video_path = r"F:\projects\epics\final\output_24.mp4"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

cap = cv2.VideoCapture(input_video_path)
if not cap.isOpened():
    print("Error opening video.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)
fps = cap.get(cv2.CAP_PROP_FPS)

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

zoom_factor = 0.5
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Resize frame
    frame = cv2.resize(frame, (width, height))

    # Run YOLO prediction
    results = model.predict(frame, imgsz=640, conf=0.5, show=False)

    # Draw bounding boxes
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            label = labels[cls_id] if cls_id < len(labels) else "unknown"
            color = action_colors.get(cls_id, (128, 128, 128))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Write to output video
    out.write(frame)

    # Optional: Show real-time detection (press 'q' to quit)
    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"\n✅ Output saved to: {output_video_path}")
