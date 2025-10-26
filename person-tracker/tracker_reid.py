import cv2
import numpy as np
from ultralytics import YOLO
from reid_model.reid_engine import ReIDEngine
from sklearn.metrics.pairwise import cosine_similarity
import os

# Initialize YOLOv8 and ReID
model = YOLO('yolov8m.pt')
reid = ReIDEngine()

# Output folders
base_output = r"F:\projects\epics\final\person-tracker-ai-cctv\output"
os.makedirs(os.path.join(base_output, "cam1"), exist_ok=True)
os.makedirs(os.path.join(base_output, "cam2"), exist_ok=True)
os.makedirs(os.path.join(base_output, "comparisons"), exist_ok=True)

# Load target image(s)
target_img = cv2.imread(r"F:\projects\epics\final\person-tracker-ai-cctv\input\target_fullbody.jpg")
if target_img is None:
    print("❌ Failed to load target image.")
    exit()

# Extract target feature
try:
    target_feat = reid.extract_features(target_img)
except Exception as e:
    print(f"❌ Feature extraction failed: {e}")
    exit()

def get_dominant_color(image):
    img = cv2.resize(image, (50, 50))
    data = img.reshape(-1, 3)
    return np.mean(data, axis=0)

target_color = get_dominant_color(target_img)

# Open video files
cap1 = cv2.VideoCapture(r"F:\projects\epics\final\person-tracker-ai-cctv\videos\cam2.mp4")
cap2 = cv2.VideoCapture(r"F:\projects\epics\final\person-tracker-ai-cctv\videos\cam4.mp4")

frame_count = 0

# Frame processing function
def process_frame(frame, cam_id, frame_count):
    results = model(frame, classes=[0])
    print(f"[Cam{cam_id}] Detected {len(results[0].boxes)} people")

    highest_similarity = 0
    best_coords = None
    best_cropped = None

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        person_img = frame[y1:y2, x1:x2]

        try:
            person_feat = reid.extract_features(person_img)
            similarity = cosine_similarity([target_feat], [person_feat])[0][0]
            detected_color = get_dominant_color(person_img)
            color_diff = np.linalg.norm(target_color - detected_color)

            print(f"[Cam{cam_id}] Similarity: {similarity:.2f}, ColorDiff: {color_diff:.1f}")

            if similarity > highest_similarity:
                highest_similarity = similarity
                best_coords = (x1, y1, x2, y2)
                best_cropped = person_img

        except Exception as e:
            print(f"[Cam{cam_id}] Feature extraction error: {e}")

    # Save best match if similarity > 0.6 or color very close
    if best_cropped is not None and (highest_similarity > 0.85):
        print(f"[Cam{cam_id}] ✅ BEST MATCH FOUND (similarity: {highest_similarity:.2f})")
        x1, y1, x2, y2 = best_coords
        # Calculate dynamic thickness and scale
        # Calculate dynamic thickness and reduced font scale
        box_height = y2 - y1
        thickness = max(2, box_height // 100)
        font_scale = max(0.4, (box_height / 150.0) * 0.8)  # reduced font size slightly

# Draw bounding box and reduced-size text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(frame, f"Matched {highest_similarity:.2f}",
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)


# Draw bounding box and text
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(frame, f"Matched {highest_similarity:.2f}",
            (x1, max(20, y1 - 10)),  # Ensure text doesn't go off-screen
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)


        # Save matched cropped person
        match_path = os.path.join(base_output, f"cam{cam_id}", f"match_{frame_count}.jpg")
        cv2.imwrite(match_path, best_cropped)

        # Save comparison image
        resized_target = cv2.resize(target_img, (128, 256))
        resized_detected = cv2.resize(best_cropped, (128, 256))
        comparison = np.hstack([resized_target, resized_detected])
        comparison_path = os.path.join(base_output, "comparisons", f"compare_cam{cam_id}_{frame_count}.jpg")
        cv2.imwrite(comparison_path, comparison)

    cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    return cv2.resize(frame, (640, 360))

# Main loop
while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 and not ret2:
        break

    if ret1:
        frame1 = process_frame(frame1, cam_id=1, frame_count=frame_count)
        cv2.imshow("Cam 1", frame1)

    if ret2:
        frame2 = process_frame(frame2, cam_id=2, frame_count=frame_count)
        cv2.imshow("Cam 2", frame2)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap1.release()
cap2.release()
cv2.destroyAllWindows()
