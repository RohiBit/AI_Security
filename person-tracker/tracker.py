import cv2
import face_recognition
import numpy as np
from ultralytics import YOLO
import os

# Load YOLOv8 model
model = YOLO('models/yolov8n.pt')

# Load target face encoding
target_img = face_recognition.load_image_file("input/target_face.jpg")
target_encodings = face_recognition.face_encodings(target_img)
if not target_encodings:
    raise Exception("No face found in input image!")
target_encoding = target_encodings[0]

# Load video
video_path = 'videos/cam1.mp4'  # or use 0 for webcam
cap = cv2.VideoCapture(video_path)

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, classes=[0])  # class 0 = person

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box[:4])
        person_crop = frame[y1:y2, x1:x2]
        
        face_locations = face_recognition.face_locations(person_crop)
        if face_locations:
            face_encodings = face_recognition.face_encodings(person_crop, face_locations)
            for encoding in face_encodings:
                similarity = face_recognition.compare_faces([target_encoding], encoding, tolerance=0.5)
                if similarity[0]:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "MATCH FOUND", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    filename = f"output/match_{frame_count}.jpg"
                    cv2.imwrite(filename, frame[y1:y2, x1:x2])
    
    frame_count += 1
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
