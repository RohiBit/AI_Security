from ultralytics import YOLO
import cv2
import face_recognition
import pickle
import os
from datetime import datetime

# Import Twilio client
from twilio.rest import Client

# Twilio configuration - replace with your actual credentials and phone numbers
TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TWILIO_FROM_NUMBER = os.environ.get('TWILIO_FROM_NUMBER')
TWILIO_TO_NUMBER = os.environ.get('DEFAULT_ALERT_NUMBER')    # Destination phone number to receive alerts

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Load trained YOLO model
model = YOLO(r"F:\projects\epics\final\yolo_model\script\runs\detect\train11\weights\best.pt")

# Load known criminal encodings
with open(r'F:\projects\epics\final\face_detection\known_faces_encodings.pkl', 'rb') as f:
    known_encodings, known_names = pickle.load(f)

# Open video file
cap = cv2.VideoCapture(0)  # 0 is usually the default webcam

# Define colors for each action
action_colors = {
    0: (255, 255, 255),  # Walking - White
    1: (0, 255, 0),      # Running - Green
    2: (255, 0, 0),      # Standing - Blue
    3: (0, 165, 255),    
    4: (0, 0, 255),      
}

# Define label names
labels = ["walking", "running", "standing", "fall", "snatching"]

# Create necessary folders if not exist
os.makedirs('snatching_detections', exist_ok=True)

# Log file for detected criminals
log_file = open('detection_log.txt', 'a')

# Frame skipping factor (Increase this value to process video faster, e.g., 2 for every 2nd frame)
frame_skip = 2
frame_count = 0

# Zoom factor for resizing the frame (Use a value less than 1.0 to make it smaller)
zoom_factor = 0.5  # Adjusted to a safer default value

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Frame could not be read or video ended.")
        break  # Stop if the video ends

    # Skip frames for faster processing
    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    # Resize frame using zoom factor if valid
    if frame is not None and frame.size > 0:
        new_width = int(frame.shape[1] * zoom_factor)
        new_height = int(frame.shape[0] * zoom_factor)
        if new_width > 0 and new_height > 0:
            frame = cv2.resize(frame, (new_width, new_height))

    # Run YOLO inference
    results = model.predict(frame, imgsz=640, conf=0.5, show=False, vid_stride=1)

    # Draw detections on frame
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Moved here
    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls_id = int(box.cls[0])  # Class ID
            
            if cls_id == 4:  # If action is 'snatching'
                # Snatching bounding box coordinates
                snatch_x1, snatch_y1, snatch_x2, snatch_y2 = x1, y1, x2, y2

                # Crop the whole person involved in snatching
                snatching_person = frame[snatch_y1:snatch_y2, snatch_x1:snatch_x2]

                # Save the cropped whole person image with timestamp
                person_img_path = f'snatching_detections/person_{timestamp}.jpg'
                cv2.imwrite(person_img_path, snatching_person)

                # Prepare to send SOS message
                sos_message = f"SOS Alert: Snatching detected at {timestamp}."

                # Detect faces in the whole frame
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)

                # Find faces that overlap with snatching bounding box
                for i, (face_encoding, face_location) in enumerate(zip(face_encodings, face_locations)):
                    top, right, bottom, left = face_location

                    # Check overlap with snatching bounding box
                    overlap_x1 = max(left, snatch_x1)
                    overlap_y1 = max(top, snatch_y1)
                    overlap_x2 = min(right, snatch_x2)
                    overlap_y2 = min(bottom, snatch_y2)

                    if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                        # This face overlaps with snatching bounding box, save it
                        face_img = frame[top:bottom, left:right]

                        matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                        name = "Unknown"

                        if True in matches:
                            matched_index = matches.index(True)
                            name = known_names[matched_index]

                            # Append known criminal info to SOS message
                            sos_message += f" Known criminal detected: {name}."

                        # Save the face image
                        face_img_path = f'snatching_detections/face_{timestamp}_{i}.jpg'
                        cv2.imwrite(face_img_path, face_img)

                        # Draw rectangle and name on frame
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                        # Log the detection
                        if name != "Unknown":
                            log_entry = f"Criminal detected: {name} at {timestamp}\n"
                            log_file.write(log_entry)
                            print(log_entry)

                # Optionally, draw snatching bounding box on frame
                cv2.rectangle(frame, (snatch_x1, snatch_y1), (snatch_x2, snatch_y2), (0, 0, 255), 2)

                # Send SOS SMS via Twilio
                try:
                    message = twilio_client.messages.create(
                        body=sos_message,
                        from_=TWILIO_FROM_NUMBER,
                        to=TWILIO_TO_NUMBER
                    )
                    print(f"Sent SOS SMS: {message.sid}")
                except Exception as e:
                    print(f"Error sending SOS SMS: {e}")

            # Draw bounding box and label on the original frame
            label = labels[cls_id] if cls_id in range(len(labels)) else "unknown"
            color = action_colors.get(cls_id, (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Save the whole frame with bounding boxes for reference
    # full_frame_path = f'snatching_detections/frame_{timestamp}.jpg'
    # cv2.imwrite(full_frame_path, frame)

    # Show real-time video (Resized frame)
    cv2.imshow("YOLO Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break  # Press 'q' to exit

    frame_count += 1

# Release resources
cap.release()
cv2.destroyAllWindows()
log_file.close()
