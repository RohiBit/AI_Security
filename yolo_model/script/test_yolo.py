from ultralytics import YOLO
import cv2
import face_recognition
import pickle
import os
from datetime import datetime
import math
from twilio.rest import Client
from supabase import create_client, Client as SupabaseClient
import time
import pyshorteners  # <-- Added for URL shortening

TWILIO_ACCOUNT_SID = os.environ.get('TWILIO_ACCOUNT_SID')
TWILIO_AUTH_TOKEN = os.environ.get('TWILIO_AUTH_TOKEN')
TTWILIO_FROM_NUMBER = os.environ.get('TWILIO_FROM_NUMBER')
TWILIO_TO_NUMBER = os.environ.get('DEFAULT_ALERT_NUMBER')

# Initialize Twilio client
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Supabase configuration
SUPABASE_URL = os.getenv('EXPO_PUBLIC_SUPABASE_URL', 'https://mjgctancbhqubfuiwknr.supabase.co')
SUPABASE_ANON_KEY = os.getenv('EXPO_PUBLIC_SUPABASE_ANON_KEY')
supabase: SupabaseClient = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

# YOLO
model = YOLO(r"F:\\projects\\epics\\final\\yolo_model\\script\\runs_4\\exp\\weights\\best.pt")

# Load known faces
with open(r"F:\\projects\\epics\\final\\yolo_model\\face_detection\\known_faces_encodings.pkl", 'rb') as f:
    known_encodings, known_names = pickle.load(f)

# Video input
video_path = r"F:\projects\epics\final\input\6.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Could not open video {video_path}")
    exit()

# Setup
action_colors = {
    0: (255, 255, 255),
    1: (0, 255, 0),
    2: (255, 0, 0),
    3: (0, 165, 255),
    4: (0, 0, 255),
    5: (0, 225, 225),
}

labels = ["walking", "running", "standing", "fall", "snatching", "Fighting"]
os.makedirs('F:/projects/epics/final/snatching_detections', exist_ok=True)
log_file = open('detection_log.txt', 'a')

frame_skip = 1
frame_count = 0
last_snatching_msg_time = None
zoom_factor = 0.5
shortener = pyshorteners.Shortener()


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) *
         math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def fetch_users_with_latest_location():
    try:
        response = supabase.table('users').select('id, phone_number').execute()
        users = []
        for user in response.data:
            latest_loc = supabase.table('location_updates')\
                .select('latitude,longitude')\
                .eq('user_id', user['id'])\
                .order('timestamp', desc=True)\
                .limit(1).execute()
            if latest_loc.data:
                users.append({
                    'phone_number': user['phone_number'],
                    'latitude': latest_loc.data[0]['latitude'],
                    'longitude': latest_loc.data[0]['longitude'],
                })
        return users
    except Exception as e:
        print(f"Error fetching users: {e}")
        return []

# Main loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Video ended.")
        break

    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    new_width = int(frame.shape[1] * zoom_factor)
    new_height = int(frame.shape[0] * zoom_factor)
    frame = cv2.resize(frame, (new_width, new_height))

    results = model.predict(frame, imgsz=640, conf=0.5, show=False)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])

            if cls_id == 4:  # snatching
                sos_message = f"Snatching detected at {timestamp}."

                # Save and upload image
                img_path = f"F:/projects/epics/final/snatching_detections/snatch_{timestamp}.jpg"
                cv2.imwrite(img_path, frame)

                try:
                    with open(img_path, "rb") as f:
                        data = f.read()
                    supabase.storage.from_("snatching").upload(
                        path=f"snatch_{timestamp}.jpg",
                        file=data
                    )
                    signed_resp = supabase.storage.from_("snatching").create_signed_url(
                        path=f"snatch_{timestamp}.jpg",
                        expires_in=60 * 60
                    )
                    signed_url = signed_resp['signedURL']
                    print(f"Signed URL: {signed_url}")

                    # Shorten URL
                    short_url = shortener.tinyurl.short(signed_url)
                    sos_message += f" View: {short_url}"

                except Exception as e:
                    print(f"Upload error: {e}")
                    sos_message += " (Image upload failed)"

                # Face recognition
                face_locations = face_recognition.face_locations(frame)
                face_encodings = face_recognition.face_encodings(frame, face_locations)
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
                    name = "Unknown"
                    if True in matches:
                        idx = matches.index(True)
                        name = known_names[idx]
                        sos_message += f" Known criminal: {name}"
                        log_file.write(f"Criminal: {name} at {timestamp}\n")

                # Send SMS
                current_time = time.time()
                if last_snatching_msg_time is None or (current_time - last_snatching_msg_time >= 10):
                    users = fetch_users_with_latest_location()
                    nearest_user = None
                    min_dist = float('inf')
                    event_lat, event_lon = 11.676938, 78.119946
                    for user in users:
                        dist = haversine_distance(event_lat, event_lon, user['latitude'], user['longitude'])
                        if dist < min_dist:
                            min_dist = dist
                            nearest_user = user
                    if nearest_user:
                        to_number = nearest_user['phone_number']
                        if not to_number.startswith('+'):
                            to_number = '+91' + to_number
                        try:
                            twilio_client.messages.create(
                                body=sos_message,
                                from_=TWILIO_FROM_NUMBER,
                                to=to_number
                            )
                            print(f"Sent SMS to nearest: {to_number}")
                        except Exception as e:
                            print(f"SMS error: {e}")
                    try:
                        twilio_client.messages.create(
                            body=sos_message,
                            from_=TWILIO_FROM_NUMBER,
                            to=DEFAULT_ALERT_NUMBER
                        )
                        print(f"Sent SMS to default: {DEFAULT_ALERT_NUMBER}")
                    except Exception as e:
                        print(f"Default SMS error: {e}")
                    last_snatching_msg_time = current_time
                else:
                    print("Skipping SMS to avoid spamming.")

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            label = labels[cls_id] if cls_id < len(labels) else "unknown"
            color = action_colors.get(cls_id, (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    frame_count += 1

cap.release()
cv2.destroyAllWindows()
log_file.close()
