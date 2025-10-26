import face_recognition
import os
import pickle

# Define the path to the known faces folder
KNOWN_FACES_DIR = r'E:\snatching\face_detection\known_faces'

# Load known faces
known_encodings = []
known_names = []

for person_name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)

        if len(encoding) > 0:
            known_encodings.append(encoding[0])
            known_names.append(person_name)

# Save encodings for future use
with open('known_faces_encodings.pkl', 'wb') as f:
    pickle.dump((known_encodings, known_names), f)

