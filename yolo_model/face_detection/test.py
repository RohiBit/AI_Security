import face_recognition
import pickle
import cv2

# Load known encodings
with open(r'F:\projects\epics\final\face_detection\known_faces_encodings.pkl', 'rb') as f:
    known_encodings, known_names = pickle.load(f)

def recognize_faces(image_path):
    image = face_recognition.load_image_file(image_path)

    # Resize image to larger size for better visibility
    scale_factor = 2.0  # Increase size by 2x
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    image = cv2.resize(image, (width, height))

    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

        print(f"Detected: {name}")

        if name == "Unknown":
            print("Detected: Unknown")

        # Increase font scale and thickness for better visibility
        font_scale = 1.0
        thickness = 2

        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), thickness)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), thickness)

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

import os

def recognize_faces_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            print(f"Processing {image_path}")
            recognize_faces(image_path)

# Example usage:
if __name__ == "__main__":
    folder_path = r'F:\projects\epics\final\snatching_detections'
    recognize_faces_in_folder(folder_path)

