import cv2
from mtcnn import MTCNN
import os

def detect_and_save_faces(image_folder, save_folder):
    detector = MTCNN()
    
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Iterate over all subfolders (e.g., criminals' names) and images
    for person_name in os.listdir(image_folder):
        person_folder = os.path.join(image_folder, person_name)
        
        if os.path.isdir(person_folder):  # Ensure it's a folder
            for image_name in os.listdir(person_folder):
                image_path = os.path.join(person_folder, image_name)
                
                # Read the image
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Failed to load image: {image_path}")
                    continue
                
                # Convert the image to RGB
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                faces = detector.detect_faces(rgb_img)
                
                # Save each detected face
                for i, face in enumerate(faces):
                    x, y, width, height = face['box']
                    face_img = img[y:y+height, x:x+width]
                    
                    # Ensure the save folder for the person exists
                    person_save_folder = os.path.join(save_folder, person_name)
                    if not os.path.exists(person_save_folder):
                        os.makedirs(person_save_folder)
                    
                    face_file = os.path.join(person_save_folder, f"{image_name}_face_{i}.jpg")
                    cv2.imwrite(face_file, face_img)
                    print(f"Face saved at: {face_file}")

# Example usage:
detect_and_save_faces(r'E:\snatching\face_detection\known_faces', r'E:\snatching\face_detection\detected_faces')