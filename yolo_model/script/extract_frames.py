import cv2
import os

def extract_frames(video_path, output_dir, frame_skip=1):
    """
    Extract frames from a video file and save them as images.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Directory to save extracted frames.
        frame_skip (int): Save every nth frame. Default is 1 (save all frames).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_filename = os.path.join(output_dir, f"frame26_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} to {output_dir}")

if __name__ == "__main__":
    video_file = r"F:\projects\epics\final\input\14.mp4"
    output_dir = r"C:\Users\rohit\OneDrive\Pictures\Screenshots"  # define the output directory
    frame_skip_value = 10  # Save every 10th frame

    extract_frames(video_file, output_dir, frame_skip=frame_skip_value)
