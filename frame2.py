import os
import cv2
import torch
import json
import time
from facenet_pytorch import MTCNN
import shutil

# Paths
video_path = "./data/videos/download_0.mp4"
output_faces_dir = "./data/face"
output_frames_dir = "./data/videoframes"
metadata_file = "./data/frame_metadata.json"

# Create directories
os.makedirs(output_faces_dir, exist_ok=True)
os.makedirs(output_frames_dir, exist_ok=True)

# Function to clear directories before processing
def clear_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)  # Deletes the entire directory
    os.makedirs(directory)  # Recreates an empty directory

clear_directory(output_faces_dir)
clear_directory(output_frames_dir)

# Initialize MTCNN on GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Open video file
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps  # Total duration in seconds

# Target FPS
target_fps = 7
frame_interval = int(fps / target_fps) if fps >= target_fps else 1

print(f"Original FPS: {fps}, Processing FPS: {target_fps}")
print(f"Total video duration: {duration:.2f} seconds")

# Metadata storage
metadata = []
frame_count = 0
saved_frame_count = 0
segment_duration = 2  # 2-second segments

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        timestamp = round(frame_count / fps, 2)  # Current time in seconds
        segment_id = int(timestamp // segment_duration)

        # Save extracted frame
        frame_filename = f"frame_{saved_frame_count:04d}.jpg"
        frame_path = os.path.join(output_frames_dir, frame_filename)
        cv2.imwrite(frame_path, frame)

        # Convert frame to RGB for MTCNN
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.tensor(frame_rgb).to(device)  # Move frame to GPU

        with torch.no_grad():  # Disable gradient tracking for inference
            boxes, _ = mtcnn.detect(frame_tensor)

        frame_metadata = {
            "frame_id": saved_frame_count,
            "timestamp": timestamp,
            "segment_id": segment_id,
            "frame_path": frame_path,
            "faces": []
        }

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)

                # Ensure coordinates are within image bounds
                h, w, _ = frame.shape
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

                # Check if cropped face is valid
                if x2 > x1 and y2 > y1:
                    face = frame[y1:y2, x1:x2]

                    # Save cropped face
                    face_filename = f"frame_{saved_frame_count:04d}_face{i}.jpg"
                    face_path = os.path.join(output_faces_dir, face_filename)
                    cv2.imwrite(face_path, face)

                    # Store face metadata
                    frame_metadata["faces"].append({
                        "face_id": i,
                        "timestamp": timestamp,  # Aligning face detection timestamp with frame timestamp
                        "face_path": face_path
                    })

        metadata.append(frame_metadata)
        saved_frame_count += 1

    frame_count += 1

cap.release()

# Save metadata
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4)

end_time = time.time()
print(f"Frames extracted and saved to {output_frames_dir}")
print(f"Faces extracted and saved to {output_faces_dir}")
print(f"Metadata saved to {metadata_file}")
print(f"Total execution time: {end_time - start_time:.2f} seconds")
