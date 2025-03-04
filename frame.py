import os
import cv2
import torch
import json
from facenet_pytorch import MTCNN

# Paths
video_path = "./data/videos/download_3.mp4"
output_faces_dir = "./data/face"
metadata_file = "./data/face_metadata.json"
os.makedirs(output_faces_dir, exist_ok=True)

# Initialize MTCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Open video file
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))  # Frames per second
print(f"Original Frames per second: {fps}")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps  # Total duration in seconds
"""
# If FPS is greater than 24, convert it to 24 FPS
target_fps = 24
if fps > target_fps:
    temp_video_path = "./data/videos/temp_video.mp4"
    os.system(f"ffmpeg -i {video_path} -filter:v fps={target_fps} {temp_video_path} -y")
    video_path = temp_video_path  # Use the converted video
    cap = cv2.VideoCapture(video_path)
    fps = target_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print("Video converted to 24 FPS")
"""
# Metadata storage
metadata = {}
frame_count = 0
segment_duration = 0.5  # 0.5-second segments

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    timestamp = frame_count / fps  # Current time in seconds
    segment_id = int(timestamp // segment_duration)  # Assign frame to a segment

    # Convert frame to RGB for MTCNN
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(frame_rgb)

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)

            # Ensure coordinates are within image bounds
            h, w, _ = frame.shape
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)

            # Check if cropped face is valid
            if x2 > x1 and y2 > y1:
                face = frame[y1:y2, x1:x2]

                # Save cropped face
                face_filename = f"frame_{frame_count:04d}_face{i}.jpg"
                face_path = os.path.join(output_faces_dir, face_filename)
                cv2.imwrite(face_path, face)

                # Store metadata
                if segment_id not in metadata:
                    metadata[segment_id] = []
                metadata[segment_id].append({
                    "frame": frame_count,
                    "timestamp": round(timestamp, 2),
                    "face_path": face_path
                })

    frame_count += 1

cap.release()

# Save metadata
with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=4)

print(f"Faces extracted and saved to {output_faces_dir}")
print(f"Metadata saved to {metadata_file}")

