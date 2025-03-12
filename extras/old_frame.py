import os
import cv2
import torch
from facenet_pytorch import MTCNN

video_path = "./data/videos/input_video.mp4"
output_faces_dir = "./data/face"
os.makedirs(output_faces_dir, exist_ok=True) 

# Initialize MTCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)  

cap = cv2.VideoCapture(video_path)
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    boxes, _ = mtcnn.detect(frame_rgb)  

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)  
            face = frame[y1:y2, x1:x2]  
            
            # Save cropped face
            face_filename = os.path.join(output_faces_dir, f"frame_{frame_count:04d}_face{i}.jpg")
            cv2.imwrite(face_filename, face)

    frame_count += 1

cap.release()
print(f"Faces extracted and saved to {output_faces_dir}")

