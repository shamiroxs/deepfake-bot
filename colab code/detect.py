import os
import cv2
import torch
import json
import time
import shutil
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image

video_path = "/content/input.mp4"
output_results_file = "/content/drive/MyDrive/DeepFakeDetection/data/output/classification_results.txt"
model_path = "/content/drive/MyDrive/DeepFakeDetection/model/efficientnet_v2_s.pth"

if os.path.exists(output_results_file):
    os.remove(output_results_file)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=True, nclass=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

print(f"Original Frames per second: {fps}")

target_fps = 7
frame_interval = int(fps / target_fps) if fps >= target_fps else 1
segment_duration = 1

print(f"Processing video at {target_fps} FPS.")
print(f"Total video duration: {duration:.2f} seconds\n")

print("Face classification results:")
def classify_segment(segment_frame_labels):
    if not segment_frame_labels:
        return "Authentic"
    
    deepfake_frames = sum(segment_frame_labels)
    if deepfake_frames / len(segment_frame_labels) > 0.3: #max 0.4
        return "Deepfake"
    return "Authentic"

frame_count = 0
segment_id = 0
segment_frame_labels = []

start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    if frame_count % frame_interval == 0:
        timestamp = frame_count / fps
        current_segment = int(timestamp // segment_duration)
        
        if current_segment != segment_id and segment_frame_labels:
            result = classify_segment(segment_frame_labels)
            print(f"Segment {segment_id}: {result}")
            with open(output_results_file, "a") as f:
                f.write(f"Segment {segment_id}: {result}\n")
            segment_frame_labels.clear()
            segment_id = current_segment
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(frame_rgb)
        
        frame_is_deepfake = False
        
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    face = frame[y1:y2, x1:x2]
                    image = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                    processed_image = preprocess_transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        output = model(processed_image)
                        prediction = torch.argmax(output, dim=1).item()
                    
                    if prediction == 1:
                        frame_is_deepfake = True
                        break  
                        
        segment_frame_labels.append(1 if frame_is_deepfake else 0)
    
    frame_count += 1

cap.release()

if segment_frame_labels:
    result = classify_segment(segment_frame_labels)
    print(f"Segment {segment_id}: {result}")
    with open(output_results_file, "a") as f:
        f.write(f"Segment {segment_id}: {result}\n")

end_time = time.time()
print(f"Total execution time: {end_time - start_time:.2f} seconds")
