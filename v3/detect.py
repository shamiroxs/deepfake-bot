import os
import cv2
import torch
import time
from multiprocessing import Process, Queue
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image

# Paths
video_path = "./data/videos/download_1.mp4"
output_results_file = "./data/output/classification_results.txt"
model_path = "./model/efficientnet_v2_s.pth"

if os.path.exists(output_results_file):
    os.remove(output_results_file)

# Initialize MTCNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=True, device=device)

# Load Classification Model
model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=True, nclass=2)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Define preprocessing transforms
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_interval = max(1, fps // 7)  # Target FPS is 7
segment_duration = 1
cap.release()

# Multiprocessing Queues
mtcnn_queue = Queue()
results_queue = Queue()

def classify_segment(segment_frame_labels):
    if not segment_frame_labels:
        return "Authentic"
    return "Deepfake" if sum(segment_frame_labels) / len(segment_frame_labels) > 0.4 else "Authentic"

def mtcnn_processing(video_path, mtcnn_queue):
    print("mtcnn processing")
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    segment_id = 0
    segment_frame_labels = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            timestamp = frame_count / fps
            current_segment = int(timestamp // segment_duration)

            if current_segment != segment_id and segment_frame_labels:
                mtcnn_queue.put((segment_id, segment_frame_labels.copy()))
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
                            break  # Mark entire frame as deepfake if any face is fake

            segment_frame_labels.append(1 if frame_is_deepfake else 0)

        frame_count += 1

    cap.release()
    if segment_frame_labels:
        mtcnn_queue.put((segment_id, segment_frame_labels))
    mtcnn_queue.put(None)  # Termination signal

def classification_processing(mtcnn_queue, results_queue):
    while True:
        data = mtcnn_queue.get()
        if data is None:
            break  # Termination signal

        segment_id, segment_faces = data
        result = classify_segment(segment_faces)
        results_queue.put((segment_id, result))

if __name__ == "__main__":
    mtcnn_process = Process(target=mtcnn_processing, args=(video_path, mtcnn_queue))
    classification_process = Process(target=classification_processing, args=(mtcnn_queue, results_queue))

    mtcnn_process.start()
    classification_process.start()

    mtcnn_process.join()
    classification_process.join()

    while not results_queue.empty():
        segment_id, result = results_queue.get()
        print(f"Segment {segment_id}: {result}")
        with open(output_results_file, "a") as f:
            f.write(f"Segment {segment_id}: {result}\n")

    print("Processing completed.")
