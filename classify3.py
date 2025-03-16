import os
import json
import torch
import cv2
import time
import numpy as np
from PIL import Image
from torch import nn
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms

# Set multiprocessing mode for Windows
import multiprocessing

# Paths
input_frames_dir = "./data/face"
metadata_path = "./data/frame_metadata.json"
output_results_json = "./data/classification_results.json"
output_results_txt = "./data/classification_results.txt"
model_path = "./model/efficientnet_v2_s.pth"
video_frames_dir = "./data/videoframes"
output_dir = "./data/output"
DEEPFAKE_THRESHOLD = 0.6623

# CUDA optimization
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EfficientNetV2Binary(nn.Module):
    def __init__(self, original_model, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            original_model.stem,
            original_model.blocks,
            original_model.head.bottleneck,
            original_model.head.avgpool,
            original_model.head.flatten,
        )
        self.classifier = nn.Linear(original_model.head.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load model
def load_model():
    print("[Face Model] Loading model...")
    start_time = time.time()
    original_model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=False)
    original_model.load_state_dict(torch.load(model_path, map_location=device))
    model = EfficientNetV2Binary(original_model, num_classes=2)
    model.to(device)
    model.eval()
    print(f"[Face Model] Model loaded in {time.time() - start_time:.2f} seconds")
    return model

def classify_faces():
    model = load_model()
    print("[Face Classification] Started processing...")
    start_time = time.time()
    
    preprocess_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    with open(metadata_path, "r") as f:
        frame_metadata = json.load(f)
    
    segment_results = {}
    
    for frame_info in frame_metadata:
        segment_id = frame_info["segment_id"]
        deepfake_confidence_sum = 0.0
        total_faces = 0
        
        images = []
        
        for face in frame_info["faces"]:
            image = Image.open(face["face_path"]).convert("RGB")
            images.append(preprocess_transform(image))

        if images:
            batch = torch.stack(images).to(device)
            with torch.no_grad():
                outputs = model(batch)
                confidences = torch.nn.functional.softmax(outputs, dim=1)[:, 1]
                
            deepfake_confidence_sum += confidences.sum().item()
            total_faces += len(confidences)
        
        avg_confidence = deepfake_confidence_sum / max(1, total_faces)
        segment_results[segment_id] = segment_results.get(segment_id, 0) + avg_confidence
    
    print(f"[Face Classification] Completed in {time.time() - start_time:.2f} seconds")
    return segment_results

def background_analysis():
    print("[Background Analysis] Started processing...")
    start_time = time.time()
    results = {}
    
    frame_files = sorted(os.listdir(video_frames_dir))
    prev_frame = None
    segment_confidences = {}
    segment_counts = {}
    segment_duration = 2
    deepflow = cv2.optflow.createOptFlow_DeepFlow()
    
    total_frames = len(frame_files)
    sample_rate = max(1, total_frames // max(6, min(10, total_frames // segment_duration)))
    
    for i, frame_file in enumerate(frame_files[::sample_rate]):
        frame_path = os.path.join(video_frames_dir, frame_file)
        curr_frame = cv2.imread(frame_path)
        
        fps = max(6, min(10, total_frames // segment_duration))
        timestamp = i / fps
        segment_id = int(timestamp // segment_duration)
        
        if prev_frame is not None:
            flow = deepflow.calc(
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
                cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
                None
            )
            noise = np.var(cv2.GaussianBlur(curr_frame.astype(np.float32), (5, 5), 0) - curr_frame.astype(np.float32))
            confidence = 1 - (np.mean(flow) * 0.5 + noise * 0.05)
            confidence = max(0, min(1, confidence))
            
            segment_confidences[segment_id] = segment_confidences.get(segment_id, 0) + confidence
            segment_counts[segment_id] = segment_counts.get(segment_id, 0) + 1
        
        prev_frame = curr_frame
    
    for segment_id in segment_confidences:
        results[segment_id] = segment_confidences[segment_id] / segment_counts[segment_id]
    
    print(f"[Background Analysis] Completed in {time.time() - start_time:.2f} seconds")
    return results

def weighted_fusion(face_results, bg_results):
    print("[Fusion] Combining results...")
    final_results = {}
    
    with open(output_results_txt, "w") as txt_file:
        for segment_id in face_results:
            face_conf = face_results.get(segment_id, 0.0)
            bg_conf = bg_results.get(segment_id, 0.0)
            final_conf = 0.7 * face_conf + 0.3 * bg_conf
            label = "Deepfake" if final_conf > DEEPFAKE_THRESHOLD else "Authentic"
            final_results[segment_id] = {"confidence": final_conf, "label": label}
            txt_file.write(f"Segment {segment_id}: {label}\n")
    
    print("[Fusion] Completed.")
    return final_results

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    with ThreadPoolExecutor() as executor:
        face_future = executor.submit(classify_faces)
        bg_future = executor.submit(background_analysis)
        
        face_results = face_future.result()
        bg_results = bg_future.result()
    
    final_results = weighted_fusion(face_results, bg_results)
    
    with open(output_results_json, "w") as f:
        json.dump(final_results, f, indent=4)
    
    print("[Final Output] Deepfake detection complete!")
