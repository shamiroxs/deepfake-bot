import os
import json
import torch
import time
import torchvision.transforms as transforms
from PIL import Image
from torch import nn

# Paths
input_frames_dir = "./data/face"
metadata_path = "./data/output/face_metadata.json"
output_results_file = "./data/output/classification_results.txt"
model_path = "./model/efficientnet_v2_s.pth"
DEEPFAKE_CONFIDENCE = 0.4

start_time = time.time()

# Model configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
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

# Load face metadata
with open(metadata_path, "r") as f:
    face_metadata = json.load(f)

# Perform classification
segment_results = {}

with open(output_results_file, "w") as results_file:
    for segment, frames in face_metadata.items():
        deepfake_frames = set()
        total_frames = set()
        
        # Process frames in batches
        batch_images = []
        batch_frame_ids = []
        
        for frame_info in frames:
            frame_path = frame_info["face_path"]
            frame_id = frame_info["frame"]
            
            try:
                image = Image.open(frame_path).convert("RGB")
                processed_image = preprocess_transform(image)
                batch_images.append(processed_image)
                batch_frame_ids.append(frame_id)
            except Exception as e:
                print(f"Skipping {frame_path}: {e}")
                continue
        
        if batch_images:
            batch_tensor = torch.stack(batch_images).to(device)
            
            with torch.no_grad():
                outputs = model(batch_tensor)
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for frame_id, pred in zip(batch_frame_ids, predictions):
                total_frames.add(frame_id)
                if pred == 1:  # Deepfake
                    deepfake_frames.add(frame_id)
        
        # Apply deepfake threshold per segment
        if len(total_frames) > 0 and (len(deepfake_frames) / len(total_frames)) > DEEPFAKE_CONFIDENCE:
            segment_label = "Deepfake"
        else:
            segment_label = "Authentic"
        
        segment_results[segment] = segment_label
        results_file.write(f"Segment {segment}: {segment_label}\n")

end_time = time.time()

# Print results
print("Face classification results:")
for segment, label in segment_results.items():
    print(f"Segment {segment}: {label}")

print(f"Total execution time: {end_time - start_time:.2f} seconds")
