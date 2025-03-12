import os
import json
import torch
import cv2
import time
import torchvision.transforms as transforms
from PIL import Image
from torch import nn

# Paths
input_frames_dir = "./data/face"
metadata_path = "./data/face_metadata.json"
output_results_file = "./classification_results.txt"
model_path = "./model/efficientnet_v2_s.pth"
DEEPFAKE_CONFIDENCE = 0.75

start_time = time.time()

# Model configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNetV2 model with a custom classifier for binary classification
class EfficientNetV2Binary(nn.Module):
    def __init__(self, original_model, num_classes=2):
        super(EfficientNetV2Binary, self).__init__()
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

# Load the pre-trained model
print("Loading model...")
original_model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=False)
original_model.load_state_dict(torch.load(model_path))
model = EfficientNetV2Binary(original_model, num_classes=2)
model.to(device)
model.eval()

# Define preprocessing transforms for EfficientNetV2
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
        
        for frame_info in frames:
            frame_path = frame_info["face_path"]
            frame_id = frame_info["frame"]  # Unique frame identifier
            
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            image = Image.open(frame_path).convert("RGB")
            processed_image = preprocess_transform(image).unsqueeze(0).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(processed_image)
                _, predicted = torch.max(outputs, 1)
            
            total_frames.add(frame_id)  # Track frames in the segment
            if predicted.item() == 0:
                deepfake_frames.add(frame_id)  # If any face is deepfake, mark the frame
        
        # Apply 20% deepfake threshold per segment
        if len(total_frames) > 0 and (len(deepfake_frames) / len(total_frames)) > DEEPFAKE_CONFIDENCE:
            segment_label = "Deepfake"
        else:
            segment_label = "Authentic"
        
        segment_results[segment] = segment_label
        results_file.write(f"Segment {segment}: {segment_label}\n")
        
end_time = time.time()

# Print final segment classifications
print("Segment-wise classification results:")
for segment, label in segment_results.items():
    print(f"Segment {segment}: {label}")


print(f"Total execution time: {end_time - start_time:.2f} seconds")
