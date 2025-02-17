import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

# Paths
input_frames_dir = "./data/face"
output_processed_dir = "./data/processed_face"
os.makedirs(output_processed_dir, exist_ok=True)  # Create output folder if not exists

# EfficientNetV2 preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to EfficientNetV2 input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Process each frame
for frame_name in sorted(os.listdir(input_frames_dir)):
    frame_path = os.path.join(input_frames_dir, frame_name)
    frame = cv2.imread(frame_path)

    if frame is None:
        continue  # Skip invalid images

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = Image.fromarray(frame_rgb)  # Convert to PIL image
    processed_image = transform(image)  # Apply preprocessing

    # Convert back to numpy and save (for visualization and further processing)
    processed_np = processed_image.permute(1, 2, 0).numpy() * 255  # De-normalize for saving
    processed_np = processed_np.astype("uint8")

    processed_filename = os.path.join(output_processed_dir, frame_name)
    cv2.imwrite(processed_filename, processed_np)

print(f"Processed frames saved to {output_processed_dir}")

