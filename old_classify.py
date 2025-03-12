import os
import torch
import cv2
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
from PIL import Image
from torch import nn

# Paths
input_frames_dir = "./data/face"
output_results_file = "./classification_results.txt"
model_path = "./model/efficientnet_v2_s.pth"

# Model configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNetV2 model with a custom classifier for binary classification
class EfficientNetV2Binary(nn.Module):
    def __init__(self, original_model, num_classes=2):
        super(EfficientNetV2Binary, self).__init__()
        # Extract the feature part (stem + blocks)
        self.features = nn.Sequential(
            original_model.stem,         # First, the stem layer
            original_model.blocks,       # Then the blocks (MBConv layers)
            original_model.head.bottleneck,  # The bottleneck layer just before the head
            original_model.head.avgpool,    # The average pooling
            original_model.head.flatten    # Flattening
        )
        # Replace the classifier with a new one for binary classification
        self.classifier = nn.Linear(original_model.head.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)  # Get the feature representation
        x = torch.flatten(x, 1)  # Flatten the features
        x = self.classifier(x)  # Apply the custom classifier
        return x


# Load the pre-trained EfficientNetV2 model and update the classifier
print("Loading model...")
original_model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=False)
original_model.load_state_dict(torch.load(model_path))
model = EfficientNetV2Binary(original_model, num_classes=2)  # binary classification (Deepfake vs Authentic)
model.to(device)
model.eval()  # Set the model to evaluation mode

# Define preprocessing transforms for EfficientNetV2
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for EfficientNetV2 input size
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Perform classification on the frames
def classify_frames():
    deepfake_count = 0
    authentic_count = 0

    with open(output_results_file, "w") as results_file:
        for frame_name in sorted(os.listdir(input_frames_dir)):
            frame_path = os.path.join(input_frames_dir, frame_name)
            frame = cv2.imread(frame_path)

            if frame is None:
                continue  # Skip invalid frames

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            image = Image.fromarray(frame_rgb)  # Convert to PIL image
            processed_image = preprocess_transform(image).unsqueeze(0).to(device)  # Apply preprocessing and add batch dimension

            # Inference (forward pass)
            with torch.no_grad():
                outputs = model(processed_image)
                _, predicted = torch.max(outputs, 1)  # Get the class with highest probability

            # Write the result to the file and count predictions
            if predicted.item() == 0:  # 0 is Deepfake
                deepfake_count += 1
                results_file.write(f"{frame_name}: Deepfake\n")
            else:  # 1 is Authentic
                authentic_count += 1
                results_file.write(f"{frame_name}: Authentic\n")

    print(f"Deepfake Count: {deepfake_count}")
    print(f"Authentic Count: {authentic_count}")
    print(f"Classification results saved to {output_results_file}")
    
    # Majority voting to decide the final classification
    if deepfake_count > authentic_count:
        result = "Deepfake"
    else:
        result = "Authentic"
    print(result)
    results_file.write(f"Final classification: {result}\n")

if __name__ == "__main__":
    classify_frames()

