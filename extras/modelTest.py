import torch
import torchvision.transforms as transforms
from PIL import Image

# Paths
model_path = "./model/efficientnet_v2_s.pth"
test_image_path = "sample_image.jpg"  # Change this to a known deepfake/authentic image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load EfficientNetV2 model with a custom classifier for binary classification
class EfficientNetV2Binary(torch.nn.Module):
    def __init__(self, original_model, num_classes=2):
        super(EfficientNetV2Binary, self).__init__()
        self.features = torch.nn.Sequential(
            original_model.stem,
            original_model.blocks,
            original_model.head.bottleneck,
            original_model.head.avgpool,
            original_model.head.flatten,
        )
        self.classifier = torch.nn.Linear(original_model.head.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load the model
print("Loading model...")
original_model = torch.hub.load('hankyul2/EfficientNetV2-pytorch', 'efficientnet_v2_s', pretrained=False)
original_model.load_state_dict(torch.load(model_path, map_location=device))
model = EfficientNetV2Binary(original_model, num_classes=2)
model.to(device)
model.eval()

# Define preprocessing transforms
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load test image
image = Image.open(test_image_path).convert("RGB")
processed_image = preprocess_transform(image).unsqueeze(0).to(device)

# Inference
with torch.no_grad():
    outputs = model(processed_image)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)  # Convert logits to probabilities
    predicted_class = torch.argmax(probabilities, dim=1).item()

# Print results
print(f"Raw Model Output: {outputs.cpu().numpy()}")
print(f"Softmax Probabilities: {probabilities.cpu().numpy()}")
print(f"Predicted Class: {predicted_class}")

# Interpret results
if predicted_class == 0:
    print("Model classifies this image as: Deepfake")
else:
    print("Model classifies this image as: Authentic")
