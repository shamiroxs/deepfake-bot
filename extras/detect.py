import torch
import timm
import cv2
import numpy as np
from facenet_pytorch import MTCNN
from torchvision import transforms

# Initialize MTCNN for face detection
mtcnn = MTCNN(image_size=160, margin=14)

# Load EfficientNetV2 from timm
model = timm.create_model("efficientnetv2_b0", pretrained=False, num_classes=2)

# Load converted model weights
checkpoint_path = "./model/efficientnetv2.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu")))

# Set model to evaluation mode
model.eval()

# Define transformations (EfficientNetV2 expects 224x224 images)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_frames(video_path):
    """Extract frames from video using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def detect_faces_and_classify(frames):
    """Detect faces using MTCNN and classify using EfficientNetV2."""
    deepfake_scores = []
    
    for frame in frames:
        try:
            face = mtcnn(frame)
            if face is not None:
                face = transform(face.permute(1, 2, 0).numpy()).unsqueeze(0)  # Preprocess face
                
                with torch.no_grad():
                    output = model(face)
                    prediction = torch.nn.functional.softmax(output, dim=1)
                    deepfake_scores.append(prediction[0][1].item())  # Probability of deepfake

        except Exception as e:
            print(f"[ERROR] Failed to process frame: {e}")

    return deepfake_scores

def aggregate_results(scores, threshold=0.5):
    """Aggregate results from frames."""
    if not scores:
        return "No faces detected"
    
    avg_score = np.mean(scores)
    return "Deepfake Detected" if avg_score > threshold else "Video is Authentic"

def main(video_path):
    """Process the input video and detect deepfakes."""
    frames = extract_frames(video_path)
    deepfake_scores = detect_faces_and_classify(frames)
    result = aggregate_results(deepfake_scores)
    print(f"[FINAL RESULT] {result}")

if __name__ == "__main__":
    video_path = "input_video.mp4"
    main(video_path)

