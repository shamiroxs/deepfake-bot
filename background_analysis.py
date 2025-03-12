import cv2
import numpy as np
import os

# Paths
video_path = "./data/videos/download_3.mp4"
output_path = "./background_analysis.json"

# Open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Initialize DeepFlow optical flow
deepflow = cv2.optflow.createOptFlow_DeepFlow()

# Read first frame
ret, prev_frame = cap.read()
if not ret:
    print("Error: Could not read first frame.")
    cap.release()
    exit()

# Convert to grayscale
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

frame_index = 0
suspicious_frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Compute optical flow
    flow = deepflow.calc(prev_gray, gray, None)
    
    # Analyze motion
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_magnitude = np.mean(magnitude)
    
    # Threshold for suspicious background motion
    if avg_magnitude > 5.0:  # Threshold can be adjusted based on experiments
        suspicious_frames.append(frame_index)
    
    # Update previous frame
    prev_gray = gray
    frame_index += 1

cap.release()

# Save results
with open(output_path, "w") as f:
    f.write(str(suspicious_frames))

print(f"Background analysis completed. Results saved to {output_path}")

