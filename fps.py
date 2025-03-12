import cv2

video_path = "./data/videos/download(1).mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Get the FPS
fps = cap.get(cv2.CAP_PROP_FPS)

# Print the FPS
print(f"Frames per second: {fps}")

# Release the video capture object
cap.release()


