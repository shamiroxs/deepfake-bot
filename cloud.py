import os
import cv2
import boto3
import requests
from urllib.parse import urlparse
from botocore.exceptions import NoCredentialsError

# AWS S3 setup (For AWS Cloud Storage)
AWS_ACCESS_KEY = ''
AWS_SECRET_KEY = ''
AWS_BUCKET_NAME = ''
AWS_REGION = 'eu-north-1'

# Initialize S3 client
s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY,
                         aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

def list_files_in_s3(bucket_name):
    """
    List files in the S3 bucket.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            file_keys = [item['Key'] for item in response['Contents']]
            print(f"Files found in S3: {file_keys}")
            return file_keys
        else:
            print("No files found in the S3 bucket.")
            return []
    except Exception as e:
        print(f"Error listing files in S3: {e}")
        return []

def download_from_s3(bucket_name, object_key, local_path):
    """
    Download video from AWS S3 to local storage.
    """
    try:
        s3_client.download_file(bucket_name, object_key, local_path)
        print(f"Downloaded: {object_key} -> {local_path}")
        return local_path
    except NoCredentialsError:
        print("Credentials not available.")
        return None
    except Exception as e:
        print(f"Failed to download {object_key}: {e}")
        return None

def validate_video(file_path):
    """
    Validate if the downloaded file is a valid video.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        return False, "Invalid video file"
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 1:
        return False, "Video has no frames"
    
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    if file_size > 50:  
        return False, "Video exceeds size limit"
    
    cap.release()
    return True, "Valid video"

def main():
    # Get all file keys from S3
    file_keys = list_files_in_s3(AWS_BUCKET_NAME)
    
    if not file_keys:
        print("No video files found.")
        return
    
    save_directory = os.path.join(os.getcwd(), "data", "videos")
    os.makedirs(save_directory, exist_ok=True)
    
    for index, object_key in enumerate(file_keys):
        local_video_path = os.path.join(os.getcwd(), f"download_{index}.mp4")
        
        # Download video
        downloaded_file = download_from_s3(AWS_BUCKET_NAME, object_key, local_video_path)
        
        if downloaded_file:
            # Validate video
            is_valid, message = validate_video(local_video_path)
            if is_valid:
                print(f"✅ Video {object_key} is valid and saved as {local_video_path}")
            else:
                print(f"❌ Invalid video ({object_key}): {message}")
                os.remove(local_video_path)  # Delete invalid file

if __name__ == "__main__":
    main()
