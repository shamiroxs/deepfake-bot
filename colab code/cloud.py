import os
from google.cloud import storage

# Google Cloud Storage details
project_id = "deepfakedetectionbot"
bucket_name = "myapp-code-storage"
source_blob_name = "videos/download.mp4"  # Path inside the bucket
destination_file_name = "/content/input.mp4"  # Local path in Colab

# Initialize the Google Cloud Storage client
client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(source_blob_name)

# Download the file
blob.download_to_filename(destination_file_name)

print(f"✅ Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")
