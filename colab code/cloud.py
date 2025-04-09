import os
from google.cloud import storage

project_id = "deepfakedetectionbot"
bucket_name = "myapp-code-storage"
source_blob_name = "videos/download.mp4"  
destination_file_name = "/content/input.mp4" 

client = storage.Client()
bucket = client.bucket(bucket_name)
blob = bucket.blob(source_blob_name)

blob.download_to_filename(destination_file_name)

print(f"✅ Downloaded {source_blob_name} from bucket {bucket_name} to {destination_file_name}.")
