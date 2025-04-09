import subprocess
import requests
import json
from google.cloud import pubsub_v1
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/drive/MyDrive/DeepFakeDetection/service-account.json"  

project_id = "DeepfakeDetectionBot"
subscription_id = "deepfake-detection-sub"
WEBHOOK_URL = "https://facebook-webhook-513982996952.us-central1.run.app/receive_result"

def callback(message):
    print(f"Received message: {message.data}")
    message.ack()  
    
    print("Executing main.py...")
    subprocess.run(["python3", "/content/drive/MyDrive/DeepFakeDetection/main.py"], check=True)

    with open("/content/result.json", "r") as result_file:
        result = json.load(result_file)

    send_result_to_webhook(result)
    
def send_result_to_webhook(result):
    try:
        result_text = result.get("final_decision", "UNKNOWN")

        response = requests.post(WEBHOOK_URL, json={"final_decision": result_text})

        if response.status_code == 200:
            print("Successfully sent the result to the Facebook webhook.")
        else:
            print(f"Failed to send result: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"Error sending result to webhook: {e}")

subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

# Start listening
print("Listening for messages on Pub/Sub...")
streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)
streaming_pull_future.result()
