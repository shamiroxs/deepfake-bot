import subprocess
import requests
import json
from google.cloud import pubsub_v1
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/content/drive/MyDrive/DeepFakeDetection/service-account.json"  # Replace with the actual path


# Google Cloud project and subscription details
project_id = "DeepfakeDetectionBot"
subscription_id = "deepfake-detection-sub"

# Your Facebook webhook URL (replace with your actual URL)
WEBHOOK_URL = "https://facebook-webhook-513982996952.us-central1.run.app/receive_result"

# Callback function when a Pub/Sub message is received
def callback(message):
    print(f"Received message: {message.data}")
    message.ack()  # Acknowledge the message to Pub/Sub

    # Run main.py (assuming it's already present in Colab)
    print("Executing main.py...")
    subprocess.run(["python3", "/content/drive/MyDrive/DeepFakeDetection/main.py"], check=True)

    # After detection, load the result from the saved file
    with open("/content/result.json", "r") as result_file:
        result = json.load(result_file)

    # Send the result back to the Facebook webhook
    send_result_to_webhook(result)
    
def send_result_to_webhook(result):
    try:
        # Extract only the final decision
        result_text = result.get("final_decision", "UNKNOWN")

        # Send only the result text as a JSON payload
        response = requests.post(WEBHOOK_URL, json={"final_decision": result_text})

        if response.status_code == 200:
            print("Successfully sent the result to the Facebook webhook.")
        else:
            print(f"Failed to send result: {response.status_code}, {response.text}")

    except Exception as e:
        print(f"Error sending result to webhook: {e}")

# Set up Pub/Sub subscriber
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path(project_id, subscription_id)

# Start listening for Pub/Sub messages
print("Listening for messages on Pub/Sub...")
streaming_pull_future = subscriber.subscribe(subscription_path, callback=callback)

# Keep the script running to listen for messages
streaming_pull_future.result()
