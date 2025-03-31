import os
import requests
from flask import Flask, request, jsonify
from google.cloud import storage
from google.cloud import pubsub_v1

app = Flask(__name__)

VERIFY_TOKEN = "gopu@yadhu@shamir@123"
PAGE_ACCESS_TOKEN = "EAAQiflGBAxIBOZC6zAjiOFEnW0PlMre5SioNEQXDNRydNBeyWv17ilMrQOCdwI6xg8CFgqct2utYKqDV8AiZA3wJQP5L55KtdTRZCMZASwh7Md0exLWmxVlJ6Rn6MBXyhyv1kZCkcZBTFNiZBKgh76DkWMXxawqUNG3ih33nBmNa7dH00XcVOYjbZAQITPXPueipF9KJCXZCJKgZDZD"  # Get from Facebook Developer Console

BUCKET_NAME = "myapp-code-storage"
VIDEO_PATH = "videos/download.mp4"

processed_messages = set()

sender_map = {}

@app.route("/receive_result", methods=["POST"])
def receive_result():
    try:
        # Get the JSON data sent from Colab
        result = request.get_json()
        print(f"Received result: {result}")

        if "final_decision" not in result:
            return jsonify({"status": "error", "message": "Missing 'final_decision' in result"}), 400

        final_decision = result["final_decision"]

        # Get the sender ID (assuming it's stored based on the last video message_id)
        if not sender_map:
            return jsonify({"status": "error", "message": "No sender ID found"}), 400
        
        sender_id = list(sender_map.values())[-1]  # Get the last sender ID
        
        # Send the detection result to the user
        send_message(sender_id, f"Video is {final_decision}")

        return jsonify({"status": "success", "message": f"Result received and sent to user: {final_decision}"}), 200

    except Exception as e:
        print(f"Error processing result: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "Facebook webhook is running"

@app.route("/webhook", methods=["GET"])
def verify():
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if token == VERIFY_TOKEN:
        return challenge
    return "Invalid verification token", 403

@app.route("/webhook", methods=["POST"])
def webhook():
    data = request.get_json()
    if data.get("object") == "page":
        for entry in data.get("entry", []):
            for message_event in entry.get("messaging", []):
                sender_id = message_event["sender"]["id"]
                message_id = message_event["message"]["mid"]
                
                if message_id in processed_messages:
                    continue  # Skip duplicate processing
                
                processed_messages.add(message_id)
                
                if "attachments" in message_event["message"]:
                    for attachment in message_event["message"]["attachments"]:
                        if attachment["type"] == "video":
                            video_url = attachment["payload"]["url"]
                            
                            sender_map[message_id] = sender_id
                            
                            upload_video_to_gcs(video_url)
                            send_message(sender_id, "Video received and uploaded successfully!")
    return "EVENT_RECEIVED", 200

def upload_video_to_gcs(video_url):
    response = requests.get(video_url, stream=True)
    if response.status_code == 200:
        client = storage.Client.from_service_account_json("/app/service-account.json")
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(VIDEO_PATH)
        blob.upload_from_string(response.content, content_type="video/mp4")
        print("Video uploaded to Google Cloud Storage")

        # Publish message to Pub/Sub
        publish_to_pubsub("Video uploaded")

def publish_to_pubsub(message):
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path("deepfakedetectionbot", "deepfake-detection-topic")
    publisher.publish(topic_path, message.encode("utf-8"))

def send_message(recipient_id, message_text):
    url = "https://graph.facebook.com/v12.0/me/messages"
    params = {"access_token": PAGE_ACCESS_TOKEN}
    headers = {"Content-Type": "application/json"}
    payload = {"recipient": {"id": recipient_id}, "message": {"text": message_text}}
    requests.post(url, params=params, headers=headers, json=payload)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

