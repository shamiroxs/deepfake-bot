import os
import sys

VIDEO_DIR = "./data/videos"
SELECTION_FILE = "./selection.txt"

def list_videos():
    """Lists all video files in the directory."""
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Directory '{VIDEO_DIR}' not found.")
        return []

    videos = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
    return videos

def main():
    videos = list_videos()
    
    if not videos:
        print("No video files found. Exiting...")
        sys.exit(1)  # Exit if no videos are available

    print("\nAvailable video files:")
    for idx, video in enumerate(videos, start=1):
        print(f"{idx}. {video}")

    print("0. Exit")  # Option to exit and delete

    while True:
        try:
            choice = int(input("\nSelect a video by entering its number (or 0 to exit): "))
            
            if choice == 0:
                # Check if there's a previously selected video to delete
                if os.path.exists(SELECTION_FILE):
                    with open(SELECTION_FILE, "r") as f:
                        selected_video = f.read().strip()
                    
                    if os.path.exists(selected_video):
                        print(f"Deleting {selected_video}...")
                        os.remove(selected_video)
                        print("File deleted successfully.")
                
                print("Exiting program...")
                sys.exit(0)  # Stop execution

            if 1 <= choice <= len(videos):
                selected_video = os.path.join(VIDEO_DIR, videos[choice - 1])
                with open(SELECTION_FILE, "w") as f:
                    f.write(selected_video)
                
                print(f"Selected: {videos[choice - 1]}")
                return  # Continue execution

            else:
                print("Invalid choice, please try again.")

        except ValueError:
            print("Please enter a valid number.")

if __name__ == "__main__":
    main()
