import json

# Paths
classification_results_path = "./classification_results.txt"
metadata_path = "./data/face_metadata.json"

# Load segment-wise classification results
segment_labels = {}
with open(classification_results_path, "r") as f:
    for line in f:
        if line.startswith("Segment"):
            parts = line.strip().split(": ")
            segment_labels[int(parts[0].split()[1])] = parts[1]  # Extract segment number and label

# Load metadata to get timestamps
with open(metadata_path, "r") as f:
    face_metadata = json.load(f)

# Identify deepfake segments
deepfake_segments = [seg for seg, label in segment_labels.items() if label == "Deepfake"]

# Identify continuous deepfake regions
flagged_intervals = []
if deepfake_segments:
    consecutive_authentic = 0
    consecutive_deepfake = 0
    first_deepfake = None  # Track the first deepfake in a valid sequence
    tracking_deepfake = False

    for seg in range(min(segment_labels.keys()), max(segment_labels.keys()) + 1):
        if seg in deepfake_segments:
            if first_deepfake is None:
                first_deepfake = seg  # Mark first deepfake segment
            
            consecutive_deepfake += 1
            consecutive_authentic = 0  # Reset authentic counter
            
            if consecutive_deepfake >= 6:  # Start tracking from the first deepfake in the sequence
                if not tracking_deepfake:
                    flagged_intervals.append((first_deepfake, seg))
                    tracking_deepfake = True
                else:
                    flagged_intervals[-1] = (flagged_intervals[-1][0], seg)  # Extend interval
        else:
            consecutive_authentic += 1
            consecutive_deepfake = 0  # Reset deepfake counter
            first_deepfake = None  # Reset first deepfake marker
            
            if consecutive_authentic >= 6:  # Stop tracking if too many authentic frames appear
                tracking_deepfake = False

print("Flagged Intervals:", flagged_intervals)

# Merge close intervals (gap < 3 sec)
merged_intervals = []
for interval in flagged_intervals:
    if merged_intervals and (interval[0] * 0.5 - merged_intervals[-1][1] * 0.5) < 3:
        merged_intervals[-1] = (merged_intervals[-1][0], interval[1])  # Merge intervals
    else:
        merged_intervals.append(interval)

# Final decision
if len(merged_intervals) == 1 and merged_intervals[0][0] == 0 and merged_intervals[0][1] == max(segment_labels.keys()):
    print("Video is deepfake")
else:
    print("Video is flagged as MANIPULATED!")
    print("Deepfake timestamps (seconds):")
    for start_seg, end_seg in merged_intervals:
        start_time = start_seg * 0.5  # Each segment is 0.5 seconds long
        end_time = end_seg * 0.5
        print(f"- {start_time:.1f}s to {end_time:.1f}s")

