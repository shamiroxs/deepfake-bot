import json

# Paths
classification_results_path = "./data/output/classification_results.txt"
metadata_path = "./data/output/face_metadata.json"

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

    for seg in sorted(segment_labels.keys()):
        if seg in deepfake_segments:
            if first_deepfake is None:
                first_deepfake = seg  # Mark first deepfake segment
            
            consecutive_deepfake += 1
            consecutive_authentic = 0
            
            if consecutive_deepfake > 3:  # Start tracking from the first deepfake in the sequence
                consecutive_authentic = 0  # Reset authentic counter
                if not tracking_deepfake:
                    flagged_intervals.append((first_deepfake, seg))
                    tracking_deepfake = True
                else:
                    flagged_intervals[-1] = (flagged_intervals[-1][0], seg)  # Extend interval
        else:
            consecutive_authentic += 1
            consecutive_deepfake = 0
            first_deepfake = None  # Reset first deepfake marker
            
            if consecutive_authentic >= 3:  # Stop tracking if too many authentic frames appear
                tracking_deepfake = False
                consecutive_deepfake = 0  # Reset deepfake counter

print("Flagged Intervals:", flagged_intervals)

# Merge close intervals (gap < 2 sec)
merged_intervals = []
for interval in flagged_intervals:
    if merged_intervals and (interval[0] - merged_intervals[-1][1]) < 2:
        merged_intervals[-1] = (merged_intervals[-1][0], interval[1])  # Merge intervals
    else:
        merged_intervals.append(interval)

first_segment = min(segment_labels.keys())  # Dynamic first segment
last_segment = max(segment_labels.keys())

# Final decision
if len(merged_intervals) == 1 and merged_intervals[0] == (first_segment, last_segment):
    print("Video is deepfake")
else:
    if not merged_intervals:
        print("Video is AUTHENTIC")
    else:
        print("Video is flagged as MANIPULATED!")
        print("Deepfake timestamps (seconds):")
        for start_seg, end_seg in merged_intervals:
            start_time = start_seg  
            end_time = end_seg
            print(f"- {start_time:.1f}s to {end_time:.1f}s")

