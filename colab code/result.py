import json

classification_results_path = "/content/drive/MyDrive/DeepFakeDetection/data/output/classification_results.txt"
result_json_path = "/content/result.json"

segment_labels = {}
with open(classification_results_path, "r") as f:
    for line in f:
        if line.startswith("Segment"):
            parts = line.strip().split(": ")
            segment_labels[int(parts[0].split()[1])] = parts[1]  
deepfake_segments = [seg for seg, label in segment_labels.items() if label == "Deepfake"]

flagged_intervals = []
if deepfake_segments:
    consecutive_authentic = 0
    consecutive_deepfake = 0
    first_deepfake = None  
    tracking_deepfake = False

    for seg in sorted(segment_labels.keys()):
        if seg in deepfake_segments:
            if first_deepfake is None:
                first_deepfake = seg  
            
            consecutive_deepfake += 1
            consecutive_authentic = 0
            
            if consecutive_deepfake > 3:  
                consecutive_authentic = 0  
                if not tracking_deepfake:
                    flagged_intervals.append((first_deepfake, seg))
                    tracking_deepfake = True
                else:
                    flagged_intervals[-1] = (flagged_intervals[-1][0], seg)  
        else:
            consecutive_authentic += 1
            consecutive_deepfake = 0
            first_deepfake = None  
            
            if consecutive_authentic >= 3:  
                tracking_deepfake = False
                consecutive_deepfake = 0  

print("Flagged Intervals:", flagged_intervals)

# Gap < 2 sec
merged_intervals = []
for interval in flagged_intervals:
    if merged_intervals and (interval[0] - merged_intervals[-1][1]) < 2:
        merged_intervals[-1] = (merged_intervals[-1][0], interval[1])  
    else:
        merged_intervals.append(interval)

first_segment = min(segment_labels.keys())  
last_segment = max(segment_labels.keys())

if len(merged_intervals) == 1 and merged_intervals[0] == (first_segment, last_segment):
    print("Video is DEEPFAKE")
    final_decision = "DEEPFAKE"
else:
    if not merged_intervals:
        print("Video is AUTHENTIC")
        final_decision = "AUTHENTIC"
    else:
        print("Video is DEEPFAKE")
        final_decision = "DEEPFAKE"

with open(result_json_path, "w") as f:
    json.dump({"final_decision": final_decision}, f)


