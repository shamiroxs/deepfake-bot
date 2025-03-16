import os
import json
import torch
import cv2
import time
import numpy as np
from PIL import Image
from torch import nn
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms

# Set multiprocessing mode for Windows
import multiprocessing

# Paths
input_frames_dir = "./data/face"
metadata_path = "./data/frame_metadata.json"
output_results_json = "./data/classification_results.json"
output_results_txt = "./data/classification_results.txt"
result_summary_json = "./data/final_result.json"
result_summary_txt = "./data/final_result.txt"
model_path = "./model/efficientnet_v2_s.pth"
video_frames_dir = "./data/videoframes"
output_dir = "./data/output"
DEEPFAKE_THRESHOLD = 0.6623

# CUDA optimization
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EfficientNetV2Binary(nn.Module):
    def __init__(self, original_model, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            original_model.stem,
            original_model.blocks,
            original_model.head.bottleneck,
            original_model.head.avgpool,
            original_model.head.flatten,
        )
        self.classifier = nn.Linear(original_model.head.classifier.in_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Load classification results
def load_classification_results():
    with open(output_results_json, "r") as f:
        return json.load(f)

# Identify deepfake segments
def analyze_results(segment_labels):
    deepfake_segments = [seg for seg, data in segment_labels.items() if data["label"] == "Deepfake"]
    flagged_intervals = []
    if deepfake_segments:
        first_deepfake = None
        tracking_deepfake = False
        consecutive_authentic = 0
        consecutive_deepfake = 0

        for seg in sorted(segment_labels.keys()):
            if seg in deepfake_segments:
                if first_deepfake is None:
                    first_deepfake = seg
                consecutive_deepfake += 1
                if consecutive_deepfake >= 3:
                    consecutive_authentic = 0
                    if not tracking_deepfake:
                        flagged_intervals.append((first_deepfake, seg))
                        tracking_deepfake = True
                    else:
                        flagged_intervals[-1] = (flagged_intervals[-1][0], seg)
            else:
                consecutive_authentic += 1
                first_deepfake = None
                if consecutive_authentic >= 3:
                    tracking_deepfake = False
                    consecutive_deepfake = 0

    return merge_intervals(flagged_intervals, segment_labels)

# Merge close intervals
def merge_intervals(intervals, segment_labels):
    merged_intervals = []
    for interval in intervals:
        if merged_intervals and (interval[0] - merged_intervals[-1][1]) < 2:
            merged_intervals[-1] = (merged_intervals[-1][0], interval[1])
        else:
            merged_intervals.append(interval)
    return final_decision(merged_intervals, segment_labels)

# Final decision
def final_decision(merged_intervals, segment_labels):
    first_segment = min(segment_labels.keys())
    last_segment = max(segment_labels.keys())
    result = {}

    if len(merged_intervals) == 1 and merged_intervals[0] == (first_segment, last_segment):
        result["verdict"] = "Video is deepfake"
    elif not merged_intervals:
        result["verdict"] = "Video is AUTHENTIC"
    else:
        result["verdict"] = "Video is flagged as MANIPULATED"
        result["deepfake_timestamps"] = [
            {"start": start, "end": end} for start, end in merged_intervals
        ]

    with open(result_summary_json, "w") as f:
        json.dump(result, f, indent=4)
    
    with open(result_summary_txt, "w") as f:
        f.write(result["verdict"] + "\n")
        if "deepfake_timestamps" in result:
            f.write("Deepfake timestamps (seconds):\n")
            for interval in result["deepfake_timestamps"]:
                f.write(f"- {interval['start']}s to {interval['end']}s\n")

    return result

if __name__ == "__main__":
    classification_results = load_classification_results()
    final_result = analyze_results(classification_results)
    print("[Final Output]", final_result["verdict"])
