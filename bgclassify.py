import os
import cv2
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

start_time = time.time()

class DeepFakeDetection:
    def __init__(self, metadata_path="./data/output/bg_metadata.json", output_dir="./data/output"):
        self.metadata_path = metadata_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, "bgclassification_result.txt")
    
    def optical_flow_analysis(self, prev_frame, curr_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        deepflow = cv2.optflow.createOptFlow_DeepFlow()
        flow = deepflow.calc(prev_gray, curr_gray, None)
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(magnitude))
    
    def noise_and_compression_analysis(self, frame):
        noise = np.var(cv2.GaussianBlur(frame, (5,5), 0) - frame)
        return float(noise)
    
    def classify_segment(self, flow_scores, noise_scores, flow_threshold=0.5, noise_threshold=10.0):
        avg_flow = np.mean(flow_scores) if flow_scores else 0.0
        avg_noise = np.mean(noise_scores) if noise_scores else 0.0
        
        if avg_flow < flow_threshold and avg_noise > noise_threshold:
            return "deepfake"
        return "authentic"
    
    def analyze_background(self):
        if not os.path.exists(self.metadata_path):
            print(f"Metadata file {self.metadata_path} not found.")
            return
        
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)
        
        segment_results = {}
        with ThreadPoolExecutor() as executor:
            for segment_id, frames in metadata.items():
                flow_scores, noise_scores = [], []
                for i in range(len(frames) - 1):
                    prev_frame = cv2.imread(frames[i]["frame_path"])
                    curr_frame = cv2.imread(frames[i + 1]["frame_path"])
                    
                    if prev_frame is None or curr_frame is None:
                        continue
                    
                    flow_scores.append(self.optical_flow_analysis(prev_frame, curr_frame))
                    noise_scores.append(self.noise_and_compression_analysis(curr_frame))
                
                segment_results[int(segment_id)] = self.classify_segment(flow_scores, noise_scores)
        
        # Save results to text file
        with open(self.output_file, "w") as f:
            for segment_id in sorted(segment_results.keys()):
                f.write(f"Segment {segment_id}: {segment_results[segment_id]}\n")
        
        print(f"Background classification complete! Results saved in {self.output_file}")

end_time = time.time()

if __name__ == "__main__":
    detector = DeepFakeDetection()
    detector.analyze_background()
    
    print(f"Total execution time: {end_time - start_time:.2f} seconds")		
