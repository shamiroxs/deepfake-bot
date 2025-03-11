import os
import cv2
import json
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor

class DeepFakeDetection:
    def __init__(self, frames_dir="./data/videoframes", output_dir="./data/output"):
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(output_dir, "detection_results.json")
        self.bg_output_file = os.path.join(output_dir, "bgoutput.txt")

    def optical_flow_analysis(self, prev_frame, curr_frame):
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        deepflow = cv2.optflow.createOptFlow_DeepFlow()
        flow = deepflow.calc(prev_gray, curr_gray, None)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(magnitude))

    def noise_and_compression_analysis(self, frame):
        noise = np.var(cv2.GaussianBlur(frame, (5,5), 0) - frame)
        return float(noise)

    def classify_frame(self, optical_flow_score, noise_score, flow_threshold=0.5, noise_threshold=10.0):
        if optical_flow_score < flow_threshold and noise_score > noise_threshold:
            return "deepfake"
        return "authentic"

    def process_frame_pair(self, frame_idx, frame_path, prev_frame, curr_frame):
        if np.mean(curr_frame) < 5:  # Ignore almost completely black frames
            return {
                "frame_index": frame_idx,
                "frame_path": frame_path,
                "optical_flow_score": 0.0,
                "noise_score": 0.0,
                "classification": "ignored"
            }
        
        optical_flow_score = self.optical_flow_analysis(prev_frame, curr_frame)
        noise_score = self.noise_and_compression_analysis(curr_frame)
        classification = self.classify_frame(optical_flow_score, noise_score)
        result = {
            "frame_index": frame_idx,
            "frame_path": frame_path,
            "optical_flow_score": optical_flow_score,
            "noise_score": noise_score,
            "classification": classification
        }
        
        with open(self.bg_output_file, "a") as f:
            f.write(f"Frame {frame_idx}: Optical Flow Score: {optical_flow_score}, Noise Score: {noise_score}, Classification: {classification}\n")
        
        return result

    def analyze_frames(self):
        start_time = time.time()
        results = []
        frame_files = sorted(os.listdir(self.frames_dir))
        futures = []
        
        with ThreadPoolExecutor() as executor:
            for i in range(len(frame_files) - 1):
                frame_idx = i
                frame_path = os.path.join(self.frames_dir, frame_files[i])
                next_frame_path = os.path.join(self.frames_dir, frame_files[i + 1])
                
                prev_frame = cv2.imread(frame_path)
                curr_frame = cv2.imread(next_frame_path)
                
                futures.append(
                    executor.submit(
                        self.process_frame_pair, 
                        frame_idx, 
                        frame_path,
                        prev_frame, 
                        curr_frame
                    )
                )
            
            # Retrieve results
            for future in futures:
                results.append(future.result())

        # Save results to JSON
        with open(self.output_file, "w") as f:
            json.dump(results, f, indent=4)

        print(f"Analysis complete! Results saved in {self.output_file}")
        print(f"Processing time: {time.time() - start_time:.2f} seconds")

        return results

if __name__ == "__main__":
    detector = DeepFakeDetection()
    detector.analyze_frames()
