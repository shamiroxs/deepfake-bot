import os

# File paths
classification_results_path = "./data/output/classification_results.txt"
bg_classification_results_path = "./data/output/bgclassification_result.txt"
final_results_path = "./data/output/final_classification_result.txt"

def load_classification_results(file_path):
    results = {}
    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(": ")
            if len(parts) == 2:
                segment, classification = parts
                results[segment] = classification.lower()
    return results

def combine_results():
    # Load classification results
    face_results = load_classification_results(classification_results_path)
    bg_results = load_classification_results(bg_classification_results_path)
    
    final_results = {}
    for segment, face_classification in face_results.items():
        if face_classification == "deepfake":
            final_results[segment] = "Deepfake"
        else:
            bg_classification = bg_results.get(segment, "authentic")
            if bg_classification == "authentic":
                final_results[segment] = "Authentic"
            else:
                final_results[segment] = "Deepfake"
    
    # Save final classification results
    with open(final_results_path, "w") as file:
        for segment, classification in sorted(final_results.items()):
            file.write(f"{segment}: {classification}\n")
    
    print(f"Final classification results saved to {final_results_path}")

if __name__ == "__main__":
    combine_results()
