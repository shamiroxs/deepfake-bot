import os
import re

# File paths
classification_results_path = "./data/output/classification_results.txt"
bg_classification_results_path = "./data/output/bgclassification_result.txt"
final_results_path = "./data/output/final_classification_result.txt"

def load_classification_results(file_path):
    """Loads classification results from a file into a dictionary."""
    results = {}
    if not os.path.exists(file_path):  # Check if file exists
        print(f"Warning: {file_path} not found.")
        return results

    with open(file_path, "r") as file:
        for line in file:
            parts = line.strip().split(": ")
            if len(parts) == 2:
                segment, classification = parts
                results[segment] = classification.lower()
    return results

def natural_sort_key(segment):
    """Splits segment names into numbers and text for natural sorting."""
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', segment)]

def combine_results():
    """Combines face and background classification results, then writes sorted results to a file."""
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

    # Save final classification results in sorted order
    with open(final_results_path, "w") as file:
        for segment, classification in sorted(final_results.items(), key=lambda x: natural_sort_key(x[0])):
            file.write(f"{segment}: {classification}\n")
            print(f"Segment {segment}: {classification}")

    print(f"Final classification results saved to {final_results_path}")

if __name__ == "__main__":
    combine_results()
