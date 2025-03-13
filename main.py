import subprocess
import sys
import os

SELECTION_FILE = "./selection.txt"

# List of scripts to execute in order
scripts = ["frame.py", "classify.py", "result.py"]

for script in scripts:
    print(f"Running {script}...\n")

    # Run the script without capturing output so it prints in real-time
    result = subprocess.run(["python", script])

print("Execution completed.")
