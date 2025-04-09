import subprocess
import time

scripts = ["/content/drive/MyDrive/DeepFakeDetection/cloud.py", "/content/drive/MyDrive/DeepFakeDetection/detect.py", "/content/drive/MyDrive/DeepFakeDetection/result.py"]

start_time = time.time()

for script in scripts:
    try:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)
        print(f"{script} executed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script}: {e}")
        break

end_time = time.time()
print(f"Overall execution time: {end_time - start_time:.2f} seconds")
