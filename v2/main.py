import subprocess
import time

# List of scripts to execute in order
scripts = ["detect.py", "result.py"]

# Start time
start_time = time.time()

for script in scripts:
    try:
        print(f"Running {script}...")
        subprocess.run(["python", script], check=True)
        print(f"{script} executed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"Error executing {script}: {e}")
        break

# End time
end_time = time.time()

# Print overall execution time
print(f"Overall execution time: {end_time - start_time:.2f} seconds")
