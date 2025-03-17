import multiprocessing
import subprocess
import time

def run_classify():
    subprocess.run(["python", "classify.py"])

def run_bgclassify():
    subprocess.run(["python", "bgclassify.py"])

if __name__ == "__main__":
    start_time = time.time()

    print("Running frame.py..")
    subprocess.run(["python", "frame.py"])

    print("Starting parallel processes: classify.py & bgclassify.py\n")

    multiprocessing.set_start_method('spawn', force=True)

    process1 = multiprocessing.Process(target=run_classify)
    process2 = multiprocessing.Process(target=run_bgclassify)

    process1.start()
    process2.start()

    process1.join()
    process2.join()

    print("Running result.py...\n")
    subprocess.run(["python", "combine.py"])
    subprocess.run(["python", "result.py"])

    end_time = time.time()
    print("Execution completed.")
    print(f"Overall execution time: {end_time - start_time:.2f} seconds")
