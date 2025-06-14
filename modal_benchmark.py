import argparse
import time
import json
from datetime import datetime
from typing import List, Dict
import modal
import requests
from PIL import Image as PILImage
import io

OutpaintInference = modal.Cls.from_name("outpaint", "OutpaintInference")
infer = OutpaintInference().run_batch

def load_test_input():
    with open('test_input.json', 'r') as f:
        return json.load(f)

def load_image_from_url(url: str) -> PILImage.Image:
    response = requests.get(url)
    return PILImage.open(io.BytesIO(response.content))

def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark Modal OutpaintInference endpoint')
    parser.add_argument('--num-requests', type=int, default=1, help='Number of parallel requests to make')
    parser.add_argument('--input-file', type=str, default='test_input.json', help='Path to input JSON file')
    return parser.parse_args()

def run_benchmark(num_requests: int, input_file: str):
    
    # Load test data
    test_data = load_test_input()
    test_image = load_image_from_url(test_data['input'])
    
    # Track results
    start_time = time.time()
    
    # Create and spawn tasks
    tasks = []
    for _ in range(num_requests):
        # Set unique result key for each request
        test_data['result_key'] = f'test_{time.time()}'
        print(f"Submitting request {test_data['result_key']}")
        task = infer.spawn(test_data)
        print(f"Task {task.object_id} spawned")
        tasks.append(task)
    
    # Wait for all tasks to complete
    print("Waiting for all tasks to complete...")
    results = [task.get() for task in tasks]
    print(f"{len(results)} of {num_requests} tasks completed ✓")
    end_time = time.time()
    total_time = end_time - start_time
    
    # Calculate statistics
    avg_time_per_request = total_time / num_requests
    cost_per_second = 0.000306
    total_cost = cost_per_second * total_time
    
    # Print results
    print("\nLocal execution time is highly inaccurate (off by more than 5 seconds!!!)")
    print("\nPlease refer to https://modal.com/apps/visibl/main/deployed/outpaint instead")
    print("\nBenchmark Results:")
    print(f"Total Requests: {num_requests}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Request: {avg_time_per_request:.2f} seconds")
    print(f"Total Cost: ${total_cost:.6f}")
    


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args.num_requests, args.input_file)
