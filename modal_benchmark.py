import argparse
import io
import json
import math
import time
from datetime import datetime
from typing import Dict, List

import modal
import requests
from PIL import Image as PILImage

from config import modal_settings

OutpaintInference = modal.Cls.from_name("outpaint-inference", "OutpaintInference")
infer = OutpaintInference().run_batch


def load_test_input():
    with open("test_input.json", "r") as f:
        return json.load(f)


def load_image_from_url(url: str) -> PILImage.Image:
    response = requests.get(url)
    return PILImage.open(io.BytesIO(response.content))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark Modal OutpaintInference endpoint"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help="Number of parallel requests to make",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="test_input.json",
        help="Path to input JSON file",
    )
    return parser.parse_args()


def run_benchmark(num_requests: int, input_file: str):

    # Load test data
    test_data = load_test_input()
    test_image = load_image_from_url(test_data["input"])

    # Track results
    start_time = time.time()

    # Create and spawn tasks
    tasks = []
    for _ in range(num_requests):
        # Set unique result key for each request
        test_data["result_key"] = f"test_{int(time.time() * 1000)}"
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
    containers_spawned = min(
        math.ceil(num_requests / modal_settings.max_batch_size),
        modal_settings.max_containers,
    )
    avg_time_per_request = total_time / num_requests * containers_spawned
    cost_per_second = 0.000306
    total_cost = cost_per_second * total_time * containers_spawned
    avg_cost_per_request = total_cost / num_requests

    # Print results
    print("\nBenchmark Results:")
    print(f"Total Requests: {num_requests}")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Average Time per Request: {avg_time_per_request:.2f} seconds")
    print(f"Average cost per request: ${avg_cost_per_request:.6f}")
    print(f"Containers spawned: {containers_spawned}")
    print(f"Total Cost: ${total_cost:.6f}")


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(args.num_requests, args.input_file)
