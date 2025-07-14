#!/usr/bin/env python3
"""
Test script for outpaint.py that validates all options and output dimensions.
Now uses batch mode for efficient testing.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image


def main():
    """Run all tests using batch mode."""
    print("Starting outpaint.py test suite (batch mode)...")

    # Ensure test directory exists
    test_dir = Path("test")
    test_dir.mkdir(exist_ok=True)

    # Check if test image exists
    test_image = test_dir / "test.png"
    if not test_image.exists():
        print(f"Creating test image at {test_image}")
        # Create a simple test image
        img = Image.new("RGB", (512, 512), color="blue")
        img.save(test_image)

    # Get test image dimensions
    input_img = Image.open(test_image)
    input_width, input_height = input_img.size
    print(f"Test image dimensions: {input_width}x{input_height}")

    # Define test cases as batch config
    test_configs = [
        # Ratio tests
        {
            "name": "Ratio 9:16",
            "input": str(test_image),
            "output": "test/test_9_16.png",
            "ratio": "9:16",
            "steps": 2,
            "expected_width": 720,
            "expected_height": 1280,
        },
        {
            "name": "Ratio 16:9",
            "input": str(test_image),
            "output": "test/test_16_9.png",
            "ratio": "16:9",
            "steps": 2,
            "expected_width": 1280,
            "expected_height": 720,
        },
        {
            "name": "Ratio 1:1",
            "input": str(test_image),
            "output": "test/test_1_1.png",
            "ratio": "1:1",
            "steps": 2,
            "expected_width": 1024,
            "expected_height": 1024,
        },
        # Dimension expansion tests
        {
            "name": "Expand all sides equally",
            "input": str(test_image),
            "output": "test/test_expand_all.png",
            "left": 100,
            "right": 100,
            "top": 100,
            "bottom": 100,
            "steps": 2,
            "expected_width": input_width + 200,
            "expected_height": input_height + 200,
        },
        {
            "name": "Expand horizontal only",
            "input": str(test_image),
            "output": "test/test_expand_horizontal.png",
            "left": 150,
            "right": 150,
            "steps": 2,
            "expected_width": input_width + 300,
            "expected_height": input_height,
        },
        {
            "name": "Expand vertical only",
            "input": str(test_image),
            "output": "test/test_expand_vertical.png",
            "top": 200,
            "bottom": 200,
            "steps": 2,
            "expected_width": input_width,
            "expected_height": input_height + 400,
        },
        {
            "name": "Expand one side only",
            "input": str(test_image),
            "output": "test/test_expand_right.png",
            "right": 300,
            "steps": 2,
            "expected_width": input_width + 300,
            "expected_height": input_height,
        },
        # Alignment tests with ratio
        {
            "name": "Ratio with Left alignment",
            "input": str(test_image),
            "output": "test/test_align_left.png",
            "ratio": "16:9",
            "alignment": "Left",
            "steps": 2,
            "expected_width": 1280,
            "expected_height": 720,
        },
        {
            "name": "Ratio with Top alignment",
            "input": str(test_image),
            "output": "test/test_align_top.png",
            "ratio": "9:16",
            "alignment": "Top",
            "steps": 2,
            "expected_width": 720,
            "expected_height": 1280,
        },
        # Resize option tests
        {
            "name": "Ratio with 50% resize",
            "input": str(test_image),
            "output": "test/test_resize_50.png",
            "ratio": "1:1",
            "resize": "50%",
            "steps": 2,
            "expected_width": 1024,
            "expected_height": 1024,
        },
        {
            "name": "Expand with 25% resize",
            "input": str(test_image),
            "output": "test/test_resize_25.png",
            "left": 100,
            "right": 100,
            "resize": "25%",
            "steps": 2,
            "expected_width": input_width + 200,
            "expected_height": input_height,
        },
        # Prompt test
        {
            "name": "With prompt",
            "input": str(test_image),
            "output": "test/test_prompt.png",
            "ratio": "16:9",
            "prompt": "beautiful landscape",
            "steps": 2,
            "expected_width": 1280,
            "expected_height": 720,
        },
        # Overlap test
        {
            "name": "Custom overlap",
            "input": str(test_image),
            "output": "test/test_overlap.png",
            "left": 50,
            "right": 50,
            "overlap": 20,
            "steps": 2,
            "expected_width": input_width + 100,
            "expected_height": input_height,
        },
        # Edge cases
        {
            "name": "Minimal expansion",
            "input": str(test_image),
            "output": "test/test_minimal.png",
            "right": 10,
            "steps": 2,
            "expected_width": input_width + 10,
            "expected_height": input_height,
        },
        {
            "name": "Large expansion",
            "input": str(test_image),
            "output": "test/test_large.png",
            "left": 500,
            "right": 500,
            "steps": 2,
            "expected_width": input_width + 1000,
            "expected_height": input_height,
        },
    ]

    # Extract test metadata (name and expected dimensions)
    test_metadata = []
    batch_configs = []

    for config in test_configs:
        # Save metadata
        test_metadata.append(
            {
                "name": config.pop("name"),
                "output": config["output"],
                "expected_width": config.pop("expected_width"),
                "expected_height": config.pop("expected_height"),
            }
        )
        # Add to batch
        batch_configs.append(config)

    # Write batch config file
    batch_file = test_dir / "test_batch.json"
    with open(batch_file, "w") as f:
        json.dump(batch_configs, f, indent=2)

    print(f"Created batch config with {len(batch_configs)} tests")

    # Run batch test
    print(f"\n{'='*60}")
    print("Running batch tests...")
    start_time = time.time()

    command = f"python outpaint.py --batch {batch_file}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    elapsed_time = time.time() - start_time
    print(f"Batch processing completed in {elapsed_time:.2f}s")

    if result.returncode != 0:
        print(f"❌ Batch processing failed with code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False

    # Validate results
    print(f"\n{'='*60}")
    print("Validating results...")

    passed = 0
    failed = 0

    for metadata in test_metadata:
        print(f"\nTest: {metadata['name']}")
        print(f"Output: {metadata['output']}")
        print(f"Expected: {metadata['expected_width']}x{metadata['expected_height']}")

        output_path = Path(metadata["output"])
        if not output_path.exists():
            print(f"❌ FAILED: Output file not created")
            failed += 1
            continue

        # Check dimensions
        output_image = Image.open(output_path)
        actual_width, actual_height = output_image.size
        print(f"Actual: {actual_width}x{actual_height}")

        # Allow small differences due to VAE requirements
        width_diff = abs(actual_width - metadata["expected_width"])
        height_diff = abs(actual_height - metadata["expected_height"])

        if width_diff <= 8 and height_diff <= 8:
            print(f"✅ PASSED")
            passed += 1
        else:
            print(f"❌ FAILED: Dimensions mismatch")
            failed += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Test Summary:")
    print(f"Total tests: {len(test_metadata)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(test_metadata)*100):.1f}%")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Average time per image: {elapsed_time/len(test_metadata):.2f}s")

    # Cleanup
    cleanup = input("\nDelete test files? (y/n): ").lower().strip() == "y"
    if cleanup:
        # Delete batch config
        if batch_file.exists():
            batch_file.unlink()
            print(f"Deleted: {batch_file}")

        # Delete output images
        for metadata in test_metadata:
            output_file = Path(metadata["output"])
            if output_file.exists():
                output_file.unlink()
                print(f"Deleted: {output_file}")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
