#!/bin/bash

# Deployment script for Diffusers Image Outpaint on H100 GPU server

echo "Starting deployment of Diffusers Image Outpaint..."

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi

echo "GPU Information:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN environment variable is not set."
    echo "Please set it by running:"
    echo "export HF_TOKEN=your_token_here"
    echo "You can get your token from https://huggingface.co/settings/tokens"
    exit 1
fi

echo "HF_TOKEN is set, proceeding with deployment..."


# Activate virtual environment
source .venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install --cache-dir=.venv/pip-cache -r requirements.txt
if ! pip install --cache-dir=.venv/pip-cache -r requirements.txt; then
    echo "ERROR: Failed to install requirements"
    exit 1
fi

# Set environment variables for optimal H100 performance
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Optional: Enable TF32 for better performance on H100
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# Start the application
python outpaint.py --input test/test.png --ratio 9:16 --output test/test_9_16.png