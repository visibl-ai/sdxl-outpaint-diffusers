import modal
import argparse
from typing import Optional
import io
from pathlib import Path
import requests
import tempfile
import os
from fastapi import FastAPI, Request
from fastapi.responses import Response

CACHE_DIR = "/cache"
RESULTS_DIR = "/results"  # Define results directory as absolute path
MINUTES = 60

app = modal.App("outpaint")
web_app = FastAPI()

# Create the base image and include local dependencies
image = (
    modal.Image.debian_slim(python_version="3.11.6")
    .run_commands(
        "apt-get update",
        "apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6"
    )
    .pip_install_from_requirements("requirements.txt")
    .add_local_file("outpaint.py", "/root/outpaint.py", copy=True)
    .add_local_file("pipeline_fill_sd_xl.py", "/root/pipeline_fill_sd_xl.py", copy=True)
    .add_local_file("controlnet_union.py", "/root/controlnet_union.py", copy=True)
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster downloads
            "HF_HUB_CACHE": CACHE_DIR,
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1",
        }
    )
)

# Import after defining image to ensure files are available
from outpaint import init_model, main as inference_fn

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("results", create_if_missing=True)

@app.cls(
    image=image,
    gpu="A10G",
    timeout=5 * MINUTES,
    volumes={CACHE_DIR: cache_volume, RESULTS_DIR: results_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
class Inference:
    @modal.enter()
    def load_pipeline(self):
        self.pipe = init_model(cache_dir=CACHE_DIR)

    @modal.method()
    def run(self, input: str, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0,
            ratio: str = None, prompt: str = "", steps: int = 20, overlap: int = 10,
            alignment: str = "Middle", resize: str = "Full", custom_resize: int = 50, batch: str = None):
        
        # If input is a URL, download it first
        if input.startswith(('http://', 'https://')):
            input = self._download_and_save_image(input)
        
        # Set output path in results directory
        output_path = os.path.join(RESULTS_DIR, "output.png")
        
        # Create a dict of all arguments except 'self'
        args_dict = {
            'input': input,
            'output': output_path,  # Specify the output path
            'left': left,
            'right': right,
            'top': top,
            'bottom': bottom,
            'ratio': ratio,
            'prompt': prompt,
            'steps': steps,
            'overlap': overlap,
            'alignment': alignment,
            'resize': resize,
            'custom_resize': custom_resize,
            'batch': batch
        }
        args = argparse.Namespace(**args_dict)
        
        # Run inference
        result_path = inference_fn(is_cli=False, args=args)
        
        # Read the result and return as bytes
        with open(result_path, 'rb') as f:
            return f.read()
        
    def _download_and_save_image(self, url: str) -> str:
        """Download image from URL and save to a temporary file."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Create a temporary file with .png extension
            temp_dir = tempfile.mkdtemp()
            temp_path = os.path.join(temp_dir, "input.png")
            
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            return temp_path
        except Exception as e:
            raise ValueError(f"Failed to download image from URL: {e}")

@web_app.post("/inference")
async def inference_endpoint(request: Request):
    data = await request.json()
    result = await Inference().run.remote.aio(**data)
    return Response(content=result, media_type="image/png")

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    return web_app


