import modal
import argparse
from typing import Optional
import io
from pathlib import Path
import json
import requests
import tempfile
import os
import time
import logging
from fastapi import FastAPI, Request, HTTPException, Header, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import Response
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster downloads
        "HF_HUB_CACHE": CACHE_DIR,
        "CUDA_VISIBLE_DEVICES": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"
    })
)

# Import after defining image to ensure files are available
from outpaint import download_and_save_image, load_model, setup_model, main as inference_fn

cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("results", create_if_missing=True)

@app.cls(
    image=image,
    gpu="A10G",
    timeout=5 * MINUTES,
    volumes={CACHE_DIR: cache_volume, RESULTS_DIR: results_volume},
    secrets=[modal.Secret.from_name("huggingface-token")],
    enable_memory_snapshot=True,
)
class Inference:
    @modal.enter(snap=True)
    def load_base_models(self):
        """Load the base models which are snapshot-friendly"""
        logger.info("Loading base models (with snapshot)")
        self.model, self.vae, _ = load_model(cache_dir=CACHE_DIR, load_pipeline=False)

    @modal.enter(snap=False)
    def setup_pipeline(self):
        """Setup the pipeline which isn't snapshot-friendly"""
        logger.info("Setting up pipeline (without snapshot)")
        self.pipe = setup_model(self.model, self.vae)
        # Clear any unused memory after setup
        torch.cuda.empty_cache()

    def _upload_to_url(self, file_path: str, url: str):
        logger.info(f"Uploading to {url}")
        with open(file_path, 'rb') as f:
            response = requests.put(url, data=f.read(), headers={"Content-Type": "image/png"})
            response.raise_for_status()
        return url

    @modal.method()
    def run(self, input: str = None, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0,
            ratio: str = None, prompt: str = "", steps: int = 20, overlap: int = 10,
            alignment: str = "Middle", resize: str = "Full", custom_resize: int = 50, 
            batch: list = None, output_url: str = None):
        
        # If input is a URL, download it first
        if input and input.startswith(('http://', 'https://')):
            input = download_and_save_image(input)
        
        # Set output path in results directory
        output_path = os.path.join(RESULTS_DIR, f"output_{int(time.time())}.png")
        
        # Create a dict of all arguments except 'self'
        args_dict = {
            'input': input,
            'output': output_path,
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
        result_paths = inference_fn(is_cli=False, args=args)

        # Process the results
        result = []
        for i, result_path in enumerate(result_paths):
            output_url = batch[i]['output_url'] if batch else output_url
            # If output URL (typically a signed URL) provided, upload and return the URL
            # Otherwise just return the local path
            if output_url:
                logger.info("Using provided output URL for upload")
                result.append(self._upload_to_url(result_path, output_url))
            else:
                logger.info(f"No output URL provided, returning local path: {result_path}")
                result.append(result_path)
        
        return result

async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    OUTPAINT_API_KEY = os.environ.get("OUTPAINT_API_KEY")
    
    if not OUTPAINT_API_KEY:
        logger.error("API key not configured in secrets")
        raise HTTPException(status_code=500, detail="API key not configured")
        
    if credentials.credentials != OUTPAINT_API_KEY:
        logger.warning("Invalid API key provided")
        raise HTTPException(status_code=401, detail="Invalid API key")


@web_app.post("/inference")
async def inference_endpoint(request: Request, api_key: str = Depends(verify_api_key)):
    data = await request.json()
    if not data.get('input') and not data.get('batch'):
        return {"error": "Either input or batch must be provided"}

    uploaded_url = await Inference().run.remote.aio(**data)
    return {"url": uploaded_url}

@app.function(image=image, secrets=[modal.Secret.from_name("custom-secret")])
@modal.asgi_app()
def fastapi_app():
    return web_app


