import modal
import argparse
import requests
import os
import time
import logging
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from urllib.parse import urlparse, urlunparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_DIR = "/cache"
RESULTS_DIR = "/results"  # Define results directory as absolute path
MINUTES = 60

from modal import App

app = App("outpaint-inference")

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
    timeout=30 * MINUTES,
    volumes={CACHE_DIR: cache_volume, RESULTS_DIR: results_volume},
    secrets=[modal.Secret.from_name("huggingface-token"), modal.Secret.from_name("visibl-secret")],
    enable_memory_snapshot=True,
    retries=1,
    max_containers=3,
)
class OutpaintInference:
    @modal.enter(snap=True)
    def load_base_models(self):
        """Load the base models which are snapshot-friendly"""
        logger.info("Loading base models (with snapshot)")
        self._model, self._vae, _ = load_model(cache_dir=CACHE_DIR, load_pipeline=False)

    @modal.enter(snap=False)
    def setup_pipeline(self):
        """Setup the pipeline which isn't snapshot-friendly"""
        logger.info("Setting up pipeline (without snapshot)")
        self._pipe = setup_model(self._model, self._vae)
        # Clear any unused memory after setup
        import torch
        torch.cuda.empty_cache()

    def _upload_to_url(self, file_path: str, url: str):
        logger.info(f"Uploading to {url}")
        start_time = time.time()
        with open(file_path, "rb") as f:
            content_type = guess_image_content_type(url) or "image/webp"
            response = requests.put(url, data=f.read(), headers={"Content-Type": content_type})
            response.raise_for_status()
        upload_time = time.time() - start_time
        logger.info(f"Upload completed in {upload_time:.2f} seconds")
        
        # Extract the base URL by removing query parameters
        parsed = urlparse(url)
        # Reconstruct URL without query parameters
        base_url = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            '',  # params
            '',  # query
            ''   # fragment
        ))
        return base_url
    
    def run(self, input: str = None, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0,
        ratio: str = None, prompt: str = "", steps: int = 20, overlap: int = 10,
        alignment: str = "Middle", resize: str = "Full", custom_resize: int = 50, 
        batch: list = None, output_url: str = None, callback_url: str = None, result_key: str = None):
        
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
        result = {}
        for i, result_path in enumerate(result_paths):
            result_key = batch[i].get("result_key") if batch else result_key
            result_key = result_key or str(i)

            output_url = batch[i]['output_url'] if batch else output_url
            # If output URL (typically a signed URL) provided, upload and return the URL
            # Otherwise just return the local path
            if output_url:
                logger.info("Using provided output URL for upload")
                result[result_key] = self._upload_to_url(result_path, output_url)
            else:
                logger.info(f"No output URL provided, returning local path: {result_path}")
                result[result_key] = result_path
            
            callback_url = batch[i].get("callback_url") if batch else callback_url
            if callback_url:
                self._post_to_callback(callback_url, {"status": "completed", "results": [result]})

        return result

    def _post_to_callback(self, callback_url: str, data: dict):
        """Helper method to post data to callback URL"""
        try:
            # Convert any PosixPath objects to strings in the data
            def convert_paths(obj):
                if isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                elif hasattr(obj, '__str__'):  # This will catch PosixPath and other path-like objects
                    return str(obj)
                return obj

            serializable_data = convert_paths(data)
            
            # Get API token from environment
            callback_token = os.environ.get("CALLBACK_API_TOKEN")
            
            # Prepare headers with token if available
            headers = {"Content-Type": "application/json"}
            if callback_token:
                headers["Authorization"] = f"Bearer {callback_token}"
            
            start_time = time.time()
            response = requests.post(
                callback_url,
                json=serializable_data,
                headers=headers
            )
            response.raise_for_status()
            callback_time = time.time() - start_time
            logger.info(f"Successfully posted to callback URL: {callback_url} in {callback_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Failed to post to callback URL: {str(e)}", exc_info=True)

    @modal.batched(max_batch_size=50, wait_ms=5000)
    async def run_batch(self, input: list[dict]) -> list[str]:
        """Process a batch of inference requests"""
        # Use first callback URL for final callback with all results
        callback_url = input[0].get("callback_url") if input else None
        try:
            # Get the valid parameter names from the run method
            valid_params = {
                "input", "left", "right", "top", "bottom", "ratio", "prompt",
                "steps", "overlap", "alignment", "resize", "custom_resize",
                "batch", "output_url", "callback_url", "result_key",
            }
            # Filter each input dict to only include valid parameters
            filtered_inputs = [
                {k: v for k, v in input_dict.items() if k in valid_params}
                for input_dict in input
            ]
            results = [self.run(**input_dict) for input_dict in filtered_inputs]

            # If callback URL is provided, post results
            DISABLED_BATCH_CALLBACK = True
            if callback_url and not DISABLED_BATCH_CALLBACK:
                self._post_to_callback(callback_url, {"status": "completed", "results": results})

            # Modal batched function expects a list
            return results
        except Exception as e:
            logger.error(f"Error in batch inference: {str(e)}", exc_info=True)
            # If callback URL is provided, post error
            if callback_url:
                self._post_to_callback(callback_url, {"status": "error", "error": str(e)})
            raise e


# Should guess content type from a signed URL
def guess_image_content_type(file_path: str) -> str | None:
    # Initialize common image types explicitly
    image_mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp',
        '.tiff': 'image/tiff',
        '.svg': 'image/svg+xml',
        '.ico': 'image/vnd.microsoft.icon'
    }

    # Handle URLs by extracting the path component
    if file_path.startswith(('http://', 'https://')):
        parsed = urlparse(file_path)
        file_path = parsed.path

    # Get the file extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    # Return the matched MIME type or None if not an image
    return image_mime_types.get(ext)


