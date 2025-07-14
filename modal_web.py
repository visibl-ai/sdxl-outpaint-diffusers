import logging
import os

import modal
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config import modal_settings, settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from modal import App

app = App(modal_settings.web_app_id)

web_image = modal.Image.debian_slim().pip_install("fastapi[standard]", "requests")

OutpaintInference = modal.Cls.from_name(
    modal_settings.inference_app_id, "OutpaintInference"
)


@app.function(image=web_image, cpu=0.25, memory=512, enable_memory_snapshot=True)
@modal.fastapi_endpoint(method="POST")
def run(body: dict):
    """FastAPI endpoint for batched inference"""
    if not body.get("input"):
        raise HTTPException(status_code=400, detail="No input provided")

    # Start the batch processing without waiting
    handle = OutpaintInference().run_batch.spawn(body)
    logger.info(f"Batch processing started: {handle}")
    return {
        "status": "processing",
        "message": "Batch processing started",
        "job_id": handle.object_id,
    }


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
):
    OUTPAINT_API_KEY = settings.api_key

    if not OUTPAINT_API_KEY:
        logger.error("API key not configured in secrets")
        raise HTTPException(status_code=500, detail="API key not configured")

    if credentials.credentials != OUTPAINT_API_KEY:
        logger.warning("Invalid API key provided")
        raise HTTPException(status_code=401, detail="Invalid API key")
