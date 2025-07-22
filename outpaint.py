#!/usr/bin/env python3
"""
CLI tool for image outpainting using Stable Diffusion XL with ControlNet.

Usage:
    # Single image mode
    python outpaint.py --input input.png --left 100 --right 100 --top 50 --bottom 50 [options]

    # Batch mode with JSON config
    python outpaint.py --batch config.json

Example:
    python outpaint.py --input input.png --left 100 --right 100 --top 50 --bottom 50 --prompt "beautiful landscape" --output result.png

Batch config JSON format:
    [
        {
            "input": "image1.png",
            "output": "output1.png",
            "left": 100, "right": 100, "top": 0, "bottom": 0,
            "prompt": "sunset sky"
        },
        {
            "input": "image2.png",
            "ratio": "16:9",
            "alignment": "Left"
        }
    ]
"""
import datetime
import logging
import os
import tempfile
import time

import requests

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

logger.info(f"Script execution started at: {datetime.datetime.now().isoformat()}")

# Time imports
import_start = time.time()
logger.info(f"Starting imports at: {datetime.datetime.now().isoformat()}")

import argparse

logger.info(f"  argparse imported in {time.time() - import_start:.2f}s")
import sys

logger.info(f"  sys imported in {time.time() - import_start:.2f}s")

import json

logger.info(f"  json imported in {time.time() - import_start:.2f}s")
from pathlib import Path

logger.info(f"  pathlib imported in {time.time() - import_start:.2f}s")
from PIL import Image, ImageDraw

logger.info(f"  PIL imported in {time.time() - import_start:.2f}s")
import torch

logger.info(f"  torch imported in {time.time() - import_start:.2f}s")
from diffusers import AutoencoderKL, TCDScheduler

logger.info(f"  diffusers imports in {time.time() - import_start:.2f}s")
from diffusers.models.model_loading_utils import load_state_dict

logger.info(f"  load_state_dict imported in {time.time() - import_start:.2f}s")
from huggingface_hub import hf_hub_download

logger.info(f"  hf_hub_download imported in {time.time() - import_start:.2f}s")

from controlnet_union import ControlNetModel_Union

logger.info(f"  ControlNetModel_Union imported in {time.time() - import_start:.2f}s")
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

logger.info(f"Total import time: {time.time() - import_start:.2f}s")

logger.info(f"Time since script start: {time.time() - import_start:.2f}s")

_MODEL_INITIALIZED = False
pipe = None  # Global variable for the pipeline


def init_model(*, cache_dir=None):
    """Initialize model by loading components and moving them to GPU."""
    global _MODEL_INITIALIZED, pipe
    if _MODEL_INITIALIZED:
        return pipe

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This app requires a GPU.")

    # Load model components on CPU first
    model, vae, pipe = load_model(cache_dir=cache_dir)

    # Move to GPU and setup
    pipe = setup_model(model, vae)
    _MODEL_INITIALIZED = True

    return pipe


def load_model(*, cache_dir=None, load_pipeline=True):
    """Load model components on CPU before GPU initialization."""
    global pipe
    # Set HuggingFace cache directory before any HF imports
    # This ensures all HF libraries use the same cache location
    CACHE_DIR = cache_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".cache", "huggingface"
    )
    os.environ["HF_HOME"] = CACHE_DIR
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(CACHE_DIR, "hub")
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "datasets")

    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_DIR, exist_ok=True)

    # Initialize models and pipeline
    logger.info(f"Starting model loading at {datetime.datetime.now().isoformat()}...")
    init_start = time.time()

    # Download ControlNet config
    logger.info("Downloading ControlNet config...")
    config_start = time.time()
    config_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
        filename="config_promax.json",
    )
    logger.info(f"Config download completed in {time.time() - config_start:.2f}s")

    # Load ControlNet model
    logger.info("Loading ControlNet model...")
    controlnet_start = time.time()
    logger.info("  Loading config...")
    config = ControlNetModel_Union.load_config(config_file)
    logger.info(f"  Config loaded in {time.time() - controlnet_start:.2f}s")
    logger.info("  Creating model from config...")
    controlnet_model = ControlNetModel_Union.from_config(config)
    logger.info(f"  Model created in {time.time() - controlnet_start:.2f}s")
    logger.info("  Downloading model weights...")
    download_start = time.time()
    model_file = hf_hub_download(
        "xinsir/controlnet-union-sdxl-1.0",
        filename="diffusion_pytorch_model_promax.safetensors",
    )
    logger.info(f"  Model weights downloaded in {time.time() - download_start:.2f}s")
    logger.info("  Loading state dict...")
    state_dict_start = time.time()
    state_dict = load_state_dict(model_file)
    logger.info(f"  State dict loaded in {time.time() - state_dict_start:.2f}s")
    model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
        controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
    )
    logger.info(f"ControlNet loaded in {time.time() - controlnet_start:.2f}s")

    # Load VAE
    logger.info("Loading VAE...")
    vae_start = time.time()
    logger.info("  Downloading/loading VAE model...")
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
    )
    logger.info(f"  VAE loaded in {time.time() - vae_start:.2f}s")

    # Load pipeline only if requested
    pipe = None
    if load_pipeline:
        logger.info("Loading StableDiffusion pipeline...")
        pipe_start = time.time()
        logger.info("  Downloading/loading pipeline components...")
        pipe = StableDiffusionXLFillPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0_Lightning",
            torch_dtype=torch.float16,
            vae=vae,
            controlnet=model,
            variant="fp16",
        )
        logger.info(f"  Pipeline created in {time.time() - pipe_start:.2f}s")

    logger.info(f"Total CPU loading time: {time.time() - init_start:.2f}s")
    return model, vae, pipe


def setup_model(model, vae):
    """Move model components to GPU and configure them."""
    global pipe
    device = "cuda:0"

    # Move model to GPU
    logger.info("Moving ControlNet to GPU...")
    gpu_start = time.time()
    model.to(device=device, dtype=torch.float16)
    logger.info(f"ControlNet moved to GPU in {time.time() - gpu_start:.2f}s")

    # Move VAE to GPU
    logger.info("Moving VAE to GPU...")
    vae_gpu_start = time.time()
    vae = vae.to(device)
    logger.info(f"VAE moved to GPU in {time.time() - vae_gpu_start:.2f}s")

    # Create or move pipeline to GPU and configure scheduler
    logger.info("Moving pipeline to GPU...")
    pipe_gpu_start = time.time()

    if pipe is None:
        logger.info("Creating new pipeline...")
        pipe = StableDiffusionXLFillPipeline.from_pretrained(
            "SG161222/RealVisXL_V5.0_Lightning",
            torch_dtype=torch.float16,
            vae=vae,
            controlnet=model,
            variant="fp16",
        )

    pipe = pipe.to(device)
    pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)
    pipe.vae = vae
    pipe.controlnet = model

    logger.info(f"Pipeline moved to GPU in {time.time() - pipe_gpu_start:.2f}s")

    return pipe


def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Checks if the image can be expanded based on the alignment."""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True


def prepare_image_and_mask(
    image,
    width,
    height,
    overlap_percentage,
    resize_option,
    custom_resize_percentage,
    alignment,
    overlap_left,
    overlap_right,
    overlap_top,
    overlap_bottom,
):
    start_time = time.time()
    logger.info(
        f"Preparing image and mask - target size: {width}x{height}, alignment: {alignment}"
    )

    target_size = (width, height)

    # Calculate the scaling factor to fit the image within the target size
    scale_factor = min(target_size[0] / image.width, target_size[1] / image.height)
    new_width = int(image.width * scale_factor)
    new_height = int(image.height * scale_factor)

    # Resize the source image to fit within target size
    source = image.resize((new_width, new_height), Image.LANCZOS)

    # Apply resize option using percentages
    if resize_option == "Full":
        resize_percentage = 100
    elif resize_option == "50%":
        resize_percentage = 50
    elif resize_option == "33%":
        resize_percentage = 33
    elif resize_option == "25%":
        resize_percentage = 25
    else:  # Custom
        resize_percentage = custom_resize_percentage

    # Calculate new dimensions based on percentage
    resize_factor = resize_percentage / 100
    new_width = int(source.width * resize_factor)
    new_height = int(source.height * resize_factor)

    # Ensure minimum size of 64 pixels
    new_width = max(new_width, 64)
    new_height = max(new_height, 64)

    # Resize the image
    source = source.resize((new_width, new_height), Image.LANCZOS)

    # Calculate the overlap in pixels based on the percentage
    overlap_x = int(new_width * (overlap_percentage / 100))
    overlap_y = int(new_height * (overlap_percentage / 100))

    # Ensure minimum overlap of 1 pixel
    overlap_x = max(overlap_x, 1)
    overlap_y = max(overlap_y, 1)

    # Calculate margins based on alignment
    if alignment == "Middle":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Left":
        margin_x = 0
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Right":
        margin_x = target_size[0] - new_width
        margin_y = (target_size[1] - new_height) // 2
    elif alignment == "Top":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = 0
    elif alignment == "Bottom":
        margin_x = (target_size[0] - new_width) // 2
        margin_y = target_size[1] - new_height

    # Adjust margins to eliminate gaps
    margin_x = max(0, min(margin_x, target_size[0] - new_width))
    margin_y = max(0, min(margin_y, target_size[1] - new_height))

    # Create a new background image and paste the resized source image
    background = Image.new("RGB", target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    # Create the mask
    mask = Image.new("L", target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Calculate overlap areas
    white_gaps_patch = 2

    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = (
        margin_x + new_width - overlap_x
        if overlap_right
        else margin_x + new_width - white_gaps_patch
    )
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = (
        margin_y + new_height - overlap_y
        if overlap_bottom
        else margin_y + new_height - white_gaps_patch
    )

    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = (
            margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
        )
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = (
            margin_y + new_height - overlap_y
            if overlap_bottom
            else margin_y + new_height
        )

    # Draw the mask
    mask_draw.rectangle(
        [(left_overlap, top_overlap), (right_overlap, bottom_overlap)], fill=0
    )

    logger.info(f"Image and mask prepared in {time.time() - start_time:.2f}s")
    return background, mask


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Outpaint images using Stable Diffusion XL with ControlNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image mode - Expand image by 100px on all sides
  python outpaint.py --input input.png --left 100 --right 100 --top 100 --bottom 100

  # Expand only horizontally with custom prompt
  python outpaint.py --input input.png --left 200 --right 200 --top 0 --bottom 0 --prompt "sunset sky"

  # Use preset aspect ratios
  python outpaint.py --input photo.jpg --ratio 16:9

  # Batch mode - Process multiple images from JSON config
  python outpaint.py --batch batch_config.json
        """,
    )

    # Add mutually exclusive group for single vs batch mode
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--input", "-i", type=str, help="Path to input image (single mode)"
    )
    mode_group.add_argument(
        "--batch", type=str, help="Path to JSON config file for batch processing"
    )

    # Single mode arguments
    parser.add_argument(
        "--left",
        "-l",
        type=int,
        default=0,
        help="Pixels to expand on the left (default: 0)",
    )
    parser.add_argument(
        "--right",
        "-r",
        type=int,
        default=0,
        help="Pixels to expand on the right (default: 0)",
    )
    parser.add_argument(
        "--top",
        "-t",
        type=int,
        default=0,
        help="Pixels to expand on the top (default: 0)",
    )
    parser.add_argument(
        "--bottom",
        "-b",
        type=int,
        default=0,
        help="Pixels to expand on the bottom (default: 0)",
    )
    parser.add_argument(
        "--ratio",
        type=str,
        default=None,
        choices=["9:16", "16:9", "1:1"],
        help="Target aspect ratio (overrides individual expansion values)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output file path (default: input_outpainted.png)",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        type=str,
        default="",
        help="Text prompt for generation (default: empty)",
    )
    parser.add_argument(
        "--steps",
        "-s",
        type=int,
        default=20,
        help="Number of inference steps (default: 20)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=10,
        help="Overlap percentage for blending (default: 10)",
    )
    parser.add_argument(
        "--alignment",
        "-a",
        type=str,
        default="Middle",
        choices=["Middle", "Left", "Right", "Top", "Bottom"],
        help="Alignment of original image (default: Middle)",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default="Full",
        choices=["Full", "50%", "33%", "25%", "Custom"],
        help="Resize option for input image (default: Full)",
    )
    parser.add_argument(
        "--custom-resize",
        type=int,
        default=50,
        help="Custom resize percentage if --resize is Custom (default: 50)",
    )

    return parser.parse_args()


def outpaint_image(
    image_path,
    width=None,
    height=None,
    left=None,
    right=None,
    top=None,
    bottom=None,
    prompt="",
    steps=20,
    overlap=10,
    alignment="Middle",
    resize_option="Full",
    custom_resize=50,
):
    """
    Perform image outpainting with specified parameters.

    Returns:
        PIL.Image: The outpainted image
    """
    total_start = time.time()

    # Load the input image
    logger.info(f"Loading input image: {image_path}")
    load_start = time.time()
    try:
        image = Image.open(image_path).convert("RGB")
        logger.info(
            f"Image loaded in {time.time() - load_start:.2f}s - size: {image.size}"
        )
    except Exception as e:
        raise ValueError(f"Failed to load image: {e}")

    # Use provided width/height or calculate from expansion amounts
    if width is not None and height is not None:
        target_width = width
        target_height = height
    else:
        target_width = image.width + left + right
        target_height = image.height + top + bottom

    # Determine overlap settings based on expansion directions
    if width is not None and height is not None:
        # When using direct dimensions (ratio mode), enable all overlaps
        overlap_left = True
        overlap_right = True
        overlap_top = True
        overlap_bottom = True
    else:
        # When using expansion amounts
        overlap_left = left > 0
        overlap_right = right > 0
        overlap_top = top > 0
        overlap_bottom = bottom > 0

    # Prepare image and mask
    prep_start = time.time()
    background, mask = prepare_image_and_mask(
        image,
        target_width,
        target_height,
        overlap,
        resize_option,
        custom_resize,
        alignment,
        overlap_left,
        overlap_right,
        overlap_top,
        overlap_bottom,
    )

    # Check if expansion is valid
    if not can_expand(
        background.width, background.height, target_width, target_height, alignment
    ):
        logger.info("Expansion not valid for alignment, switching to Middle")
        alignment = "Middle"

    # Create control net image
    logger.info("Creating ControlNet input image...")
    cnet_start = time.time()
    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)
    logger.info(f"ControlNet input created in {time.time() - cnet_start:.2f}s")

    # Prepare prompt
    final_prompt = f"{prompt} , high quality, 4k"
    logger.info(f"Using prompt: {final_prompt}")

    # Encode prompt
    logger.info("Encoding prompt...")
    encode_start = time.time()
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(final_prompt, "cuda", True)
    logger.info(f"Prompt encoded in {time.time() - encode_start:.2f}s")

    # Generate image
    logger.info(f"Starting image generation with {steps} steps...")
    gen_start = time.time()

    # Get the final image from the generator
    image = None
    for step_num, img in enumerate(
        pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
            image=cnet_image,
            num_inference_steps=steps,
        )
    ):
        image = img
        logger.info(f"  Step {step_num + 1}/{steps} completed")

    logger.info(f"Image generation completed in {time.time() - gen_start:.2f}s")

    # Composite the final image (matching app.py logic)
    logger.info("Compositing final image...")
    comp_start = time.time()
    image = image.convert("RGBA")

    # Resize if needed to match cnet_image size
    if image.size != cnet_image.size:
        logger.info(f"Resizing generated image from {image.size} to {cnet_image.size}")
        image = image.resize(cnet_image.size, Image.LANCZOS)

    cnet_image.paste(image, (0, 0), mask)
    logger.info(f"Compositing completed in {time.time() - comp_start:.2f}s")

    logger.info(f"Total outpainting time: {time.time() - total_start:.2f}s")
    return cnet_image


def process_single_image(config):
    """Process a single image with given configuration."""
    # Validate input file
    if config["input"].startswith(("http://", "https://")):
        input_path = Path(download_and_save_image(config["input"]))
    else:
        input_path = Path(config["input"])

    if not input_path.exists():
        raise FileNotFoundError(f"Input file '{config['input']}' not found")

    # Handle ratio mode
    width = None
    height = None
    if config.get("ratio"):
        # Set target dimensions based on ratio (same as app.py)
        if config["ratio"] == "9:16":
            width = 720
            height = 1280
        elif config["ratio"] == "16:9":
            width = 1280
            height = 720
        elif config["ratio"] == "1:1":
            width = 1024
            height = 1024

        logger.info(f"Using {config['ratio']} ratio: target {width}x{height}")

    # Determine output path
    if config.get("output"):
        output_path = Path(config["output"])
    else:
        output_path = input_path.parent / f"{input_path.stem}_outpainted.png"

    # Validate expansion values (only if not using ratio mode)
    if not config.get("ratio"):
        left = config.get("left", 0)
        right = config.get("right", 0)
        top = config.get("top", 0)
        bottom = config.get("bottom", 0)

        if left < 0 or right < 0 or top < 0 or bottom < 0:
            raise ValueError("Expansion values must be non-negative")

        if left == 0 and right == 0 and top == 0 and bottom == 0:
            raise ValueError("At least one expansion dimension must be greater than 0")

    # Perform outpainting
    logger.info(f"Starting outpainting process for: {input_path}")
    process_start = time.time()

    if config.get("ratio"):
        result = outpaint_image(
            input_path,
            width=width,
            height=height,
            prompt=config.get("prompt", ""),
            steps=config.get("steps", 20),
            overlap=config.get("overlap", 10),
            alignment=config.get("alignment", "Middle"),
            resize_option=config.get("resize", "Full"),
            custom_resize=config.get("custom_resize", 50),
        )
    else:
        result = outpaint_image(
            input_path,
            left=config.get("left", 0),
            right=config.get("right", 0),
            top=config.get("top", 0),
            bottom=config.get("bottom", 0),
            prompt=config.get("prompt", ""),
            steps=config.get("steps", 20),
            overlap=config.get("overlap", 10),
            alignment=config.get("alignment", "Middle"),
            resize_option=config.get("resize", "Full"),
            custom_resize=config.get("custom_resize", 50),
        )

    # Save the result
    logger.info(f"Saving result to: {output_path}")
    save_start = time.time()
    result.save(output_path, "PNG")
    logger.info(f"Image saved in {time.time() - save_start:.2f}s")

    logger.info(f"Total process time: {time.time() - process_start:.2f}s")
    return output_path


def download_and_save_image(url: str) -> str:
    """Download image from URL and save to a temporary file."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Create a temporary file with .png extension
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "input.png")

        with open(temp_path, "wb") as f:
            f.write(response.content)

        return temp_path
    except Exception as e:
        raise ValueError(f"Failed to download image from URL: {e}")


def main(*, is_cli: bool = True, args=None):
    """Main CLI function."""
    logger.info(
        f"Main function execution started at: {datetime.datetime.now().isoformat()}"
    )
    if is_cli or args is None:
        args = parse_arguments()

    try:
        if args.batch:
            # Batch mode
            if is_cli:
                logger.info(f"Loading batch configuration from: {args.batch}")
                with open(args.batch, "r") as f:
                    batch_configs = json.load(f)
            else:
                batch_configs = args.batch

                if not isinstance(batch_configs, list):
                    raise ValueError("Batch config must be a JSON array")

            logger.info(f"Processing {len(batch_configs)} images in batch mode")
            batch_start = time.time()

            successful = 0
            failed = 0
            output_paths = []

            for i, config in enumerate(batch_configs, 1):
                logger.info(f"\n{'='*60}")
                logger.info(
                    f"Processing image {i}/{len(batch_configs)}: {config.get('input', 'unknown')}"
                )

                try:
                    output_path = process_single_image(config)
                    print(f"✓ Image {i}/{len(batch_configs)}: {output_path}")
                    output_paths.append(output_path)
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to process image {i}: {e}")
                    print(
                        f"✗ Image {i}/{len(batch_configs)} failed: {e}", file=sys.stderr
                    )
                    failed += 1

            logger.info(f"\n{'='*60}")
            logger.info(
                f"Batch processing completed in {time.time() - batch_start:.2f}s"
            )
            logger.info(f"Successful: {successful}, Failed: {failed}")
            print(f"\nBatch complete: {successful} successful, {failed} failed")

            return output_paths
        else:
            # Single image mode - convert args to config dict
            config = {
                "input": args.input,
                "output": args.output,
                "left": args.left,
                "right": args.right,
                "top": args.top,
                "bottom": args.bottom,
                "ratio": args.ratio,
                "prompt": args.prompt,
                "steps": args.steps,
                "overlap": args.overlap,
                "alignment": args.alignment,
                "resize": args.resize,
                "custom_resize": args.custom_resize,
            }

            output_path = process_single_image(config)
            print(f"\nOutpainted image saved to: {output_path}")
            return [output_path]

    except Exception as e:
        logger.error(f"Error: {e}")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    init_model()
    main()
