#!/usr/bin/env python3
"""
CLI tool for image outpainting using Stable Diffusion XL with ControlNet.

Usage:
    python outpaint.py --input input.png --left 100 --right 100 --top 50 --bottom 50 [options]
    
Example:
    python outpaint.py --input input.png --left 100 --right 100 --top 50 --bottom 50 --prompt "beautiful landscape" --output result.png
"""

import argparse
import sys
from pathlib import Path
from PIL import Image, ImageDraw
import torch
from diffusers import AutoencoderKL, TCDScheduler
from diffusers.models.model_loading_utils import load_state_dict
from huggingface_hub import hf_hub_download

from controlnet_union import ControlNetModel_Union
from pipeline_fill_sd_xl import StableDiffusionXLFillPipeline

# Initialize models and pipeline
print("Initializing models...")

config_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="config_promax.json",
)

config = ControlNetModel_Union.load_config(config_file)
controlnet_model = ControlNetModel_Union.from_config(config)
model_file = hf_hub_download(
    "xinsir/controlnet-union-sdxl-1.0",
    filename="diffusion_pytorch_model_promax.safetensors",
)
state_dict = load_state_dict(model_file)
model, _, _, _, _ = ControlNetModel_Union._load_pretrained_model(
    controlnet_model, state_dict, model_file, "xinsir/controlnet-union-sdxl-1.0"
)

# Ensure CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. This app requires a GPU.")

# Use the first available GPU (usually cuda:0)
device = "cuda:0"
model.to(device=device, dtype=torch.float16)

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
).to(device)

pipe = StableDiffusionXLFillPipeline.from_pretrained(
    "SG161222/RealVisXL_V5.0_Lightning",
    torch_dtype=torch.float16,
    vae=vae,
    controlnet=model,
    variant="fp16",
).to(device)

pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)


def can_expand(source_width, source_height, target_width, target_height, alignment):
    """Checks if the image can be expanded based on the alignment."""
    if alignment in ("Left", "Right") and source_width >= target_width:
        return False
    if alignment in ("Top", "Bottom") and source_height >= target_height:
        return False
    return True


def prepare_image_and_mask(image, width, height, overlap_percentage, resize_option, custom_resize_percentage, alignment, overlap_left, overlap_right, overlap_top, overlap_bottom):
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
    background = Image.new('RGB', target_size, (255, 255, 255))
    background.paste(source, (margin_x, margin_y))

    # Create the mask
    mask = Image.new('L', target_size, 255)
    mask_draw = ImageDraw.Draw(mask)

    # Calculate overlap areas
    white_gaps_patch = 2

    left_overlap = margin_x + overlap_x if overlap_left else margin_x + white_gaps_patch
    right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width - white_gaps_patch
    top_overlap = margin_y + overlap_y if overlap_top else margin_y + white_gaps_patch
    bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height - white_gaps_patch
    
    if alignment == "Left":
        left_overlap = margin_x + overlap_x if overlap_left else margin_x
    elif alignment == "Right":
        right_overlap = margin_x + new_width - overlap_x if overlap_right else margin_x + new_width
    elif alignment == "Top":
        top_overlap = margin_y + overlap_y if overlap_top else margin_y
    elif alignment == "Bottom":
        bottom_overlap = margin_y + new_height - overlap_y if overlap_bottom else margin_y + new_height

    # Draw the mask
    mask_draw.rectangle([
        (left_overlap, top_overlap),
        (right_overlap, bottom_overlap)
    ], fill=0)

    return background, mask


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Outpaint images using Stable Diffusion XL with ControlNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Expand image by 100px on all sides
  python outpaint.py --input input.png --left 100 --right 100 --top 100 --bottom 100

  # Expand only horizontally with custom prompt
  python outpaint.py --input input.png --left 200 --right 200 --top 0 --bottom 0 --prompt "sunset sky"

  # Custom output and inference steps
  python outpaint.py --input input.png --left 0 --right 0 --top 150 --bottom 150 --output expanded.png --steps 10
  
  # Use preset aspect ratios (auto-calculates expansion)
  python outpaint.py --input photo.jpg --ratio 16:9
  python outpaint.py --input portrait.png --ratio 9:16 --prompt "scenic background"
        """
    )
    
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input image")
    parser.add_argument("--left", "-l", type=int, default=0, help="Pixels to expand on the left (default: 0)")
    parser.add_argument("--right", "-r", type=int, default=0, help="Pixels to expand on the right (default: 0)")
    parser.add_argument("--top", "-t", type=int, default=0, help="Pixels to expand on the top (default: 0)")
    parser.add_argument("--bottom", "-b", type=int, default=0, help="Pixels to expand on the bottom (default: 0)")
    parser.add_argument("--ratio", type=str, default=None,
                        choices=["9:16", "16:9", "1:1"],
                        help="Target aspect ratio (overrides individual expansion values)")
    
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file path (default: input_outpainted.png)")
    parser.add_argument("--prompt", "-p", type=str, default="",
                        help="Text prompt for generation (default: empty)")
    parser.add_argument("--steps", "-s", type=int, default=8,
                        help="Number of inference steps (default: 8)")
    parser.add_argument("--overlap", type=int, default=10,
                        help="Overlap percentage for blending (default: 10)")
    parser.add_argument("--alignment", "-a", type=str, default="Middle",
                        choices=["Middle", "Left", "Right", "Top", "Bottom"],
                        help="Alignment of original image (default: Middle)")
    parser.add_argument("--resize", type=str, default="Full",
                        choices=["Full", "50%", "33%", "25%", "Custom"],
                        help="Resize option for input image (default: Full)")
    parser.add_argument("--custom-resize", type=int, default=50,
                        help="Custom resize percentage if --resize is Custom (default: 50)")
    
    return parser.parse_args()


def outpaint_image(image_path, width=None, height=None, left=None, right=None, 
                   top=None, bottom=None, prompt="", steps=8, overlap=10, 
                   alignment="Middle", resize_option="Full", custom_resize=50):
    """
    Perform image outpainting with specified parameters.
    
    Returns:
        PIL.Image: The outpainted image
    """
    # Load the input image
    try:
        image = Image.open(image_path).convert("RGB")
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
        overlap_bottom
    )
    
    # Check if expansion is valid
    if not can_expand(background.width, background.height, target_width, target_height, alignment):
        alignment = "Middle"
    
    # Create control net image
    cnet_image = background.copy()
    cnet_image.paste(0, (0, 0), mask)
    
    # Prepare prompt
    final_prompt = f"{prompt} , high quality, 4k"
    
    # Encode prompt
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(final_prompt, "cuda", True)
    
    # Generate image
    print(f"Generating outpainted image with {steps} steps...")
    
    # Get the final image from the generator
    image = None
    for img in pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        image=cnet_image,
        num_inference_steps=steps
    ):
        image = img
    
    # Composite the final image (matching app.py logic)
    image = image.convert("RGBA")
    
    # Resize if needed to match cnet_image size
    if image.size != cnet_image.size:
        image = image.resize(cnet_image.size, Image.LANCZOS)
    
    cnet_image.paste(image, (0, 0), mask)
    
    return cnet_image


def main():
    """Main CLI function."""
    args = parse_arguments()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found.", file=sys.stderr)
        sys.exit(1)
    
    # Handle ratio mode
    if args.ratio:
        # Set target dimensions based on ratio (same as app.py)
        if args.ratio == "9:16":
            width = 720
            height = 1280
        elif args.ratio == "16:9":
            width = 1280
            height = 720
        elif args.ratio == "1:1":
            width = 1024
            height = 1024
        
        print(f"Using {args.ratio} ratio: target {width}x{height}")
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_outpainted.png"
    
    # Validate expansion values (only if not using ratio mode)
    if not args.ratio:
        if args.left < 0 or args.right < 0 or args.top < 0 or args.bottom < 0:
            print("Error: Expansion values must be non-negative.", file=sys.stderr)
            sys.exit(1)
        
        if args.left == 0 and args.right == 0 and args.top == 0 and args.bottom == 0:
            print("Error: At least one expansion dimension must be greater than 0.", file=sys.stderr)
            sys.exit(1)
    
    try:
        # Perform outpainting
        print(f"Loading image: {input_path}")
        
        if args.ratio:
            result = outpaint_image(
                input_path,
                width=width,
                height=height,
                prompt=args.prompt,
                steps=args.steps,
                overlap=args.overlap,
                alignment=args.alignment,
                resize_option=args.resize,
                custom_resize=args.custom_resize
            )
        else:
            print(f"Expanding: left={args.left}px, right={args.right}px, top={args.top}px, bottom={args.bottom}px")
            result = outpaint_image(
                input_path,
                left=args.left,
                right=args.right,
                top=args.top,
                bottom=args.bottom,
                prompt=args.prompt,
                steps=args.steps,
                overlap=args.overlap,
                alignment=args.alignment,
                resize_option=args.resize,
                custom_resize=args.custom_resize
            )
        
        # Save the result
        result.save(output_path, "PNG")
        print(f"Outpainted image saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during outpainting: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()