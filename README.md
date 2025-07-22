# Diffusers Image Outpaint

A powerful image outpainting tool that uses Stable Diffusion XL with ControlNet Union to intelligently expand image boundaries. This tool allows you to extend images in any direction while maintaining visual coherence with the original content.

## Features

- **Flexible Expansion**: Expand images by specific pixel amounts in any direction (left, right, top, bottom)
- **Aspect Ratio Presets**: Quick presets for common aspect ratios (16:9, 9:16, 1:1)
- **Smart Alignment**: Multiple alignment options (Middle, Left, Right, Top, Bottom) for positioning the original image
- **Customizable Overlap**: Control the blending zone between original and generated content
- **Batch Processing**: Process multiple images efficiently with JSON configuration
- **Resize Options**: Scale input images before outpainting (Full, 50%, 33%, 25%, or Custom percentage)
- **GPU Accelerated**: Optimized for CUDA-enabled GPUs with fp16 precision

## Requirements

- Python 3.8+
- CUDA-capable GPU (required)
- GPU memory: Test on an H100 and L4. Not sure anything smaller would work.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/diffusers-image-outpaint.git
cd diffusers-image-outpaint
```

2. Run the installation script:
```bash
./install.sh
```

The installation script will:
- Check for CUDA availability
- Create a virtual environment (`.venv`)
- Install all required dependencies
- Verify your HuggingFace token is set

3. Set up HuggingFace token (required for model downloads):
```bash
export HF_TOKEN=your_huggingface_token_here
```

You can get your token from https://huggingface.co/settings/tokens

## Usage

### Command Line Interface

#### Basic Usage

Expand an image by 100 pixels on all sides:
```bash
python outpaint.py --input image.jpg --left 100 --right 100 --top 100 --bottom 100
```

#### Aspect Ratio Mode

Convert an image to 16:9 aspect ratio:
```bash
python outpaint.py --input portrait.jpg --ratio 16:9
```

Available ratios: `16:9` (1280x720), `9:16` (720x1280), `1:1` (1024x1024)

#### Advanced Options

```bash
python outpaint.py \
  --input image.jpg \
  --left 200 --right 200 \
  --prompt "beautiful sunset sky" \
  --steps 30 \
  --overlap 15 \
  --alignment Left \
  --resize 50% \
  --output custom_output.png
```

### Batch Processing

Create a JSON configuration file:

```json
{
  "images": [
    {
      "input": "image1.jpg",
      "output": "output1.png",
      "left": 100,
      "right": 100,
      "top": 50,
      "bottom": 50,
      "prompt": "scenic landscape"
    },
    {
      "input": "image2.jpg",
      "ratio": "16:9",
      "alignment": "Middle",
      "overlap_percentage": 15
    }
  ]
}
```

Run batch processing:
```bash
python outpaint.py --batch batch_config.json
```

### Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--input` | Input image path | Required | - |
| `--output` | Output image path | `{input}_outpainted.png` | - |
| `--left` | Pixels to expand left | 0 | Any positive integer |
| `--right` | Pixels to expand right | 0 | Any positive integer |
| `--top` | Pixels to expand top | 0 | Any positive integer |
| `--bottom` | Pixels to expand bottom | 0 | Any positive integer |
| `--ratio` | Aspect ratio preset | None | `16:9`, `9:16`, `1:1` |
| `--prompt` | Text prompt for generation | "" | Any text |
| `--steps` | Inference steps | 20 | 1-50 recommended |
| `--overlap` | Overlap percentage | 10 | 0-50 |
| `--alignment` | Image alignment | Middle | `Middle`, `Left`, `Right`, `Top`, `Bottom` |
| `--resize` | Resize option | Full | `Full`, `50%`, `33%`, `25%`, `Custom` |
| `--custom-resize` | Custom resize percentage | 50 | 1-100 |

## Examples

### Horizontal Expansion
```bash
# Expand a landscape photo horizontally
python outpaint.py --input landscape.jpg --left 300 --right 300
```

### Vertical Expansion
```bash
# Extend a portrait upward
python outpaint.py --input portrait.jpg --top 400 --alignment Bottom
```

### Custom Prompt
```bash
# Add specific content with a prompt
python outpaint.py --input city.jpg --left 200 --right 200 --prompt "modern skyscrapers"
```

### Resize Before Outpainting
```bash
# Reduce input size to 50% before expanding
python outpaint.py --input large_image.jpg --ratio 16:9 --resize 50%
```

## Models Used

- **Base Model**: [SG161222/RealVisXL_V5.0_Lightning](https://huggingface.co/SG161222/RealVisXL_V5.0_Lightning)
- **ControlNet**: [xinsir/controlnet-union-sdxl-1.0](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0)
- **VAE**: [madebyollin/sdxl-vae-fp16-fix](https://huggingface.co/madebyollin/sdxl-vae-fp16-fix)

## Testing

Before running tests, activate the virtual environment and set required environment variables:

```bash
# Activate virtual environment
source .venv/bin/activate

# Set environment variables (from install.sh)
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export HF_TOKEN=your_token_here

# Run tests
python test.py
```

The test suite validates various expansion modes, alignments, and resize options.

## License

This project is licensed under the Apache License 2.0
