import unittest
import os
import tempfile
from pathlib import Path
from PIL import Image
import torch
import json
from outpaint import (
    outpaint_image,
    prepare_image_and_mask,
    can_expand,
    process_single_image,
    download_and_save_image,
    init_model
)

class TestOutpaint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a test image
        cls.test_dir = tempfile.mkdtemp()
        cls.test_image_path = os.path.join(cls.test_dir, "test.png")
        img = Image.new('RGB', (512, 512), color='blue')
        img.save(cls.test_image_path)
        
        # Initialize model once for all tests
        if torch.cuda.is_available():
            init_model()

    def test_can_expand(self):
        """Test the can_expand function"""
        # Test cases where expansion is allowed
        self.assertTrue(can_expand(100, 100, 200, 100, "Left"))
        self.assertTrue(can_expand(100, 100, 200, 100, "Right"))
        self.assertTrue(can_expand(100, 100, 100, 200, "Top"))
        self.assertTrue(can_expand(100, 100, 100, 200, "Bottom"))
        self.assertTrue(can_expand(100, 100, 200, 200, "Middle"))

        # Test cases where expansion is not allowed
        self.assertFalse(can_expand(200, 100, 100, 100, "Left"))
        self.assertFalse(can_expand(200, 100, 100, 100, "Right"))
        self.assertFalse(can_expand(100, 200, 100, 100, "Top"))
        self.assertFalse(can_expand(100, 200, 100, 100, "Bottom"))

    def test_prepare_image_and_mask(self):
        """Test image and mask preparation"""
        # Load test image
        image = Image.open(self.test_image_path)
        
        # Test basic preparation
        width = 612  # Original + 100 pixels
        height = 512  # Original height
        overlap = 10
        background, mask = prepare_image_and_mask(
            image, width, height, overlap, "Full", 50, "Middle",
            True, True, False, False
        )
        
        # Check dimensions
        self.assertEqual(background.size, (width, height))
        self.assertEqual(mask.size, (width, height))
        
        # Check image mode
        self.assertEqual(background.mode, "RGB")
        self.assertEqual(mask.mode, "L")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_outpaint_image(self):
        """Test the outpaint_image function"""
        # Test basic outpainting
        result = outpaint_image(
            self.test_image_path,
            left=50,
            right=50,
            top=20,
            bottom=20,
            prompt="test prompt",
            steps=1  # Use 1 step for faster testing
        )
        
        # Check that result is a PIL Image
        self.assertIsInstance(result, Image.Image)
        
        # Check dimensions
        self.assertEqual(result.size[0], 512 + 100)  # Original width + 50 + 50
        self.assertEqual(result.size[1], 512 + 40)  # Original height + 20 + 20

    def test_download_and_save_image(self):
        """Test image downloading function"""
        # Test with invalid URL
        with self.assertRaises(ValueError):
            download_and_save_image("https://invalid.url/image.png")
        
        # Test with valid image URL
        test_url = "https://raw.githubusercontent.com/huggingface/diffusers/main/docs/source/imgs/diffusers_library.jpg"
        try:
            result = download_and_save_image(test_url)
            self.assertTrue(os.path.exists(result))
            self.assertTrue(result.endswith('.png'))
        except ValueError:
            self.skipTest("Network error or URL no longer valid")

    def test_process_single_image(self):
        """Test processing a single image"""
        # Create test config
        config = {
            'input': self.test_image_path,
            'output': os.path.join(self.test_dir, 'output.png'),
            'left': 50,
            'right': 50,
            'top': 0,
            'bottom': 0,
            'prompt': 'test prompt',
            'steps': 1  # Use 1 step for faster testing
        }
        
        # Process image
        if torch.cuda.is_available():
            output_path = process_single_image(config)
            
            # Check that output file exists
            self.assertTrue(os.path.exists(output_path))
            
            # Check output dimensions
            output_img = Image.open(output_path)
            self.assertEqual(output_img.size[0], 512 + 100)  # Original width + 100
            self.assertEqual(output_img.size[1], 512)  # Original height
        else:
            self.skipTest("CUDA not available")

    def test_invalid_inputs(self):
        """Test error handling for invalid inputs"""
        # Test invalid file path
        with self.assertRaises(FileNotFoundError):
            process_single_image({'input': 'nonexistent.png'})
        
        # Test invalid expansion values
        with self.assertRaises(ValueError):
            process_single_image({
                'input': self.test_image_path,
                'left': -100  # Negative value
            })
        
        # Test no expansion specified
        with self.assertRaises(ValueError):
            process_single_image({
                'input': self.test_image_path,
                'left': 0, 'right': 0, 'top': 0, 'bottom': 0
            })

if __name__ == '__main__':
    unittest.main() 