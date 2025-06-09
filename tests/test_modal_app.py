import unittest
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import json
import os
from pathlib import Path
from fastapi.testclient import TestClient
from modal_app import web_app, Inference

@pytest.mark.asyncio
class TestModalApp(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web_app)
        # Create a test image
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        self.test_image = self.test_dir / "test.png"
        if not self.test_image.exists():
            from PIL import Image
            img = Image.new('RGB', (512, 512), color='blue')
            img.save(self.test_image)

    def tearDown(self):
        # Clean up test data
        if self.test_image.exists():
            self.test_image.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    @patch('modal_app.Inference')
    def test_inference_endpoint_single_image(self, mock_inference):
        """Test the inference endpoint with a single image"""
        # Mock the inference result
        mock_instance = AsyncMock()
        mock_instance.run.remote.aio = AsyncMock(return_value="http://example.com/result.png")
        mock_inference.return_value = mock_instance
        
        # Test data
        test_data = {
            "input": str(self.test_image),
            "left": 100,
            "right": 100,
            "prompt": "test prompt"
        }
        
        # Make request
        response = self.client.post("/inference", json=test_data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"url": "http://example.com/result.png"})

    @patch('modal_app.Inference')
    def test_inference_endpoint_batch(self, mock_inference):
        """Test the inference endpoint with batch processing"""
        # Mock the inference result
        mock_instance = AsyncMock()
        mock_instance.run.remote.aio = AsyncMock(return_value=[
            "http://example.com/result1.png",
            "http://example.com/result2.png"
        ])
        mock_inference.return_value = mock_instance
        
        # Test data
        test_data = {
            "batch": [
                {
                    "input": str(self.test_image),
                    "left": 100,
                    "right": 100,
                    "prompt": "test prompt 1"
                },
                {
                    "input": str(self.test_image),
                    "ratio": "16:9",
                    "prompt": "test prompt 2"
                }
            ]
        }
        
        # Make request
        response = self.client.post("/inference", json=test_data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "url": [
                "http://example.com/result1.png",
                "http://example.com/result2.png"
            ]
        })

    def test_inference_endpoint_invalid_input(self):
        """Test the inference endpoint with invalid input"""
        # Test missing input and batch
        response = self.client.post("/inference", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"error": "Either input or batch must be provided"})

    @patch('modal_app.Inference')
    def test_inference_upload_url(self, mock_inference):
        """Test inference with output URL for upload"""
        # Mock the inference result
        mock_instance = AsyncMock()
        mock_instance.run.remote.aio = AsyncMock(return_value="http://example.com/uploaded.png")
        mock_inference.return_value = mock_instance
        
        # Test data with output URL
        test_data = {
            "input": str(self.test_image),
            "left": 100,
            "right": 100,
            "output_url": "http://example.com/upload"
        }
        
        # Make request
        response = self.client.post("/inference", json=test_data)
        
        # Check response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"url": "http://example.com/uploaded.png"})

class TestInferenceClass(unittest.TestCase):
    def setUp(self):
        # Create test directory
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        # Clean up test data
        if self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir)

    @patch('modal_app.load_model')
    @patch('modal_app.setup_model')
    def test_inference_initialization(self, mock_setup_model, mock_load_model):
        """Test Inference class initialization"""
        # Mock the model components
        mock_model = MagicMock()
        mock_vae = MagicMock()
        mock_pipe = MagicMock()
        mock_load_model.return_value = (mock_model, mock_vae, None)
        mock_setup_model.return_value = mock_pipe
        
        # Create Inference instance and mock its Modal-specific attributes
        inference = Inference()
        
        # Mock the Modal class methods
        inference.load_base_models = MagicMock()
        inference.setup_pipeline = MagicMock()
        inference.run = MagicMock()
        
        # Test load_base_models
        inference.load_base_models()
        inference.load_base_models.assert_called_once()
        
        # Test setup_pipeline
        inference.setup_pipeline()
        inference.setup_pipeline.assert_called_once()

    @patch('modal_app.requests')
    def test_upload_to_url(self, mock_requests):
        """Test the _upload_to_url method"""
        # Create test file
        test_file = self.test_dir / "test_upload.txt"
        test_file.write_text("test content")
        
        # Mock successful upload
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_requests.put.return_value = mock_response
        
        # Create Inference instance and mock its Modal-specific attributes
        inference = Inference()
        inference._upload_to_url = MagicMock(return_value="http://example.com/upload")
        
        # Test upload
        result = inference._upload_to_url(str(test_file), "http://example.com/upload")
        
        # Verify request was made correctly
        self.assertEqual(result, "http://example.com/upload")

if __name__ == '__main__':
    unittest.main() 