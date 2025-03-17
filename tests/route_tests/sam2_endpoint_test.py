import json
import unittest
import numpy as np
from io import BytesIO
from PIL import Image
from app import create_app


class SAM2EndpointTestCase(unittest.TestCase):
    """Test cases for the segmentation endpoint."""

    @classmethod
    def setUpClass(cls):
        """Set up test client before running tests."""
        app = create_app()  # Initialize FastAPI app
        app.config["TESTING"] = True
        cls.client = app.test_client()

    def generate_dummy_image(self):
        """Generate a black image (100x100) in memory for testing."""
        image = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8), mode="RGB")
        img_io = BytesIO()
        image.save(img_io, format="PNG")
        img_io.seek(0)
        return img_io

    def test_segment_without_prompts(self):
        """Test segmentation without any prompts."""
        response = self.client.post(
            "/segment",
            content_type="multipart/form-data",
            data={"image": (self.generate_dummy_image(), "test.png"), "use_prompts": "false"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("segmentation_result", data)  # Ensure segmentation output exists

    def test_segment_with_valid_prompts(self):
        """Test segmentation with valid prompts."""
        request_data = {
            "use_prompts": True,
            "point_prompts": [{"x": 0.5, "y": 0.5, "label": 1}],
            "box_prompts": [{"min_x": 0.1, "min_y": 0.1, "max_x": 0.9, "max_y": 0.9}],
        }

        response = self.client.post(
            "/segment",
            content_type="multipart/form-data",
            data={
                "image": (self.generate_dummy_image(), "test.png"),
                "json": json.dumps(request_data),
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("masks", data)
        self.assertIn("quality", data)  # Ensure segmentation output is present

    def test_segment_with_invalid_prompts(self):
        """Test segmentation with invalid prompts (out of bounds)."""
        invalid_data = {
            "use_prompts": True,
            "point_prompts": [{"x": 1.5, "y": -0.2, "label": 1}],  # Invalid coordinates
            "box_prompts": [{"min_x": 0.1, "min_y": 0.1, "max_x": 2.0, "max_y": 0.9}],  # Out of range
        }

        response = self.client.post(
            "/segment",
            content_type="multipart/form-data",
            data={
                "image": (self.generate_dummy_image(), "test.png"),
                "json": json.dumps(invalid_data),
            },
        )

        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn("error", data)  # Ensure validation error is returned


if __name__ == "__main__":
    unittest.main()
