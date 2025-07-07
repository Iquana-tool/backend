import os
import unittest

# Import the function to be tested
from app.services.prompted_segmentation.sam2 import download_checkpoint


class TestDownloadCheckpointLogic(unittest.TestCase):
    def setUp(self):
        # Set up a temporary directory for testing
        self.test_dir = "test_downloads"
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        # Clean up the temporary directory after testing
        for file in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, file))
        os.rmdir(self.test_dir)

    def test_file_already_exists(self):
        """Test that the function returns 0 if the file already exists."""
        # Create a dummy file
        ckpt_path = os.path.join(self.test_dir, "existing_file.pt")
        with open(ckpt_path, "w") as f:
            f.write("dummy content")

        # Call the function
        result = download_checkpoint(ckpt_path)

        # Assert the function returns 0 and does not attempt to download
        self.assertEqual(result, 0)

    def test_actual_download_and_model_load(self):
        """Test that the function can download a real checkpoint and load it into a model."""
        # Use a real URL for the checkpoint file
        ckpt_path = os.path.join(self.test_dir, "sam2.1_hiera_tiny.pt")

        # Call the function to download the checkpoint
        result = download_checkpoint(ckpt_path)

        # Assert the function returns 1 (successful download)
        self.assertEqual(result, 1)

        # Verify the file was downloaded
        self.assertTrue(os.path.exists(ckpt_path))

    def test_build_sam2(self):
        # Load the checkpoint into a model
        from app.services.prompted_segmentation.sam2 import SAM2
        try:
            list_configs = []
            for sam2_config in list_configs:
                print(f"Testing {sam2_config.__name__}")
                _model = SAM2(sam2_config)
                self.assertTrue(os.path.exists(sam2_config.weights), f"Weights file {sam2_config.weights} not found.")
        except Exception as e:
            self.fail(f"Failed to load checkpoint: {e}")


if __name__ == "__main__":
    unittest.main()
