import torch
import numpy as np

class DummyImageEncoder:
    def __init__(self):
        """Initialize the dummy encoder."""
        print("🟢 Dummy Image Encoder Initialized.")

    def encode(self, image_tensor):
        """
        Generates a random tensor as the image embedding.
        Args:
            image_tensor (torch.Tensor): Image tensor (not used in dummy encoder).
        Returns:
            np.ndarray: Random embedding of shape (1, 256, 64, 64).
        """
        # Generate a random tensor of shape (1, 256, 64, 64)
        embedding = torch.rand((1, 256, 64, 64)).numpy()
        return embedding
