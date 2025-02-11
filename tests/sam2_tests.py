import os.path
from PIL import Image
import numpy as np
import unittest
import config as config
from app.services.sam2 import MesoScaleImageSegmenter


class TestMesoScaleImageSegmenter(unittest.TestCase):
    model: MesoScaleImageSegmenter = None

    def setUp(self):
        model = MesoScaleImageSegmenter()
        self.assertIsNotNone(model, "Model couldn't be loaded.")
        self.model = model

    def testAutomaticSegmentation(self):
        test_img = np.array(Image.open(os.path.join(config.Paths.meso_dir, "test_img.jpg")))
        segmentation = self.model.segment_without_prompts(test_img)
        self.assertIsNotNone(segmentation,)
        self.assertIsInstance(segmentation, list, f"Returned type is not a list: {type(segmentation)}")
        print(f"Segmentation result: List of length {len(segmentation)}")
        Image.fromarray(segmentation[0]['segmentation']).save(os.path.join(config.Paths.meso_dir, "test_mask.jpg"))



