import os.path
from PIL import Image
import numpy as np
import unittest
import config as config
from app.services.sam2 import MesoScaleImageSegmenter
from app.services.prompts import Prompts


class TestMesoScaleImageSegmenter(unittest.TestCase):
    model: MesoScaleImageSegmenter = None

    def setUp(self):
        model = MesoScaleImageSegmenter()
        self.assertIsNotNone(model, "Model couldn't be loaded.")
        self.model = model

    def testAutomaticSegmentation(self):
        test_img = np.array(Image.open(os.path.join(config.Paths.meso_dir, "test_img.jpg")))
        segmentation = self.model.segment_without_prompts(test_img)
        self.assertIsNotNone(segmentation, )
        self.assertIsInstance(segmentation, list, f"Returned type is not a list: {type(segmentation)}")
        print(f"Segmentation result: List of length {len(segmentation)}")
        Image.fromarray(segmentation[0]['segmentation']).save(os.path.join(config.Paths.meso_dir, "test_mask.jpg"))

    def testPointPromptedSegmentation(self):
        test_img = np.array(Image.open(os.path.join(config.Paths.meso_dir, "test_img.jpg")))
        prompts = Prompts()

        prompts.add_point_annotation(0.5, 0.5, 1)
        segmentation = self.model.segment_prompts(test_img, prompts)
        self.assertIsNotNone(segmentation, )
        self.assertIsInstance(segmentation, tuple, f"Returned type is not a tuple: {type(segmentation)}")

        prompts.add_point_annotation(0., 0., 0)
        segmentation = self.model.segment_prompts(test_img, prompts)
        self.assertIsNotNone(segmentation, )
        self.assertIsInstance(segmentation, tuple, f"Returned type is not a tuple: {type(segmentation)}")

    def testBoxPromptedSegmentation(self):
        test_img = np.array(Image.open(os.path.join(config.Paths.meso_dir, "test_img.jpg")))
        prompts = Prompts()
        prompts.add_box_annotation(0.5, 0.5, 0.7, 0.7)
        segmentation = self.model.segment_prompts(test_img, prompts)
        self.assertIsNotNone(segmentation, )
        self.assertIsInstance(segmentation, tuple, f"Returned type is not a tuple: {type(segmentation)}")

