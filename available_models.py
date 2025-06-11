from services.segmentation.sam2 import SAM2Prompted, SAM2Automatic, SAM2Prompted3D

AvailableModels = {
    "prompted": {
        'Mockup': 'MockupSegmentationModel',
        'SAM2Prompted':  SAM2Prompted,
    },
    "automatic": {
        'SAM2Automatic': SAM2Automatic,
    },
    "prompted_3d": {
        'SAM2Prompted3D': SAM2Prompted3D,
    },
    "automatic_3d": {
    }
}
