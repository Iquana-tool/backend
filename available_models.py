from app.services.segmentation.sam2 import SAM2Prompted, SAM2Automatic, SAM2Prompted3D
from app.services.segmentation.unet import Unet
from app.services.segmentation.unetplusplus import UnetPlusPlus
from app.services.segmentation.deeplabv3pp import DeepLabV3PP

AvailableModels = {
    "prompted": {
        'Mockup': 'MockupSegmentationModel',
        'SAM2Prompted': SAM2Prompted,
    },
    "automatic": {
        'SAM2Automatic': SAM2Automatic,
        'Unet': Unet,
        'UnetPlusPlus': UnetPlusPlus,
        'DeepLabV3PP': DeepLabV3PP,
    },
    "prompted_3d": {
        'SAM2Prompted3D': SAM2Prompted3D,
    },
    "automatic_3d": {
    }
}
