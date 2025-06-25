import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from app.services.segmentation.base_model import AutomaticSegmentationBaseModel
from app.schemas.segmentation.segmentations import AutomaticSegmentationRequest

# ---------------------------
# Basic Conv Block used in all decoder levels
# ---------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.block(x)

# ---------------------------
# U-Net++ Architecture 
# ---------------------------
class UNetPlusPlus(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, features=[64, 128, 256, 512]):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.upsample = lambda x: F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = ConvBlock(in_ch, features[0])
        self.conv1_0 = ConvBlock(features[0], features[1])
        self.conv2_0 = ConvBlock(features[1], features[2])
        self.conv3_0 = ConvBlock(features[2], features[3])

        self.conv0_1 = ConvBlock(features[0] + features[1], features[0])
        self.conv1_1 = ConvBlock(features[1] + features[2], features[1])
        self.conv2_1 = ConvBlock(features[2] + features[3], features[2])

        self.conv0_2 = ConvBlock(features[0]*2 + features[1], features[0])
        self.conv1_2 = ConvBlock(features[1]*2 + features[2], features[1])

        self.conv0_3 = ConvBlock(features[0]*3 + features[1], features[0])

        self.final = nn.Conv2d(features[0], out_ch, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample(x1_0)], dim=1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample(x2_0)], dim=1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.upsample(x1_1)], dim=1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample(x3_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.upsample(x2_1)], dim=1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2)], dim=1))

        return torch.sigmoid(self.final(x0_3))

# ---------------------------
# U-Net++ Wrapper for Inference
# ---------------------------
class UnetPlusPlus(AutomaticSegmentationBaseModel):
    def __init__(self, path_to_weights, path_to_config):
        super().__init__()
        with open(path_to_config, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = UNetPlusPlus(
            in_ch=self.config['in_channels'],
            out_ch=self.config['out_channels'],
            features=self.config.get('features', [64, 128, 256, 512])
        )
        self.model.load_state_dict(torch.load(path_to_weights, map_location='cpu'))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.transform = A.Compose([
            A.Resize(*self.config['input_size']),
            A.Normalize(),
            ToTensorV2()
        ])

    def process_automatic_request(self, request: AutomaticSegmentationRequest):
        image_np = request.image

        # Preprocess
        augmented = self.transform(image=image_np)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)

        # Postprocess
        mask = output.squeeze().cpu().numpy()
        return (mask > 0.5).astype(np.uint8) * 255
