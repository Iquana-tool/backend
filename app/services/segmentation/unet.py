import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import albumentations as A
from albumentations.pytorch import ToTensorV2
import yaml
import numpy as np

from app.services.segmentation.base_model import AutomaticSegmentationBaseModel
from app.schemas.segmentation.segmentations import AutomaticSegmentationRequest

# ---------------------------
# U-Net Architecture 
# ---------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = conv_block(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = conv_block(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.dec4 = conv_block(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec3 = conv_block(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec2 = conv_block(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec1 = conv_block(128, 64)

        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        bn = self.bottleneck(self.pool4(e4))

        d4 = self.dec4(torch.cat([self.up4(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return torch.sigmoid(self.final(d1))

# ---------------------------
# U-Net Wrapper for Inference
# ---------------------------
class Unet(AutomaticSegmentationBaseModel):
    def __init__(self, path_to_weights, path_to_config):
        super().__init__()
        with open(path_to_config, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = UNet(in_channels=self.config['in_channels'], out_channels=self.config['out_channels'])
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

        # Preprocessing
        augmented = self.transform(image=image_np)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)

        # Prediction
        with torch.no_grad():
            output = self.model(input_tensor)

        # Postprocessing
        mask = output.squeeze().cpu().numpy()
        return (mask > 0.5).astype(np.uint8) * 255
