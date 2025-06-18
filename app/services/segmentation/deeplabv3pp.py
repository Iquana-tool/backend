import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import yaml
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

from app.services.segmentation.base_model import AutomaticSegmentationBaseModel

# ---------------------------
# Parallel Attention Aggregation Block (PAAB)
# ---------------------------
class PAAB(nn.Module):
    def __init__(self, in_channels):
        super(PAAB, self).__init__()
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
        self.channel_fc1 = nn.Linear(in_channels, in_channels // 8)
        self.channel_fc2 = nn.Linear(in_channels // 8, in_channels)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))

        gap = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        gmp = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
        channel = self.channel_fc2(F.relu(self.channel_fc1(gap + gmp))).view(x.size(0), x.size(1), 1, 1)
        channel = torch.sigmoid(channel)

        return x * spatial + x * channel


# ---------------------------
# Multi-Scale Pyramid Pooling (MSPP)
# ---------------------------
class MSPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(MSPP, self).__init__()
        def sep_conv(in_c, out_c, k, d=1):
            return nn.Sequential(
                nn.Conv2d(in_c, in_c, kernel_size=k, padding=d, dilation=d, groups=in_c),
                nn.Conv2d(in_c, out_c, kernel_size=1)
            )

        self.branch1 = nn.Sequential(sep_conv(in_channels, out_channels, 5), sep_conv(out_channels, out_channels, 3))
        self.branch2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.branch3 = sep_conv(in_channels, out_channels, 3, d=4)
        self.branch4 = sep_conv(in_channels, out_channels, 3, d=8)
        self.branch5 = sep_conv(in_channels, out_channels, 3, d=12)
        self.branch6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), groups=in_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2), groups=out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )
        self.pool_avg = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_channels, out_channels, 1))
        self.pool_max = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Conv2d(in_channels, out_channels, 1))

    def forward(self, x):
        branches = [
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x),
            self.branch6(x),
            F.interpolate(self.pool_avg(x), size=x.shape[2:], mode='bilinear', align_corners=False),
            F.interpolate(self.pool_max(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        ]
        return branches


# ---------------------------
# Decoder
# ---------------------------
class Decoder(nn.Module):
    def __init__(self, in_channels, low_channels):
        super(Decoder, self).__init__()
        self.low_level = nn.Sequential(
            nn.Conv2d(low_channels, 96, kernel_size=1),
            nn.ReLU()
        )
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels + 96, 128, 1),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, x, low):
        low_feat = self.low_level(low)
        x = F.interpolate(x, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, low_feat], dim=1)
        return self.decode(x)


# ---------------------------
# Full DeepLabV3++ Model
# ---------------------------
class DeepLabV3PlusCustom(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model("tf_efficientnetv2_s", pretrained=True, features_only=True)
        enc_out = self.encoder.feature_info[-1]['num_chs']
        low_out = self.encoder.feature_info[1]['num_chs']
        self.mspp = MSPP(enc_out)
        self.paabs = nn.ModuleList([PAAB(256) for _ in range(8)])
        self.fuse = nn.Conv2d(256 * 8, 256, 1)
        self.decoder = Decoder(256, low_out)

    def forward(self, x):
        feats = self.encoder(x)
        low_feat = feats[1]
        x = feats[-1]
        mspp_outs = self.mspp(x)

        base_shape = mspp_outs[0].shape[2:]
        aligned_outs = []
        for out, paab in zip(mspp_outs, self.paabs):
            if out.shape[2:] != base_shape:
                out = F.interpolate(out, size=base_shape, mode='bilinear', align_corners=False)
            aligned_outs.append(paab(out))

        x = self.fuse(torch.cat(aligned_outs, dim=1))
        x = self.decoder(x, low_feat)
        return torch.sigmoid(F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False))


class DeepLabV3PP(AutomaticSegmentationBaseModel):
    def __init__(self, path_to_weights, path_to_config):
        super().__init__()
        with open(path_to_config, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model = DeepLabV3PlusCustom()
        self.model.load_state_dict(torch.load(path_to_weights, map_location='cpu'))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.transform = A.Compose([
            A.Resize(*self.config['input_size']),
            A.Normalize(),
            ToTensorV2()
        ])

    def process_automatic_request(self, image_np, **kwargs):
        augmented = self.transform(image=image_np)
        input_tensor = augmented['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        mask = output.squeeze().cpu().numpy()
        return (mask > 0.5).astype(np.uint8) * 255
