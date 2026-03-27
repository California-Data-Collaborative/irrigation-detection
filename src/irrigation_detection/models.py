"""
Model definitions for irrigation detection.

DilatedUNet1D: U-Net with dilated convolutions for hourly irrigation extraction.
  - Input:  (batch, 1, 336) — 14 days of hourly total consumption
  - Output: (batch, 1, 336) — predicted hourly irrigation volume
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

WINDOW_SIZE = 336  # 14 days × 24 hours


class DilatedBlock(nn.Module):
    """Two-layer dilated convolution block with batch norm and ReLU."""

    def __init__(self, in_c: int, out_c: int, dilation: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DilatedUNet1D(nn.Module):
    """
    1D U-Net with dilated bottleneck for irrigation signal extraction.

    Architecture:
        Encoder:    4 blocks (1→16→32→64→128) with max pooling
        Bottleneck: dilated convolutions (d=2, d=4) for long-range context
        Decoder:    4 blocks with skip connections and upsampling
        Output:     ReLU activation (irrigation >= 0)
    """

    def __init__(self):
        super().__init__()

        # Encoder
        self.enc1 = DilatedBlock(1, 16, dilation=1)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = DilatedBlock(16, 32, dilation=1)
        self.pool2 = nn.MaxPool1d(2)
        self.enc3 = DilatedBlock(32, 64, dilation=1)
        self.pool3 = nn.MaxPool1d(2)
        self.enc4 = DilatedBlock(64, 128, dilation=1)
        self.pool4 = nn.MaxPool1d(2)

        # Bottleneck (dilated for long-range context)
        self.bottleneck = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.dec4 = DilatedBlock(256 + 128, 128)
        self.up3 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.dec3 = DilatedBlock(128 + 64, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.dec2 = DilatedBlock(64 + 32, 32)
        self.up1 = nn.Upsample(scale_factor=2, mode="linear", align_corners=True)
        self.dec1 = DilatedBlock(32 + 16, 16)

        self.final = nn.Conv1d(16, 1, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Bottleneck
        b = self.bottleneck(p4)

        # Decode with skip connections
        d4 = self.up4(b)
        if d4.shape[2] != e4.shape[2]:
            d4 = d4[:, :, : e4.shape[2]]
        d4 = self.dec4(torch.cat((d4, e4), dim=1))

        d3 = self.up3(d4)
        if d3.shape[2] != e3.shape[2]:
            d3 = d3[:, :, : e3.shape[2]]
        d3 = self.dec3(torch.cat((d3, e3), dim=1))

        d2 = self.up2(d3)
        if d2.shape[2] != e2.shape[2]:
            d2 = d2[:, :, : e2.shape[2]]
        d2 = self.dec2(torch.cat((d2, e2), dim=1))

        d1 = self.up1(d2)
        if d1.shape[2] != e1.shape[2]:
            d1 = d1[:, :, : e1.shape[2]]
        d1 = self.dec1(torch.cat((d1, e1), dim=1))

        return self.relu(self.final(d1))


def load_model(model_path: str, device: str = "cpu") -> DilatedUNet1D:
    """Load a pre-trained DilatedUNet1D from disk."""
    model = DilatedUNet1D().to(device)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded DilatedUNet1D from {model_path} (device={device})")
    return model
