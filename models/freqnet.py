"""
Lightweight ResNet-style backbone with DCT frequency modules.
"""
from __future__ import annotations

from torch import nn

from .dct_layers import HFRI_DCT, HFRF_DCT, DCT_FCL


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.downsample = (
            nn.Identity()
            if (in_ch == out_ch and stride == 1)
            else nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=True)
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.act(self.conv1(x))
        out = self.conv2(out)
        out = out + identity
        out = self.act(out)
        return out


class FreqNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        num_classes: int = 1,
        hfri_low_freq_ratio: float = 0.125,
        hfrf_low_freq_ratio: float = 0.125,
        enable_channel_hfrf_dct: bool = True,
        hfrf_channel_low_freq_ratio: float = 0.15,
        dct_fcl_kernel_size: int = 3,
        dct_fcl_activation: str = "gelu",
    ):
        super().__init__()

        self.hfri = HFRI_DCT(low_freq_ratio=hfri_low_freq_ratio)

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

        self.block1 = BasicBlock(base_channels, base_channels, stride=1)
        self.hfrf1 = HFRF_DCT(
            low_freq_ratio=hfrf_low_freq_ratio,
            enable_channel=enable_channel_hfrf_dct,
            channel_low_freq_ratio=hfrf_channel_low_freq_ratio,
        )

        self.block2 = BasicBlock(base_channels, base_channels * 2, stride=2)
        self.fcl1 = DCT_FCL(base_channels * 2, kernel_size=dct_fcl_kernel_size, activation=dct_fcl_activation)

        self.block3 = BasicBlock(base_channels * 2, base_channels * 4, stride=2)
        self.hfrf2 = HFRF_DCT(
            low_freq_ratio=hfrf_low_freq_ratio,
            enable_channel=enable_channel_hfrf_dct,
            channel_low_freq_ratio=hfrf_channel_low_freq_ratio,
        )

        self.block4 = BasicBlock(base_channels * 4, base_channels * 4, stride=1)
        self.fcl2 = DCT_FCL(base_channels * 4, kernel_size=dct_fcl_kernel_size, activation=dct_fcl_activation)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base_channels * 4, num_classes)

    def forward(self, x):
        x = self.hfri(x)
        x = self.stem(x)

        x = self.block1(x)
        x = self.hfrf1(x)

        x = self.block2(x)
        x = self.fcl1(x)

        x = self.block3(x)
        x = self.hfrf2(x)

        x = self.block4(x)
        x = self.fcl2(x)

        x = self.pool(x).flatten(1)
        logits = self.fc(x)
        return logits
