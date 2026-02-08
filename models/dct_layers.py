"""
DCT-based frequency modules for FreqNet-inspired deepfake detection.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F


_DCT_CACHE: Dict[Tuple[int, str, int | None, torch.dtype], torch.Tensor] = {}


def _get_dct_matrix(size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (size, device.type, device.index, dtype)
    cached = _DCT_CACHE.get(key)
    if cached is not None:
        return cached

    n = torch.arange(size, device=device, dtype=dtype).unsqueeze(0)
    k = torch.arange(size, device=device, dtype=dtype).unsqueeze(1)
    mat = torch.cos(math.pi * (2 * n + 1) * k / (2 * size))

    scale = torch.full((size,), math.sqrt(2.0 / size), device=device, dtype=dtype)
    scale[0] = math.sqrt(1.0 / size)
    mat = scale.unsqueeze(1) * mat

    _DCT_CACHE[key] = mat
    return mat


def _dct_2d(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x_ = x.reshape(b * c, h, w)

    c_h = _get_dct_matrix(h, x.device, x.dtype)
    c_w = _get_dct_matrix(w, x.device, x.dtype)

    x_ = torch.einsum("hn,bnw->bhw", c_h, x_)
    x_ = torch.einsum("wm,bhm->bhw", c_w, x_)
    return x_.reshape(b, c, h, w)


def _idct_2d(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    x_ = x.reshape(b * c, h, w)

    c_h = _get_dct_matrix(h, x.device, x.dtype).transpose(0, 1)
    c_w = _get_dct_matrix(w, x.device, x.dtype).transpose(0, 1)

    x_ = torch.einsum("hn,bnw->bhw", c_h, x_)
    x_ = torch.einsum("wm,bhm->bhw", c_w, x_)
    return x_.reshape(b, c, h, w)


def _dct_1d_channels(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    c_mat = _get_dct_matrix(c, x.device, x.dtype)
    x_ = x.permute(0, 2, 3, 1)
    x_ = torch.einsum("kc,bhwc->bhwk", c_mat, x_)
    return x_.permute(0, 3, 1, 2).contiguous()


def _idct_1d_channels(x: torch.Tensor) -> torch.Tensor:
    b, c, h, w = x.shape
    c_mat = _get_dct_matrix(c, x.device, x.dtype).transpose(0, 1)
    x_ = x.permute(0, 2, 3, 1)
    x_ = torch.einsum("kc,bhwc->bhwk", c_mat, x_)
    return x_.permute(0, 3, 1, 2).contiguous()


def _suppress_low_freq_block(x: torch.Tensor, low_h: int, low_w: int) -> torch.Tensor:
    if low_h <= 0 or low_w <= 0:
        return x
    x = x.clone()
    x[..., :low_h, :low_w] = 0
    return x


def _suppress_low_freq_channels(x: torch.Tensor, low_c: int) -> torch.Tensor:
    if low_c <= 0:
        return x
    x = x.clone()
    x[:, :low_c, ...] = 0
    return x


class HFRI_DCT(nn.Module):
    """
    High-Frequency Representation of Image (HFRI-DCT).
    """

    def __init__(self, low_freq_ratio: float = 0.125, clamp_range: Tuple[float, float] | None = None):
        super().__init__()
        self.low_freq_ratio = low_freq_ratio
        self.clamp_range = clamp_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_freq = _dct_2d(x)
        low_h = int(round(h * self.low_freq_ratio))
        low_w = int(round(w * self.low_freq_ratio))
        x_freq = _suppress_low_freq_block(x_freq, low_h, low_w)
        x_spatial = _idct_2d(x_freq)
        if self.clamp_range is not None:
            x_spatial = torch.clamp(x_spatial, self.clamp_range[0], self.clamp_range[1])
        return x_spatial


class HFRF_DCT(nn.Module):
    """
    High-Frequency Representation of Feature (HFRF-DCT).
    """

    def __init__(
        self,
        low_freq_ratio: float = 0.125,
        enable_channel: bool = True,
        channel_low_freq_ratio: float = 0.15,
    ):
        super().__init__()
        self.low_freq_ratio = low_freq_ratio
        self.enable_channel = enable_channel
        self.channel_low_freq_ratio = channel_low_freq_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        x_freq = _dct_2d(x)
        low_h = int(round(h * self.low_freq_ratio))
        low_w = int(round(w * self.low_freq_ratio))
        x_freq = _suppress_low_freq_block(x_freq, low_h, low_w)
        x_spatial = _idct_2d(x_freq)

        if self.enable_channel:
            c_low = int(round(c * self.channel_low_freq_ratio))
            x_cfreq = _dct_1d_channels(x_spatial)
            x_cfreq = _suppress_low_freq_channels(x_cfreq, c_low)
            x_channel = _idct_1d_channels(x_cfreq)
            x_hf = x_spatial + x_channel
        else:
            x_hf = x_spatial

        return x + x_hf


class DCT_FCL(nn.Module):
    """
    Frequency Convolution Layer (DCT-FCL).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        activation: str = "gelu",
        padding: str = "reflect",
    ):
        super().__init__()

        if kernel_size not in (3, 5):
            raise ValueError("kernel_size must be 3 or 5")
        if padding not in ("reflect", "valid"):
            raise ValueError("padding must be 'reflect' or 'valid'")

        self.padding_mode = padding
        pad = kernel_size // 2
        if padding == "reflect":
            self.pad = nn.ReflectionPad2d(pad)
            conv_padding = 0
        else:
            self.pad = nn.Identity()
            conv_padding = 0

        self.conv = nn.Conv2d(channels, channels, kernel_size, padding=conv_padding, bias=True)
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU(inplace=True)
        else:
            raise ValueError("activation must be 'relu' or 'gelu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_freq = _dct_2d(x)
        x_freq = self.pad(x_freq)
        x_freq = self.conv(x_freq)
        x_freq = self.act(x_freq)
        x_spatial = _idct_2d(x_freq)

        if x_spatial.shape != x.shape:
            diff_h = x.shape[-2] - x_spatial.shape[-2]
            diff_w = x.shape[-1] - x_spatial.shape[-1]
            pad_left = diff_w // 2
            pad_right = diff_w - pad_left
            pad_top = diff_h // 2
            pad_bottom = diff_h - pad_top
            x_spatial = F.pad(x_spatial, (pad_left, pad_right, pad_top, pad_bottom))

        return x + x_spatial
