"""
PyTorch Lightning module for FreqNet-based deepfake detection.
"""
from __future__ import annotations

import torch
from torch import nn
import pytorch_lightning as pl

from models import FreqNet


class DeepfakeLitModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        enable_channel_hfrf_dct: bool = True,
        hfri_low_freq_ratio: float = 0.125,
        hfrf_low_freq_ratio: float = 0.125,
        hfrf_channel_low_freq_ratio: float = 0.15,
        dct_fcl_kernel_size: int = 3,
        dct_fcl_activation: str = "gelu",
        base_channels: int = 32,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = FreqNet(
            base_channels=base_channels,
            hfri_low_freq_ratio=hfri_low_freq_ratio,
            hfrf_low_freq_ratio=hfrf_low_freq_ratio,
            enable_channel_hfrf_dct=enable_channel_hfrf_dct,
            hfrf_channel_low_freq_ratio=hfrf_channel_low_freq_ratio,
            dct_fcl_kernel_size=dct_fcl_kernel_size,
            dct_fcl_activation=dct_fcl_activation,
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits.squeeze(1)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, labels = batch
        labels = labels.float()

        logits = self(images)
        loss = self.loss_fn(logits, labels)

        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == labels).float().mean()

        self.log(f"{stage}_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
