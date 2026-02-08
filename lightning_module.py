"""
PyTorch Lightning module for FreqNet-based deepfake detection.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from models import FreqNet


class DeepfakeLitModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight: float | None = None,
        compile_model: bool = False,
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
        initial_pos_weight = 1.0 if pos_weight is None else float(pos_weight)
        self.register_buffer("pos_weight", torch.tensor(initial_pos_weight, dtype=torch.float32))
        self._compiled = False

    def on_fit_start(self) -> None:
        if self.hparams.compile_model and not self._compiled:
            try:
                self.model = torch.compile(self.model)
                self._compiled = True
            except Exception as exc:
                print(f"Warning: torch.compile failed: {exc}")

        if self.hparams.pos_weight is not None:
            return

        dm = self.trainer.datamodule
        train_dataset = getattr(dm, "train_dataset", None)
        if train_dataset is None:
            return

        pos_count, neg_count = self._count_labels(train_dataset)
        if pos_count is None or neg_count is None or pos_count == 0:
            return

        self.pos_weight = torch.tensor(neg_count / pos_count, device=self.device, dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits.squeeze(1)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        images, labels = batch
        labels = labels.float()

        logits = self(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight)

        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == labels).float().mean()

        self.log(
            f"{stage}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            f"{stage}_acc",
            acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
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

    def _count_labels(self, dataset):
        pos = 0
        neg = 0

        if hasattr(dataset, "samples"):
            for _, label in dataset.samples:
                if int(label) == 1:
                    pos += 1
                else:
                    neg += 1
            return pos, neg

        if hasattr(dataset, "hf_dataset"):
            hf_dataset = dataset.hf_dataset
            label_map = getattr(dataset, "label_map", None) or {0: 1, 1: 0, "real": 1, "fake": 0}

            try:
                labels = hf_dataset["label"]
            except Exception:
                labels = [item["label"] for item in hf_dataset]

            for label in labels:
                if isinstance(label, str):
                    mapped = label_map.get(label)
                else:
                    mapped = label_map.get(int(label), int(label))
                if mapped == 1:
                    pos += 1
                else:
                    neg += 1
            return pos, neg

        return None, None
