"""
PyTorch Lightning module for FreqNet-based deepfake detection.
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torchvision.utils import make_grid, save_image

from models import FreqNet


class DeepfakeLitModule(pl.LightningModule):
    def __init__(
        self,
        lr: float = 5e-5,
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
        label_smoothing_real: float = 0.9,
        topk_k: int = 25,
        diagnostics_dir: str = "diagnostics",
        norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
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
        self._topk_samples: list[dict] = []
        self._val_logits_real: list[torch.Tensor] = []
        self._val_logits_fake: list[torch.Tensor] = []

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
        if len(batch) == 3:
            images, labels, paths = batch
        else:
            images, labels = batch
            paths = None

        raw_labels = labels.float()
        labels_for_loss = raw_labels

        if stage == "train" and self.hparams.label_smoothing_real is not None:
            smooth_value = float(self.hparams.label_smoothing_real)
            labels_for_loss = torch.where(
                raw_labels > 0.5,
                torch.full_like(raw_labels, smooth_value),
                raw_labels,
            )

        logits = self(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels_for_loss, pos_weight=self.pos_weight)

        preds = (torch.sigmoid(logits) > 0.5).float()
        acc = (preds == raw_labels).float().mean()

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
        if stage == "val":
            per_sample_losses = F.binary_cross_entropy_with_logits(
                logits,
                labels_for_loss,
                pos_weight=self.pos_weight,
                reduction="none",
            )
            self._accumulate_logits(raw_labels, logits)
            self._accumulate_topk(images, raw_labels, logits, per_sample_losses, paths)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def on_validation_epoch_start(self) -> None:
        self._topk_samples = []
        self._val_logits_real = []
        self._val_logits_fake = []

    def on_validation_epoch_end(self) -> None:
        self._log_logits_histograms()
        self._export_topk_diagnostics()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def _accumulate_logits(self, labels: torch.Tensor, logits: torch.Tensor) -> None:
        labels = labels.detach().to("cpu")
        logits = logits.detach().to("cpu")
        real_mask = labels > 0.5
        fake_mask = ~real_mask
        if real_mask.any():
            self._val_logits_real.append(logits[real_mask])
        if fake_mask.any():
            self._val_logits_fake.append(logits[fake_mask])

    def _accumulate_topk(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
        losses: torch.Tensor,
        paths,
    ) -> None:
        images_cpu = images.detach().to("cpu")
        labels_cpu = labels.detach().to("cpu")
        logits_cpu = logits.detach().to("cpu")
        losses_cpu = losses.detach().to("cpu")

        for idx in range(losses_cpu.shape[0]):
            path_value = None
            if paths is not None:
                path_value = paths[idx]
            if path_value is None:
                path_value = f"val://{idx}"
            sample = {
                "loss": float(losses_cpu[idx].item()),
                "label": int(labels_cpu[idx].item()),
                "logit": float(logits_cpu[idx].item()),
                "path": str(path_value),
                "image": images_cpu[idx],
            }
            self._topk_samples.append(sample)

        self._topk_samples.sort(key=lambda item: item["loss"], reverse=True)
        self._topk_samples = self._topk_samples[: int(self.hparams.topk_k)]

    def _log_logits_histograms(self) -> None:
        if not self.logger:
            return
        if not hasattr(self.logger, "experiment"):
            return

        if self._val_logits_real:
            real_logits = torch.cat(self._val_logits_real, dim=0)
            self.logger.experiment.add_histogram(
                "logits/real",
                real_logits,
                global_step=self.current_epoch,
            )
        if self._val_logits_fake:
            fake_logits = torch.cat(self._val_logits_fake, dim=0)
            self.logger.experiment.add_histogram(
                "logits/fake",
                fake_logits,
                global_step=self.current_epoch,
            )

    def _export_topk_diagnostics(self) -> None:
        if not self._topk_samples:
            return

        topk = sorted(self._topk_samples, key=lambda item: item["loss"], reverse=True)
        epoch = int(self.current_epoch)

        diagnostics_dirs = [Path(self.hparams.diagnostics_dir)]
        if self.logger and hasattr(self.logger, "log_dir") and self.logger.log_dir:
            diagnostics_dirs.append(Path(self.logger.log_dir) / "diagnostics")

        for diagnostics_dir in diagnostics_dirs:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            csv_path = diagnostics_dir / f"top_loss_epoch_{epoch:03d}.csv"
            with csv_path.open("w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["rank", "loss", "label", "logit", "path", "notes"])
                for idx, sample in enumerate(topk, start=1):
                    writer.writerow(
                        [
                            idx,
                            f"{sample['loss']:.6f}",
                            sample["label"],
                            f"{sample['logit']:.6f}",
                            sample["path"],
                            "",
                        ]
                    )

            images = torch.stack([sample["image"] for sample in topk], dim=0)
            images = self._denormalize(images).clamp(0.0, 1.0)
            grid = make_grid(images, nrow=5)
            grid_path = diagnostics_dir / f"top_loss_epoch_{epoch:03d}.png"
            save_image(grid, grid_path)

    def _denormalize(self, images: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.hparams.norm_mean, device=images.device).view(1, 3, 1, 1)
        std = torch.tensor(self.hparams.norm_std, device=images.device).view(1, 3, 1, 1)
        return images * std + mean

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
