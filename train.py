"""
Training entrypoint for FreqNet-based deepfake detection.
"""
from __future__ import annotations

import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar

from data.datamodule import DeepfakeDataModule
from lightning_module import DeepfakeLitModule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FreqNet with DCT modules")
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--data-dir", type=str, help="Path to local dataset root")
    data_group.add_argument("--hf-dataset-id", type=str, help="HuggingFace dataset ID")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=None)
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--disable-channel-hfrf-dct", action="store_true")
    parser.add_argument("--hfri-low-freq-ratio", type=float, default=0.125)
    parser.add_argument("--hfrf-low-freq-ratio", type=float, default=0.125)
    parser.add_argument("--hfrf-channel-low-freq-ratio", type=float, default=0.15)
    parser.add_argument("--dct-fcl-kernel-size", type=int, default=3)
    parser.add_argument("--dct-fcl-activation", type=str, default="gelu")
    parser.add_argument("--accelerator", type=str, default="auto")
    parser.add_argument("--devices", type=str, default="auto")
    parser.add_argument("--pbar-refresh-rate", type=int, default=1)
    parser.add_argument(
        "--hf-label-order",
        type=str,
        choices=["real-fake", "fake-real"],
        default=None,
        help="Explicitly set HF label order if no ClassLabel names exist",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(42, workers=True)

    hf_label_map = None
    if args.hf_label_order == "real-fake":
        hf_label_map = {0: 1, 1: 0, "real": 1, "fake": 0}
    elif args.hf_label_order == "fake-real":
        hf_label_map = {0: 0, 1: 1, "fake": 0, "real": 1}

    datamodule = DeepfakeDataModule(
        data_dir=args.data_dir,
        hf_dataset_id=args.hf_dataset_id,
        hf_label_map=hf_label_map,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = DeepfakeLitModule(
        lr=args.lr,
        weight_decay=args.weight_decay,
        pos_weight=args.pos_weight,
        compile_model=args.compile_model,
        enable_channel_hfrf_dct=not args.disable_channel_hfrf_dct,
        hfri_low_freq_ratio=args.hfri_low_freq_ratio,
        hfrf_low_freq_ratio=args.hfrf_low_freq_ratio,
        hfrf_channel_low_freq_ratio=args.hfrf_channel_low_freq_ratio,
        dct_fcl_kernel_size=args.dct_fcl_kernel_size,
        dct_fcl_activation=args.dct_fcl_activation,
        base_channels=args.base_channels,
    )

    callbacks = [
        ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="freqnet-{epoch}-{val_loss:.4f}"),
        LearningRateMonitor(logging_interval="epoch"),
        TQDMProgressBar(refresh_rate=args.pbar_refresh_rate),
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        log_every_n_steps=50,
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
