"""Distributed ERA5 ConvLSTM training.

Usage:
    torchrun --nproc_per_node=2 scripts/train.py
    torchrun --nproc_per_node=1 scripts/train.py  # single GPU / testing

See docs/DISTRIBUTED_TRAINING.md for multi-node setup.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

from pipeline.data.dataset import ERA5Dataset
from pipeline.data.era5 import LOCAL_ZARR_PATH, VARIABLES as REAL_VARIABLES
from pipeline.data.synthetic import SyntheticERA5Dataset
from pipeline.models.convlstm import ConvLSTMForecast
from pipeline.training.distributed import (
    cleanup_distributed,
    is_main_process,
    save_checkpoint,
    setup_distributed,
    train_one_epoch,
    validate,
)


def build_datasets(args):
    """Build synthetic or real datasets depending on CLI configuration."""
    if args.data_mode == "synthetic":
        train_dataset = SyntheticERA5Dataset(
            n_samples=args.n_train,
            seq_len=args.seq_len,
            n_vars=args.synthetic_vars,
        )
        val_dataset = SyntheticERA5Dataset(
            n_samples=args.n_val,
            seq_len=args.seq_len,
            n_vars=args.synthetic_vars,
        )
        variable_names = [f"synthetic_var_{idx}" for idx in range(args.synthetic_vars)]
        return train_dataset, val_dataset, variable_names

    variables = args.variables or list(REAL_VARIABLES)
    full_dataset = ERA5Dataset(
        zarr_path=args.zarr_path,
        variables=variables,
        seq_len=args.seq_len,
    )

    total_samples = len(full_dataset)
    val_samples = max(1, int(total_samples * args.val_fraction))
    train_samples = total_samples - val_samples
    if train_samples <= 0:
        raise ValueError(
            "Not enough samples for a train/validation split. "
            "Decrease --seq_len or --val_fraction, or provide a larger zarr store."
        )

    # Use a chronological split to avoid leaking future timesteps into training.
    train_dataset = Subset(full_dataset, range(0, train_samples))
    val_dataset = Subset(full_dataset, range(train_samples, total_samples))
    return train_dataset, val_dataset, variables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",         type=int,   default=10)
    parser.add_argument("--batch_size",     type=int,   default=16,
                        help="Per-GPU batch size. Effective batch = batch_size × world_size.")
    parser.add_argument("--lr",             type=float, default=1e-3,
                        help="Base LR. Linear scaling rule: lr × world_size if scaling.")
    parser.add_argument("--hidden",         type=int,   default=64)
    parser.add_argument("--n_train",        type=int,   default=800)
    parser.add_argument("--n_val",          type=int,   default=200)
    parser.add_argument("--seq_len",        type=int,   default=6)
    parser.add_argument("--num_workers",    type=int,   default=2)
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    parser.add_argument("--data_mode",      choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--zarr_path",      type=str,   default=LOCAL_ZARR_PATH)
    parser.add_argument("--variables",      nargs="+",  default=None,
                        help="Variables to train on when --data_mode=real.")
    parser.add_argument("--val_fraction",   type=float, default=0.2,
                        help="Fraction of real-data windows reserved for validation.")
    parser.add_argument("--synthetic_vars", type=int,   default=4,
                        help="Number of channels for synthetic training mode.")
    parser.add_argument("--backend",        choices=["auto", "nccl", "gloo"], default="auto",
                        help="Distributed backend. auto=nccl when CUDA is available, else gloo.")
    args = parser.parse_args()

    requested_backend = None if args.backend == "auto" else args.backend
    local_rank, backend = setup_distributed(requested_backend)
    device = torch.device(f"cuda:{local_rank}" if backend == "nccl" else "cpu")
    world_size = dist.get_world_size()
    rank       = dist.get_rank()

    if is_main_process():
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"ERA5 ConvLSTM — Distributed Training")
        print(f"World size: {world_size} GPUs")
        print(f"Data mode: {args.data_mode}")
        print(f"Backend: {backend}")
        print(f"Device: {device}")
        print(f"Per-GPU batch size: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"{'='*60}\n")

    train_dataset, val_dataset, variables = build_datasets(args)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True,
    )

    if is_main_process():
        print(f"Variables: {variables}")
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        if args.data_mode == "real":
            print(f"Zarr path: {args.zarr_path}")

    model = ConvLSTMForecast(
        in_channels=len(variables),
        hidden_channels=args.hidden,
        out_channels=len(variables),
    ).to(device)
    if backend == "nccl":
        model = DDP(model, device_ids=[local_rank])
    else:
        model = DDP(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch, rank)
        val_loss   = validate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed    = time.time() - t0

        if is_main_process():
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"Train Loss: {train_loss:.5f} | "
                f"Val Loss: {val_loss:.5f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e} | "
                f"Time: {elapsed:.1f}s"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                ckpt_path = os.path.join(args.checkpoint_dir, "best_model.pt")
                save_checkpoint(model, optimizer, epoch, val_loss, ckpt_path)
                print(f"  New best val loss: {val_loss:.5f} — checkpoint saved")

    if is_main_process():
        print(f"\nTraining complete. Best val loss: {best_val_loss:.5f}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
