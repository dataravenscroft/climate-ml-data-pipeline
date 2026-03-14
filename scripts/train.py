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
from torch.utils.data import DataLoader, DistributedSampler

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
    parser.add_argument("--checkpoint_dir", type=str,   default="checkpoints")
    args = parser.parse_args()

    local_rank = setup_distributed()
    device     = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    rank       = dist.get_rank()

    if is_main_process():
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"ERA5 ConvLSTM — Distributed Training")
        print(f"World size: {world_size} GPUs")
        print(f"Per-GPU batch size: {args.batch_size}")
        print(f"Effective batch size: {args.batch_size * world_size}")
        print(f"{'='*60}\n")

    train_dataset = SyntheticERA5Dataset(n_samples=args.n_train)
    val_dataset   = SyntheticERA5Dataset(n_samples=args.n_val)

    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   shuffle=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=2, pin_memory=True,
    )

    model = ConvLSTMForecast(hidden_channels=args.hidden).to(device)
    model = DDP(model, device_ids=[local_rank])

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
