from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader, Dataset


class ERA5Dataset(Dataset):
    """Zarr-backed sliding window dataset for ConvLSTM training.

    Reads directly from local or cloud zarr — no full load into RAM.
    __getitem__ loads only seq_len+1 timesteps per call.

    Normalization stats are computed once at init and stored as scalars.
    For production, compute mean/std over a full year and save to JSON
    rather than recomputing on every instantiation.

    Returns:
        x: (seq_len, C, H, W)  — input sequence
        y: (C, H, W)           — target next frame
    """

    def __init__(
        self,
        zarr_path: str,
        variables: Sequence[str],
        seq_len: int = 6,
    ) -> None:
        self.ds        = xr.open_zarr(zarr_path, consolidated=True)[list(variables)]
        self.variables = list(variables)
        self.seq_len   = seq_len
        self.n_times   = self.ds.sizes["time"]

        print(f"  Computing normalization stats ...")
        self.means: dict[str, float] = {}
        self.stds:  dict[str, float] = {}

        for v in self.variables:
            mean_val = float(self.ds[v].mean().compute())
            std_val  = float(self.ds[v].std().compute())
            self.means[v] = mean_val
            self.stds[v]  = std_val if std_val > 0 else 1.0
            print(f"    {v}: mean={mean_val:.4f}  std={std_val:.4f}")

    def __len__(self) -> int:
        return self.n_times - self.seq_len

    def __getitem__(self, idx: int):
        window = self.ds.isel(time=slice(idx, idx + self.seq_len + 1))

        arr = np.stack(
            [
                ((window[v].values - self.means[v]) / self.stds[v]).astype(np.float32)
                for v in self.variables
            ],
            axis=1,
        )  # (T+1, C, H, W)

        x = torch.from_numpy(arr[:self.seq_len])   # (seq_len, C, H, W)
        y = torch.from_numpy(arr[self.seq_len])    # (C, H, W)
        return x, y


def build_dataloader(zarr_path: str) -> DataLoader:
    """Build a DataLoader from a local zarr store using default variables.

    num_workers=4: M2/32GB can handle multiprocess workers without OOM.
    persistent_workers=True: keeps worker processes alive between epochs.
    pin_memory=False: only helps with CUDA, not MPS.
    """
    print("\n── Step 5: PyTorch DataLoader ──")

    dataset = ERA5Dataset(
        zarr_path=zarr_path,
        variables=[
            "2m_temperature",
            "volumetric_soil_water_layer_1",
            "leaf_area_index_high_vegetation",
        ],
        seq_len=6,
    )

    print(f"  Dataset length: {len(dataset)} windows")
    x, y = dataset[0]
    print(f"  Sample x: {x.shape}  (seq_len, C, H, W)")
    print(f"  Sample y: {y.shape}  (C, H, W)")

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
        pin_memory=False,
    )

    batch_x, batch_y = next(iter(loader))
    print(f"  Batch x: {batch_x.shape}")
    print(f"  Batch y: {batch_y.shape}")
    print(f"  Batch x mean: {batch_x.mean():.4f}  std: {batch_x.std():.4f}")

    return loader
