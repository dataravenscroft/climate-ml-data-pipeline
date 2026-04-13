from __future__ import annotations

from typing import Sequence

from torch.utils.data import Subset

from pipeline.data.dataset import ERA5Dataset
from pipeline.data.era5 import LOCAL_ZARR_PATH, VARIABLES as REAL_VARIABLES
from pipeline.data.synthetic import SyntheticERA5Dataset


def build_datasets(
    data_mode: str = "synthetic",
    seq_len: int = 6,
    synthetic_vars: int = 4,
    n_train: int = 800,
    n_val: int = 200,
    zarr_path: str = LOCAL_ZARR_PATH,
    variables: Sequence[str] | None = None,
    val_fraction: float = 0.2,
):
    """Build synthetic or real datasets and return variable names."""
    if data_mode == "synthetic":
        train_dataset = SyntheticERA5Dataset(
            n_samples=n_train,
            seq_len=seq_len,
            n_vars=synthetic_vars,
        )
        val_dataset = SyntheticERA5Dataset(
            n_samples=n_val,
            seq_len=seq_len,
            n_vars=synthetic_vars,
        )
        variable_names = [f"synthetic_var_{idx}" for idx in range(synthetic_vars)]
        return train_dataset, val_dataset, variable_names

    selected_variables = list(variables or REAL_VARIABLES)
    full_dataset = ERA5Dataset(
        zarr_path=zarr_path,
        variables=selected_variables,
        seq_len=seq_len,
    )

    total_samples = len(full_dataset)
    val_samples = max(1, int(total_samples * val_fraction))
    train_samples = total_samples - val_samples
    if train_samples <= 0:
        raise ValueError(
            "Not enough samples for a train/validation split. "
            "Decrease seq_len or val_fraction, or provide a larger zarr store."
        )

    train_dataset = Subset(full_dataset, range(0, train_samples))
    val_dataset = Subset(full_dataset, range(train_samples, total_samples))
    return train_dataset, val_dataset, selected_variables
