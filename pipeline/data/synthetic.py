"""Synthetic ERA5-like data generators for offline testing."""
from __future__ import annotations

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


def make_era5_dataset(
    n_times: int = 24,
    lat_res: float = 2.5,
    lon_res: float = 2.5,
) -> xr.Dataset:
    """Create a synthetic ERA5-like xarray Dataset with CF conventions.

    Generates z500, t850, u10, v10 with realistic planetary wave structure.
    Use in place of real ERA5 for testing without network access.

    In production: replace with xr.open_zarr("gs://your-bucket/era5.zarr")
    """
    times = xr.cftime_range("2020-01-01", periods=n_times, freq="6H")
    lats  = np.arange(-90, 90 + lat_res, lat_res)
    lons  = np.arange(0,  360,           lon_res)

    n_lat, n_lon = len(lats), len(lons)
    rng = np.random.default_rng(42)

    lon_grid, lat_grid = np.meshgrid(np.deg2rad(lons), np.deg2rad(lats))
    base = np.sin(2 * lon_grid) * np.cos(lat_grid)  # wavenumber-2 structure

    def make_field(mean, std, temporal_scale, spatial_scale=1.0):
        field = np.zeros((n_times, n_lat, n_lon))
        for t in range(n_times):
            field[t] = (
                mean
                + spatial_scale * base * std * np.cos(0.26 * t)
                + std * 0.3 * rng.standard_normal((n_lat, n_lon))
            )
        return field.astype(np.float32)

    return xr.Dataset(
        {
            "z500": xr.DataArray(
                make_field(mean=5500, std=200, temporal_scale=0.1),
                dims=["time", "lat", "lon"],
                attrs={"long_name": "Geopotential height at 500 hPa", "units": "m",
                       "standard_name": "geopotential_height", "pressure_level": 500},
            ),
            "t850": xr.DataArray(
                make_field(mean=280, std=15, temporal_scale=0.05),
                dims=["time", "lat", "lon"],
                attrs={"long_name": "Temperature at 850 hPa", "units": "K",
                       "standard_name": "air_temperature", "pressure_level": 850},
            ),
            "u10": xr.DataArray(
                make_field(mean=0, std=8, temporal_scale=0.2),
                dims=["time", "lat", "lon"],
                attrs={"long_name": "10m U-component of wind", "units": "m s**-1",
                       "standard_name": "eastward_wind"},
            ),
            "v10": xr.DataArray(
                make_field(mean=0, std=6, temporal_scale=0.2),
                dims=["time", "lat", "lon"],
                attrs={"long_name": "10m V-component of wind", "units": "m s**-1",
                       "standard_name": "northward_wind"},
            ),
        },
        coords={
            "time": times,
            "lat":  xr.DataArray(lats, dims=["lat"],
                        attrs={"units": "degrees_north", "standard_name": "latitude"}),
            "lon":  xr.DataArray(lons, dims=["lon"],
                        attrs={"units": "degrees_east", "standard_name": "longitude"}),
        },
        attrs={
            "title":       "Synthetic ERA5-like reanalysis",
            "source":      "Generated for ERA5 workflow demo",
            "conventions": "CF-1.8",
        },
    )


class SyntheticERA5Dataset(Dataset):
    """In-memory synthetic ERA5 dataset for training without real data.

    Generates spatiotemporal fields with realistic planetary wave structure.
    Replace with pipeline.data.dataset.ERA5Dataset for production use.

    Structure: each sample is T input frames → 1 target frame.
    Variables: 4 atmospheric fields on a height×width grid.
    """

    def __init__(
        self,
        n_samples: int = 1000,
        seq_len: int = 6,
        n_vars: int = 4,
        height: int = 32,
        width: int = 64,
    ):
        self.n_samples = n_samples
        self.seq_len   = seq_len

        rng = np.random.default_rng(42)
        total_frames = n_samples + seq_len

        lons = np.linspace(0, 2 * np.pi, width)
        lats = np.linspace(-np.pi / 2, np.pi / 2, height)
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        base_field = np.sin(2 * lon_grid) * np.cos(lat_grid)

        frames = []
        for t in range(total_frames):
            frame = base_field * np.cos(0.1 * t) + 0.1 * rng.standard_normal(
                (n_vars, height, width)
            )
            frames.append(frame)
        self.data = np.stack(frames, axis=0).astype(np.float32)  # (total, C, H, W)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]      # (T, C, H, W)
        y = self.data[idx + self.seq_len]             # (C, H, W)
        return torch.from_numpy(x), torch.from_numpy(y)
