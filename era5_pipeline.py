"""
era5_workflow_real.py
=====================
Real ERA5 workflow using ARCO ERA5 on Google Cloud:
  1. Open real ERA5 from public Zarr
  2. Subset a manageable region/time range
  3. Convert to local Zarr (chunked, compressed)
  4. Regrid with xESMF
  5. Parallelize with Dask
  6. Connect to a PyTorch Dataset/DataLoader

Variables used:
  - 2m_temperature
  - volumetric_soil_water_layer_1
  - leaf_area_index_high_vegetation

Source store:
  gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3

Run:
  pip install xarray zarr dask[distributed] xesmf numpy gcsfs s3fs torch
  python era5_workflow_real.py
"""

from __future__ import annotations

import os
import time
from typing import Sequence

import numpy as np
import xarray as xr
import zarr
from dask.distributed import Client
from zarr.codecs import ZstdCodec

import torch
from torch.utils.data import Dataset, DataLoader


# -----------------------------------------------------------------------------
# 0. CONFIG
# -----------------------------------------------------------------------------

ARCO_ERA5_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

VARIABLES = [
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "leaf_area_index_high_vegetation",
]

# Keep the first pass small and cheap.
TIME_START = "2020-01-01"
TIME_END   = "2020-01-14"

# North America-ish example box.
LAT_MAX = 55.0
LAT_MIN = 20.0
LON_MIN = 230.0   # 0–360 convention
LON_MAX = 300.0

LOCAL_ZARR_PATH = "data/zarr/era5_real_subset.zarr"
LOCAL_REGRID_PATH = "data/regrid/era5_real_subset_1deg.zarr"

os.makedirs("data/zarr", exist_ok=True)
os.makedirs("data/regrid", exist_ok=True)


print("=" * 72)
print("ERA5 Climate Workflow — REAL ARCO ERA5")
print("=" * 72)


# -----------------------------------------------------------------------------
# 1. OPEN REAL ERA5 AND SUBSET
# -----------------------------------------------------------------------------

print("\n── Step 1: Opening ARCO ERA5 dataset ──")

ds = xr.open_zarr(
    ARCO_ERA5_PATH,
    consolidated=True,
    storage_options={"token": "anon"},
    chunks={"time": 1},
)

print("Opened remote dataset lazily.")
print(ds)
print(f"\nAvailable time range: {ds.attrs.get('valid_time_start')} → {ds.attrs.get('valid_time_stop')}")
print(f"Variables selected: {VARIABLES}")

# Select only the variables we want, then subset.
# ERA5 latitude is descending, so use slice(lat_max, lat_min).
subset = ds[VARIABLES].sel(
    time=slice(TIME_START, TIME_END),
    latitude=slice(LAT_MAX, LAT_MIN),
    longitude=slice(LON_MIN, LON_MAX),
)

print("\nSubset overview:")
print(subset)
print(f"Subset dims: {dict(subset.sizes)}")

# Rename coords to lat/lon for cleaner xESMF + ML ergonomics.
subset = subset.rename({"latitude": "lat", "longitude": "lon"})

# Add a simple derived variable example.
subset["temperature_c"] = subset["2m_temperature"] - 273.15
subset["temperature_c"].attrs = {
    "long_name": "2m temperature",
    "units": "degC",
}

# Quick lazy summary.
print("\nLazy stats (not computed yet):")
print(subset["2m_temperature"])
print(subset["volumetric_soil_water_layer_1"])
print(subset["leaf_area_index_high_vegetation"])

# Right before Step 2, clear the inherited encodings:
subset = subset.drop_encoding()

for var in subset.data_vars:
    subset[var].encoding = {}
for coord in subset.coords:
    subset[coord].encoding = {}
# -----------------------------------------------------------------------------
# 2. WRITE LOCAL ZARR
# -----------------------------------------------------------------------------

print("\n── Step 2: Writing local chunked Zarr ──")

import os
import shutil
import time

chunks = {"time": 1, "lat": -1, "lon": -1}
subset_chunked = subset.chunk(chunks)

if os.path.exists(LOCAL_ZARR_PATH):
    shutil.rmtree(LOCAL_ZARR_PATH)

t0 = time.time()
subset_chunked.to_zarr(
    LOCAL_ZARR_PATH,
    mode="w",
    consolidated=True,
)
elapsed = time.time() - t0

print(f"Wrote local Zarr: {LOCAL_ZARR_PATH} ({elapsed:.2f}s)")


# -----------------------------------------------------------------------------
# 3. REGRID WITH xESMF
# -----------------------------------------------------------------------------

print("\n── Step 3: Regridding with xESMF ──")

try:
    import xesmf as xe

    # Regrid to a coarser 1-degree grid for a simpler ML prototype.
    target_grid = xr.Dataset(
        {
            "lat": (["lat"], np.arange(LAT_MIN, LAT_MAX + 1.0, 1.0)),
            "lon": (["lon"], np.arange(LON_MIN, LON_MAX + 1.0, 1.0)),
        }
    )

    print(f"Source grid: {len(ds_local.lat)} × {len(ds_local.lon)}")
    print(f"Target grid: {len(target_grid.lat)} × {len(target_grid.lon)}")

    # Bilinear is fine here for temperature / soil moisture / LAI demo purposes.
    regridder = xe.Regridder(
        ds_local[["2m_temperature", "volumetric_soil_water_layer_1", "leaf_area_index_high_vegetation"]],
        target_grid,
        method="bilinear",
        periodic=False,
        reuse_weights=False,
    )

    ds_regrid = regridder(
        ds_local[["2m_temperature", "volumetric_soil_water_layer_1", "leaf_area_index_high_vegetation"]]
    )

    print(f"Regridded dims: {dict(ds_regrid.sizes)}")

    ds_regrid.chunk({"time": 1, "lat": -1, "lon": -1}).to_zarr(
        LOCAL_REGRID_PATH,
        mode="w",
        consolidated=True,
    )
    print(f"Saved regridded Zarr: {LOCAL_REGRID_PATH}")

except ImportError:
    print("xESMF not installed. Install with:")
    print("  pip install xesmf")
    print("If that fails on Mac, conda-forge is sometimes easier for xesmf/esmf.")


# -----------------------------------------------------------------------------
# 4. DASK COMPUTATION
# -----------------------------------------------------------------------------

print("\n── Step 4: Dask parallel computation ──")

client = Client(n_workers=2, threads_per_worker=2, memory_limit="2GB")
print(f"Dask dashboard: {client.dashboard_link}")

ds_dask = xr.open_zarr(
    LOCAL_ZARR_PATH,
    consolidated=True,
    chunks={"time": 1, "lat": -1, "lon": -1},
)

# Build lazy computations.
temp_c_mean = (ds_dask["2m_temperature"] - 273.15).mean("time")
soil_std = ds_dask["volumetric_soil_water_layer_1"].std("time")
lai_max = ds_dask["leaf_area_index_high_vegetation"].max("time")

print("\nLazy graphs built.")
print(temp_c_mean.data)
print(soil_std.data)
print(lai_max.data)

t0 = time.time()
temp_c_mean_val = temp_c_mean.compute()
soil_std_val = soil_std.compute()
lai_max_val = lai_max.compute()
elapsed = time.time() - t0

print(f"\nComputed in {elapsed:.2f}s")
print(f"Mean 2m temp (degC), domain average: {float(temp_c_mean_val.mean()):.2f}")
print(f"Soil moisture temporal std, domain average: {float(soil_std_val.mean()):.4f}")
print(f"LAI high vegetation max, domain average: {float(lai_max_val.mean()):.4f}")


# -----------------------------------------------------------------------------
# 5. PYTORCH DATASET / DATALOADER
# -----------------------------------------------------------------------------

print("\n── Step 5: Connecting Zarr → PyTorch DataLoader ──")

class ERA5ZarrDataset(Dataset):
    """
    Reads a local or remote Zarr store directly into PyTorch-ready tensors.

    Output:
      x: (T, C, H, W)
      y: (C, H, W)
    """

    def __init__(
        self,
        zarr_path: str,
        variables: Sequence[str],
        seq_len: int = 6,
    ) -> None:
        self.ds = xr.open_zarr(zarr_path, consolidated=True)[list(variables)]
        self.variables = list(variables)
        self.seq_len = seq_len
        self.n_times = self.ds.sizes["time"]

        print(f"  Computing normalization stats for {self.variables} ...")
        self.means = {}
        self.stds = {}

        for v in self.variables:
            mean_val = self.ds[v].mean().compute().item()
            std_val = self.ds[v].std().compute().item()

            if std_val == 0:
                std_val = 1.0

            self.means[v] = float(mean_val)
            self.stds[v] = float(std_val)

    def __len__(self) -> int:
        return self.n_times - self.seq_len

    def __getitem__(self, idx: int):
        frames = self.ds.isel(time=slice(idx, idx + self.seq_len + 1))

        arr = np.stack(
            [
                ((frames[v].values - self.means[v]) / self.stds[v]).astype(np.float32)
                for v in self.variables
            ],
            axis=1,
        )  # (T+1, C, H, W)

        x = torch.from_numpy(arr[: self.seq_len])
        y = torch.from_numpy(arr[self.seq_len])
        return x, y


dataset = ERA5ZarrDataset(
    zarr_path=LOCAL_ZARR_PATH,
    variables=[
        "2m_temperature",
        "volumetric_soil_water_layer_1",
        "leaf_area_index_high_vegetation",
    ],
    seq_len=6,
)

print(f"Dataset length: {len(dataset)}")
x, y = dataset[0]
print(f"Sample x shape: {x.shape}  (T, C, H, W)")
print(f"Sample y shape: {y.shape}  (C, H, W)")
print(f"x mean: {x.mean():.4f}")
print(f"x std:  {x.std():.4f}")

loader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
batch_x, batch_y = next(iter(loader))
print(f"Batch x shape: {batch_x.shape}")
print(f"Batch y shape: {batch_y.shape}")


# -----------------------------------------------------------------------------
# 6. CLOUD PATTERNS
# -----------------------------------------------------------------------------

print("\n── Step 6: Cloud object storage patterns ──")
print("""
Public ARCO ERA5 read pattern:
  xr.open_zarr(
      "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
      consolidated=True,
      storage_options={"token": "anon"},
      chunks={"time": 1},
  )

Local → cloud write pattern:
  ds.to_zarr("gs://your-bucket/era5_subset.zarr", mode="w", consolidated=True)
  ds = xr.open_zarr("gs://your-bucket/era5_subset.zarr", chunks={"time": 1})
""")

client.close()

print("\n" + "=" * 72)
print("Workflow complete.")
print("=" * 72)
print("""
What this version does:
  ✓ pulls real ARCO ERA5 data
  ✓ uses 2m temperature, soil moisture, and high-vegetation LAI
  ✓ subsets before writing local Zarr
  ✓ regrids with xESMF
  ✓ computes lazy stats with Dask
  ✓ feeds a Zarr-backed dataset into a PyTorch DataLoader
""")