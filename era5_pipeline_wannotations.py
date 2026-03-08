"""
era5_pipeline.py
================
ERA5 climate data pipeline optimized for M1 MacBook (8GB unified memory).

Pipeline:
  1. Open real ARCO ERA5 from public GCS Zarr (lazy, no download)
  2. Subset variables / region / time
  3. Write local Zarr with ML-optimized chunking
  4. Regrid with xESMF (bilinear for state vars, conservative for flux)
  5. Dask parallel stats
  6. PyTorch Dataset / DataLoader

Chunking strategy used here: {time: 1, lat: -1, lon: -1}
  Rationale: ML training access pattern — we always want one full
  spatial snapshot per timestep. Each chunk fits ~4-16MB in memory.
  If you need temporal stats instead, rechunk to {time: -1, lat: 10, lon: 10}.

Install:
  pip install xarray zarr dask[distributed] gcsfs torch numpy
  # xESMF needs esmpy — easiest via conda:
  # conda install -c conda-forge xesmf

Run:
  python era5_pipeline.py
"""

from __future__ import annotations

import os
import shutil
import time
import warnings
from typing import Sequence

import numpy as np
import torch
import xarray as xr
import dask
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore", category=UserWarning)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

# Public ARCO ERA5 — no auth required
ARCO_ERA5_PATH = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

VARIABLES = [
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "leaf_area_index_high_vegetation",
]

# Small time window for local dev — expand when you have more disk
TIME_START = "2020-01-01"
TIME_END   = "2020-01-14"   # 2 weeks = 336 hourly timesteps

# CONUS bounding box (ERA5 uses 0–360 longitude)
LAT_MAX =  50.0
LAT_MIN =  25.0
LON_MIN = 235.0   # ~125°W
LON_MAX = 295.0   # ~65°W

LOCAL_ZARR_PATH   = "data/era5_subset.zarr"
LOCAL_REGRID_PATH = "data/era5_subset_1deg.zarr"

os.makedirs("data", exist_ok=True)

# ─── M1 DASK CONFIG ────────────────────────────────────────────────────────────
#
# M1 has 8GB unified memory shared between CPU and GPU.
# Rule of thumb: leave 3GB for OS + Python overhead + model weights.
# Dask gets 4GB total → 2 workers × 2GB each.
#
# threads_per_worker=2 is safe on M1 — GIL is released for numpy/zarr I/O
# so threads genuinely parallelize data loading without fighting each other.
# More workers = more memory overhead for scheduler bookkeeping.

def make_dask_client():
    # On M1/8GB the synchronous threaded scheduler is better than LocalCluster:
    #   - No subprocess spawning (avoids the __main__ guard issue entirely)
    #   - No port conflicts
    #   - Lower memory overhead (no scheduler/nanny processes)
    #   - Threads release the GIL for numpy/zarr I/O so parallelism still works
    #
    # "synchronous" = single thread (good for debugging)
    # "threads"     = thread pool (good for I/O-heavy workloads like zarr reads)
    dask.config.set(scheduler="threads", num_workers=4)
    print("  Dask scheduler: threaded (4 workers, no LocalCluster)")
    return None  # no client object needed


# ─── STEP 1: OPEN & SUBSET ─────────────────────────────────────────────────────
#
# xr.open_zarr is lazy — nothing is downloaded until .compute() is called.
# chunks={"time": 1} overrides the remote chunk layout immediately on open,
# telling Dask to treat each timestep as its own task.
#
# CHUNKING NOTE:
#   Remote ARCO ERA5 chunks: {time: 1, latitude: 721, longitude: 1440}
#   That's already time-first, which suits ML training.
#   We preserve that layout when writing locally.

def open_and_subset(client) -> xr.Dataset:
    print("\n── Step 1: Opening ARCO ERA5 (lazy) ──")

    ds = xr.open_zarr(
        ARCO_ERA5_PATH,
        consolidated=True,
        storage_options={"token": "anon"},
        chunks={"time": 1},          # one chunk per timestep, full spatial map
    )

    print(f"  Remote vars available: {list(ds.data_vars)[:8]} ...")
    print(f"  Subsetting: {TIME_START} → {TIME_END}, lat [{LAT_MIN},{LAT_MAX}], lon [{LON_MIN},{LON_MAX}]")

    # ERA5 latitude is stored descending (90 → -90), so slice(max, min)
    subset = ds[VARIABLES].sel(
        time=slice(TIME_START, TIME_END),
        latitude=slice(LAT_MAX, LAT_MIN),
        longitude=slice(LON_MIN, LON_MAX),
    )

    # Rename to lat/lon — xESMF and most ML libraries expect these names
    subset = subset.rename({"latitude": "lat", "longitude": "lon"})

    # Derived variable — Celsius conversion stays lazy (no .compute() yet)
    subset["t2m_celsius"] = subset["2m_temperature"] - 273.15
    subset["t2m_celsius"].attrs = {"long_name": "2m temperature", "units": "degC"}

    print(f"  Subset dims: {dict(subset.sizes)}")
    _print_memory_estimate(subset)

    return subset


def _print_memory_estimate(ds: xr.Dataset) -> None:
    """Rough in-memory size if fully loaded — useful sanity check on M1."""
    dims = ds.sizes
    n_cells = 1
    for v in dims.values():
        n_cells *= v
    n_vars = len(ds.data_vars)
    mb = n_cells * n_vars * 4 / 1e6
    print(f"  Estimated full load size: {mb:.0f} MB across {n_vars} variables")
    if mb > 2000:
        print("  ⚠  > 2GB — consider narrowing TIME_END or bbox before computing")


# ─── STEP 2: WRITE LOCAL ZARR ──────────────────────────────────────────────────
#
# CHUNKING STRATEGY — {time: 1, lat: -1, lon: -1}
#
#   This layout is optimized for ML training access pattern:
#   "give me frame T, frame T+1, ..., frame T+seq_len"
#   Each .zarr chunk = one full spatial snapshot.
#
#   Memory per chunk (one variable):
#     1 × 100lat × 240lon × 4 bytes ≈ 0.1 MB   ← very safe
#
#   What it's BAD for:
#     - Computing time-mean of a single grid point (must open every chunk)
#     - Climatology (same problem)
#   For those use cases, rechunk to {time: -1, lat: 10, lon: 10} first.
#
#   Compression: zstd level 3 is a good default on M1 —
#   fast decode, ~2-3x compression on float32 climate data.
#   Higher levels (5-9) save disk but CPU cost isn't worth it for local dev.

def write_local_zarr(subset: xr.Dataset) -> xr.Dataset:
    print("\n── Step 2: Writing local Zarr ──")

    ML_CHUNKS = {"time": 1, "lat": -1, "lon": -1}

    subset_chunked = (
        subset
        .drop_encoding()          # drop inherited remote encoding (crucial)
        .chunk(ML_CHUNKS)
    )

    # Clear per-variable encodings that came from the remote store
    for v in list(subset_chunked.data_vars) + list(subset_chunked.coords):
        subset_chunked[v].encoding = {}

    if os.path.exists(LOCAL_ZARR_PATH):
        shutil.rmtree(LOCAL_ZARR_PATH)

    print(f"  Chunk layout: {ML_CHUNKS}")
    print(f"  Writing to {LOCAL_ZARR_PATH} ...")

    t0 = time.time()
    subset_chunked.to_zarr(
        LOCAL_ZARR_PATH,
        mode="w",
        consolidated=True,
        # No custom encoding — let xarray/zarr v3 use its defaults.
        # Zarr v3 changed the compressor spec format and xarray's
        # bridge hasn't fully caught up. Default zstd is applied
        # automatically by zarr v3 anyway.
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Re-open from local — now fully offline
    ds_local = xr.open_zarr(LOCAL_ZARR_PATH, consolidated=True, chunks=ML_CHUNKS)
    print(f"  Local store dims: {dict(ds_local.sizes)}")
    return ds_local


# ─── STEP 3: REGRID WITH xESMF ─────────────────────────────────────────────────
#
# Two regridding methods — choice depends on variable physics:
#
#   bilinear:     smooth interpolation between point values
#                 use for: temperature, wind, geopotential, soil moisture
#                 preserves: local point values
#                 does NOT preserve: area integrals
#
#   conservative: area-weighted averaging
#                 use for: precipitation, radiation, surface fluxes
#                 preserves: total amount across domain (mass/energy conservation)
#                 does NOT preserve: fine spatial structure
#
# Here we regrid to 1° as a cheap ML prototype target.
# In production: ERA5 0.25° → FourCastNet 0.25° (no regrid needed)
#               ERA5 0.25° → ClimaX 5.625° (bilinear for state vars)
#               ERA5 0.25° → WRF boundary (conservative for fluxes)

def regrid(ds_local: xr.Dataset) -> xr.Dataset | None:
    print("\n── Step 3: Regridding with xESMF ──")

    try:
        import xesmf as xe
    except ImportError:
        print("  xESMF not installed — skipping regrid step.")
        print("  Install: conda install -c conda-forge xesmf")
        return None

    target_grid = xr.Dataset({
        "lat": (["lat"], np.arange(LAT_MIN, LAT_MAX + 1.0, 1.0)),
        "lon": (["lon"], np.arange(LON_MIN, LON_MAX + 1.0, 1.0)),
    })

    print(f"  Source: {ds_local.sizes['lat']} × {ds_local.sizes['lon']} (0.25°)")
    print(f"  Target: {len(target_grid.lat)} × {len(target_grid.lon)} (1.0°)")

    # State variables — bilinear is appropriate
    state_vars = ["2m_temperature", "volumetric_soil_water_layer_1",
                  "leaf_area_index_high_vegetation", "t2m_celsius"]
    state_vars = [v for v in state_vars if v in ds_local]

    regridder = xe.Regridder(
        ds_local[state_vars],
        target_grid,
        method="bilinear",
        periodic=False,
        reuse_weights=True,    # cache weights — saves time if called repeatedly
    )

    ds_regrid = regridder(ds_local[state_vars])
    print(f"  Regridded dims: {dict(ds_regrid.sizes)}")

    if os.path.exists(LOCAL_REGRID_PATH):
        shutil.rmtree(LOCAL_REGRID_PATH)

    ds_regrid.chunk({"time": 1, "lat": -1, "lon": -1}).to_zarr(
        LOCAL_REGRID_PATH, mode="w", consolidated=True
    )
    print(f"  Saved: {LOCAL_REGRID_PATH}")
    return ds_regrid


# ─── STEP 4: DASK STATS ────────────────────────────────────────────────────────
#
# Build lazy computation graphs first, then compute() once at the end.
# This lets Dask optimize the full graph before executing — avoids
# redundant reads and can fuse operations across chunks.
#
# On M1 specifically: Dask's threaded scheduler often outperforms
# the distributed LocalCluster for small datasets because it avoids
# serialization overhead. Switch to synchronous for debugging:
#   with dask.config.set(scheduler='synchronous'): ds.compute()

def compute_stats(ds_local: xr.Dataset) -> None:
    print("\n── Step 4: Dask parallel stats ──")

    # Build all lazy graphs before computing any of them
    temp_mean = (ds_local["2m_temperature"] - 273.15).mean("time")
    soil_std  = ds_local["volumetric_soil_water_layer_1"].std("time")
    lai_max   = ds_local["leaf_area_index_high_vegetation"].max("time")

    print("  Lazy graphs built — computing now ...")
    t0 = time.time()

    # Compute all three in one pass — Dask reads each chunk once for all ops
    import dask
    temp_mean_val, soil_std_val, lai_max_val = dask.compute(
        temp_mean, soil_std, lai_max
    )
    elapsed = time.time() - t0

    print(f"  Computed in {elapsed:.2f}s")
    print(f"  Mean 2m temp (°C):          {float(temp_mean_val.mean()):.2f}")
    print(f"  Soil moisture temporal std:  {float(soil_std_val.mean()):.4f}")
    print(f"  LAI high veg max:            {float(lai_max_val.mean()):.4f}")


# ─── STEP 5: PYTORCH DATASET ───────────────────────────────────────────────────
#
# Reads directly from local Zarr — no full load into RAM.
# __getitem__ loads only seq_len+1 timesteps per call.
#
# Normalization: computed once at init, stored as scalars.
# For production: compute mean/std on a full year, save to a JSON file,
# reload at inference time — never recompute on the fly.
#
# num_workers=0 in DataLoader: M1 multiprocessing with fork can OOM
# due to copy-on-write memory not being released. Use 0 for local dev,
# spawn context or num_workers=2 with persistent_workers=True in production.

class ERA5Dataset(Dataset):
    """
    Zarr-backed sliding window dataset for ConvLSTM training.

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
            # .compute() here is intentional — we need scalars, not lazy arrays
            mean_val = float(self.ds[v].mean().compute())
            std_val  = float(self.ds[v].std().compute())
            self.means[v] = mean_val
            self.stds[v]  = std_val if std_val > 0 else 1.0
            print(f"    {v}: mean={mean_val:.4f}  std={std_val:.4f}")

    def __len__(self) -> int:
        # Last valid index: n_times - seq_len - 1
        # (need seq_len inputs + 1 target)
        return self.n_times - self.seq_len

    def __getitem__(self, idx: int):
        # Load seq_len + 1 consecutive frames
        # isel with a slice loads only those chunks — not the full dataset
        window = self.ds.isel(time=slice(idx, idx + self.seq_len + 1))

        # Stack variables along channel dim → (T+1, C, H, W)
        arr = np.stack(
            [
                ((window[v].values - self.means[v]) / self.stds[v]).astype(np.float32)
                for v in self.variables
            ],
            axis=1,
        )

        x = torch.from_numpy(arr[:self.seq_len])   # (seq_len, C, H, W)
        y = torch.from_numpy(arr[self.seq_len])    # (C, H, W)
        return x, y


def build_dataloader(zarr_path: str) -> DataLoader:
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

    # num_workers=0: safest on M1 — avoids fork-related memory issues
    # For faster loading: num_workers=2, persistent_workers=True
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=False,   # pin_memory=True only helps with CUDA, not MPS
    )

    batch_x, batch_y = next(iter(loader))
    print(f"  Batch x: {batch_x.shape}")
    print(f"  Batch y: {batch_y.shape}")
    print(f"  Batch x mean: {batch_x.mean():.4f}  std: {batch_x.std():.4f}")

    return loader


# ─── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 68)
    print("ERA5 Climate Pipeline — M1 MacBook (8GB)")
    print("=" * 68)

    make_dask_client()

    subset   = open_and_subset(None)
    ds_local = write_local_zarr(subset)
    _        = regrid(ds_local)
    compute_stats(ds_local)
    loader   = build_dataloader(LOCAL_ZARR_PATH)

    print("\n" + "=" * 68)
    print("Pipeline complete.")
    print(f"  Local Zarr:   {LOCAL_ZARR_PATH}")
    print(f"  Regrid Zarr:  {LOCAL_REGRID_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()