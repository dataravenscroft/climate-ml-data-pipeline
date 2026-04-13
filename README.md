# Climate ML Data Pipeline for ERA5 Forecasting

Climate-focused machine learning project built to demonstrate the full path from
Earth-system data ingestion to spatiotemporal model training. The repo combines:

- ERA5 reanalysis preprocessing with `xarray`, `zarr`, `dask`, and `xESMF`
- PyTorch ConvLSTM modeling for next-step forecasting on gridded climate fields
- distributed training with `torchrun` and `DistributedDataParallel`

---

## Project Summary

This project ingests public ERA5 data from the ARCO archive, subsets and rechunks
it into ML-friendly Zarr stores, and trains a ConvLSTM to predict the next
atmospheric state from a sliding window of prior timesteps.

The current implemented real-data workflow uses a mixed forecasting and
land-surface variable set over a fixed CONUS domain and two-week time window:

- `2m_temperature`
- `10m_u_component_of_wind`
- `10m_v_component_of_wind`
- `temperature_850`
- `geopotential_500`
- `volumetric_soil_water_layer_1`
- `leaf_area_index_high_vegetation`
- `surface_solar_radiation_downwards`
- `total_sky_direct_solar_radiation_at_surface`

Those variables are written to a local Zarr store, regridded to 1 degree, and
fed into a PyTorch `Dataset` that supports distributed training.

The repo also includes a synthetic ERA5-like generator for fast offline testing,
which makes it easy to validate the training stack before running on real data.

## Why This Matters

Climate AI work is not just model architecture. It also depends on robust
scientific data handling, geospatial resampling, chunked storage, and scalable
training loops. This project is designed to show fluency across that full stack:

- cloud-native climate data access
- spatiotemporal ML data preparation
- geospatial regridding and array computing
- distributed deep learning for forecasting workloads

## NVIDIA Earth-2 Alignment

This repo is being developed as an Earth-2-aligned portfolio project. The goal
is to demonstrate the parts of the workflow that map well onto NVIDIA's weather
and climate AI stack:

- ERA5 ingestion and ML-ready preprocessing, which mirrors the data curation
  stage needed before large-scale forecasting or diagnostic modeling
- distributed PyTorch training, which is directly relevant to PhysicsNeMo-style
  training workflows
- gridded spatiotemporal forecasting on meteorological variables, which fits the
  problem class addressed by Earth-2 weather models
- regional diagnostic outputs, which connect forecasting infrastructure to
  decision-useful climate and resilience applications

Planned next steps for stronger Earth-2 alignment:

- add a PhysicsNeMo-aligned training recipe or dataset/config layout
- add an Earth2Studio inference demo with a pretrained weather model
- expand beyond the ConvLSTM baseline toward architectures that better match the
  current Earth-2 model ecosystem

## Technical Highlights

- Public ERA5 ingestion from ARCO on Google Cloud via `xr.open_zarr`
- Zarr-first preprocessing with chunk layouts tuned for ML sample access
- Dask-backed lazy computation for statistics and preprocessing
- xESMF bilinear regridding onto a coarser target grid
- Sliding-window PyTorch dataset for sequence-to-one forecasting
- ConvLSTM encoder for spatially aware temporal forecasting
- DDP training entry point for single-node and multi-node scaling
- Forecast evaluation with RMSE, latitude-weighted RMSE, anomaly correlation, and saved plots

## Current Status

- Implemented: ERA5 ingestion, subsetting, local Zarr writing, regridding, summary stats, PyTorch dataset, ConvLSTM model, DDP training utilities
- Implemented: training on both synthetic ERA5-like data and the real local ERA5 Zarr store
- Implemented: offline forecast evaluation script with skill metrics and visualization outputs
- Current real-data scope: CONUS subset, 9 variables, 2-week sample window
- Next step: extend the real-data training path to larger temporal coverage and richer benchmark variable sets

---

## Repository Structure

```
climate-ml-data-pipeline/
├── pipeline/                    # importable package
│   ├── data/
│   │   ├── era5.py              # ARCO ERA5 open/subset, Dask config, config constants
│   │   ├── zarr_store.py        # write_local_zarr
│   │   ├── regrid.py            # xESMF bilinear regridding
│   │   ├── dataset.py           # ERA5Dataset (zarr-backed, for real data)
│   │   └── synthetic.py        # SyntheticERA5Dataset + make_era5_dataset
│   ├── models/
│   │   └── convlstm.py          # ConvLSTMCell + ConvLSTMForecast
│   └── training/
│       ├── distributed.py       # DDP utilities + training/validation loops
│       ├── data_setup.py        # shared dataset builders for training/evaluation
│       └── metrics.py           # forecast skill metrics
├── scripts/
│   ├── run_pipeline.py                        # entry point: python scripts/run_pipeline.py
│   ├── pipeline_2_arcgridAML_currenttooling.py  # wrapper for water stress demo
│   ├── train.py                               # entry point: torchrun scripts/train.py
│   └── evaluate_forecast.py                  # checkpoint evaluation + plots
├── notebooks/
│   ├── era5_pipeline.ipynb      # step-by-step pipeline walkthrough
│   └── era5_exploration.ipynb   # exploratory analysis
├── docs/
│   └── DISTRIBUTED_TRAINING.md  # DDP setup, launch commands, speedup results
├── data/                        # gitignored — zarr stores written here
├── references/
│   └── GraphCast_Google_2023.pdf
└── requirements.txt
```

---

## Architecture Overview

```text
ARCO ERA5 (public zarr on GCS)
        |
        v
open + subset with xarray
        |
        v
write local zarr with ML-friendly chunks
        |
        +--> optional xESMF regridding + Dask summary stats
        |
        v
ERA5Dataset sliding windows
        |
        v
ConvLSTMForecast (PyTorch)
        |
        v
DDP training with torchrun
```

## ConvLSTM Forward Pass

The ConvLSTM implementation is generic over channel count and grid size.
The default synthetic setup uses:

The core architectural insight: **the T dimension exists only in the input tensor, then disappears.**

```
Input:  x  →  shape (N, T, C, H, W)
              N = batch windows
              T = timesteps (6-hour ERA5 frames)
              C = atmospheric variables
              H = latitude grid points
              W = longitude grid points
```

Inside `ConvLSTMCell.forward()`, a loop strips T away one frame at a time:

```python
for t in range(T):
    inp = x[:, t]          # (N, C, H, W) — T is gone
    h, c = self.cell(inp, h, c)
```

Conv2d never sees the T dimension. Temporal information is carried across steps
by the state tensors `h` (hidden/working) and `c` (cell/long-term memory),
both with shape `(N, hidden_channels, H, W)`.

### Gate equations (per spatial location, across H×W via Conv2d)

```
f = sigmoid(W_f * [h_prev, inp])   # forget — how much of c to keep
i = sigmoid(W_i * [h_prev, inp])   # input  — how much new info to write
o = sigmoid(W_o * [h_prev, inp])   # output — how much of c to expose
g = tanh(W_g * [h_prev, inp])      # candidate cell value

c = f ⊙ c_prev + i ⊙ g            # long-term memory update
h = o ⊙ tanh(c)                   # working state
```

The `*` is Conv2d, not matrix multiply — that's the "Conv" in ConvLSTM.
Each gate is a **spatial field**, not a scalar. A cold front appears as a
low-forget *region* in the f field across H×W.

### Gate behaviour in climate terms

| Gate | What it does | Climate example |
|------|-------------|-----------------|
| Forget (f) | How much of c to keep | Cold front arrives → f drops to ~0.2, wipes prior baseline |
| Input (i)  | How much new info to write | Abrupt regime change → i spikes to ~0.95 |
| Output (o) | How much of c to expose via h | Final timestep → o high, exposes state for forecast |
| Cell (g)   | New candidate value | Current temperature anomaly |

---

## Architecture

```
Input: (B, T=6, C=4, H=32, W=64)   ← synthetic training default
│
├── ConvLSTM Layer 1  (64 hidden channels)
│     └── 3×3 conv gates preserve spatial structure at each timestep
├── ConvLSTM Layer 2  (64 hidden channels)
│     └── hidden state h carries atmospheric memory forward in time
│
├── Final hidden state hT  → encodes full 36-hour trajectory
│
├── Projection head
│     ├── Conv2d 64 → 32  + ReLU
│     └── Conv2d 32 → 4
│
Output: (B, C=4, H=32, W=64)        ← predicted next timestep
```

**Why ConvLSTM over standard LSTM:** Standard LSTM flattens the weather
grid into a vector, destroying spatial relationships. ConvLSTM replaces
matrix multiplications in the LSTM gates with 3×3 convolutions — the
grid stays intact throughout, so the model learns spatial processes, e.g. that pressure gradients
move eastward and cold fronts have characteristic spatial shapes.

---

## Quickstart

### 1. Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> xESMF may require conda on some systems:
> `conda install -c conda-forge xesmf`

### 2. Run the real ERA5 preprocessing pipeline

```bash
python scripts/run_pipeline.py
```

This will:
- Open ARCO ERA5 lazily from public GCS (no credentials needed)
- Subset to CONUS, 2 weeks of hourly data
- Select temperature, wind, upper-air, land-surface, and solar-radiation variables
- Write local zarr with ML-optimized chunking (`{time: 1, lat: -1, lon: -1}`)
- Regrid to 1° using xESMF bilinear interpolation #Unable to install 
- Compute Dask parallel temporal statistics
- Connect the zarr store to a PyTorch DataLoader

### 3. Train on the real ERA5 Zarr store

```bash
torchrun --nproc_per_node=1 scripts/train.py \
  --data_mode real \
  --zarr_path data/era5_subset.zarr \
  --variables \
    2m_temperature \
    10m_u_component_of_wind \
    10m_v_component_of_wind \
    temperature_850 \
    geopotential_500 \
    volumetric_soil_water_layer_1 \
    leaf_area_index_high_vegetation \
    surface_solar_radiation_downwards \
    total_sky_direct_solar_radiation_at_surface \
  --epochs 10
```

This will:
- Read the local Zarr store produced by `scripts/run_pipeline.py`
- Create normalized sliding windows with `ERA5Dataset`
- Split windows chronologically into train and validation subsets
- Train the ConvLSTM on the real ERA5 variables

On CPU-only machines, add `--backend gloo`. The default `--backend auto`
selects `nccl` when CUDA is available and `gloo` otherwise.

### 4. Train on synthetic ERA5-like data

```bash
torchrun --nproc_per_node=1 scripts/train.py --data_mode synthetic --epochs 10
```

This will:
- Exercise the training stack without needing remote ERA5 access or a local Zarr store
- Use a 4-channel synthetic benchmark dataset with realistic-looking spatiotemporal structure

### 5. Run the water stress demo

```bash
python scripts/pipeline_2_arcgridAML_currenttooling.py
```

### 6. Evaluate forecast skill and save plots

```bash
python scripts/evaluate_forecast.py \
  --checkpoint checkpoints/best_model.pt \
  --data_mode real \
  --zarr_path data/era5_subset.zarr \
  --variables \
    2m_temperature \
    10m_u_component_of_wind \
    10m_v_component_of_wind \
    temperature_850 \
    geopotential_500 \
    volumetric_soil_water_layer_1 \
    leaf_area_index_high_vegetation \
    surface_solar_radiation_downwards \
    total_sky_direct_solar_radiation_at_surface
```

This writes:
- `viz/evaluation/metrics.json`
- `viz/evaluation/rmse_by_variable.png`
- `viz/evaluation/acc_by_variable.png`
- `viz/evaluation/sample_forecast_comparison.png`

### 7. Train on multiple GPUs

```bash
torchrun --nproc_per_node=2 scripts/train.py \
  --data_mode real \
  --zarr_path data/era5_subset.zarr \
  --variables \
    2m_temperature \
    10m_u_component_of_wind \
    10m_v_component_of_wind \
    temperature_850 \
    geopotential_500 \
    volumetric_soil_water_layer_1 \
    leaf_area_index_high_vegetation \
    surface_solar_radiation_downwards \
    total_sky_direct_solar_radiation_at_surface \
  --epochs 10
```

See [docs/DISTRIBUTED_TRAINING.md](docs/DISTRIBUTED_TRAINING.md) for multi-node setup.

---

## Climate Data Pipeline

The pipeline implements the standard Earth-2 / Pangeo toolchain:

**xarray** — labelled N-D arrays with CF-convention coordinates.
Supports `sel` / `isel` by coordinate value, weighted spatial means,
and direct zarr I/O.

**zarr** — chunked, compressed cloud-native storage. Chunks of
`{time: 1, lat: -1, lon: -1}` optimize for ML sample access.
Zarr v3 default zstd compression achieves ~3–5x ratio on float32 atmospheric fields.

**xESMF** — conservative and bilinear regridding between grid
resolutions. Conservative method preserves area-averaged quantities
(critical for precipitation, flux fields). Used to move between ERA5
native 0.25°, FourCastNet 0.25°, and ClimaX 5.625° grids. 

**Dask** — lazy parallel computation for terabyte-scale datasets.
Builds task graphs without loading data, executes chunk-by-chunk.
Enables full ERA5 preprocessing on a laptop. Threaded scheduler
(`dask.config.set(scheduler='threads')`) is preferred over `LocalCluster`
on M1/8GB — lower overhead, no port conflicts.

**Cloud storage** — zarr stores read/written identically whether local,
`gs://` (GCS), or `s3://` (AWS). Real ERA5 available via Pangeo:

```python
ds = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    storage_options={"token": "anon"},
    chunks={"time": 1},
)
```

---

## Distributed Training

Uses **PyTorch DistributedDataParallel (DDP)** with the **NCCL** backend.

```
torchrun spawns one process per GPU
│
├── Each process: full model copy + non-overlapping data shard
├── Backward pass: DDP hooks trigger AllReduce via NCCL
│     gradient_avg = (grad_GPU0 + grad_GPU1 + ...) / world_size
└── All GPUs apply identical weight update → models stay in sync
```

**Why DDP over DataParallel:** DDP spawns separate processes — each has
its own Python interpreter and GIL. DataParallel uses threads that share
one GIL, creating a bottleneck. DDP's gradient sync happens at the
NCCL/GPU level, bypassing Python entirely.

The training entry point supports two modes:

- `--data_mode synthetic` for quick local smoke tests
- `--data_mode real` for training on the Zarr store produced by the ERA5 pipeline

Backend selection is automatic:

- `nccl` on CUDA machines
- `gloo` on CPU-only machines, which is useful for local smoke tests and CI-style validation

Example result on A100 using synthetic data:

```
Epoch   1/10 | Train Loss: 0.03965 | Val Loss: 0.01188 | Time: 2.7s
Epoch   2/10 | Train Loss: 0.01052 | Val Loss: 0.01030 | Time: 0.9s
...
Epoch  10/10 | Train Loss: 0.01017 | Val Loss: 0.01017 | Time: 0.9s

Training complete. Best val loss: 0.01017
```

---

## Variables

| Variable | Description | Units | Used Where |
|---|---|---|---|
| `2m_temperature` | Near-surface air temperature | K | Real ERA5 preprocessing |
| `10m_u_component_of_wind` | Eastward 10 m wind | m s-1 | Real ERA5 preprocessing |
| `10m_v_component_of_wind` | Northward 10 m wind | m s-1 | Real ERA5 preprocessing |
| `temperature_850` | Air temperature at 850 hPa | K | Real ERA5 preprocessing |
| `geopotential_500` | Geopotential at 500 hPa | m2 s-2 | Real ERA5 preprocessing |
| `volumetric_soil_water_layer_1` | Top-layer soil moisture | m3 m-3 | Real ERA5 preprocessing |
| `leaf_area_index_high_vegetation` | High-vegetation leaf area index | m2 m-2 | Real ERA5 preprocessing |
| `surface_solar_radiation_downwards` | Total downward shortwave radiation at surface | J m-2 | Real ERA5 preprocessing |
| `total_sky_direct_solar_radiation_at_surface` | Direct-beam shortwave radiation at surface | J m-2 | Real ERA5 preprocessing |
| `z500`, `t850`, `u10`, `v10` | Synthetic ERA5-like benchmark variables | mixed | Synthetic data generator |

The synthetic generator in `pipeline/data/synthetic.py` uses the more classical
weather-forecasting variables (`z500`, `t850`, `u10`, `v10`), while the current
real-data training path uses a broader mix of near-surface, upper-air, land,
and radiation variables.

---

## Related Work

- [FourCastNet](https://arxiv.org/abs/2202.11214) — Fourier Neural Operator for global weather
- [GraphCast](https://arxiv.org/abs/2212.12794) — Graph network weather forecasting (DeepMind)
- [Pangu-Weather](https://arxiv.org/abs/2211.02556) — Transformer-based NWP
- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) — Framework this work targets
- [NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio) — Inference and evaluation
