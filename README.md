# ERA5 Climate Pipeline + ConvLSTM Training

PyTorch ConvLSTM scaffolding alongside an ERA5 reanalysis preprocessing
pipeline built with xarray, zarr, dask, and xESMF. The repository currently
has two connected but distinct workflows:

- a real-data ERA5 preprocessing pipeline over a CONUS subset
- a ConvLSTM training pipeline that currently trains on synthetic ERA5-like data

---

## What This Is

This repo is a climate ML sandbox that connects Earth-system data engineering
to deep learning tooling.

Today, the implemented ERA5 preprocessing pipeline pulls three real variables
from the public ARCO ERA5 archive over a fixed CONUS region and two-week time
window:

- `2m_temperature`
- `volumetric_soil_water_layer_1`
- `leaf_area_index_high_vegetation`

Those fields are written to a local Zarr store, optionally regridded to 1 degree,
and exposed through a PyTorch `Dataset`/`DataLoader`.

The modeling side of the repo implements a ConvLSTM forecaster and distributed
training utilities. Training currently uses an in-memory synthetic ERA5-like
dataset so the model stack can be exercised without downloading real data.

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
│       └── distributed.py       # DDP utilities + training/validation loops
├── scripts/
│   ├── run_pipeline.py                        # entry point: python scripts/run_pipeline.py
│   ├── pipeline_2_arcgridAML_currenttooling.py  # wrapper for water stress demo
│   └── train.py                               # entry point: torchrun scripts/train.py
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

## ConvLSTM Forward Pass

The ConvLSTM implementation is generic over channel count and grid size.
In this repo's synthetic training setup, the default shape is:

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
Input: (B, T=6, C=4, H=32, W=64)   ← default synthetic training setup
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

### 2. Run the data pipeline (real ARCO ERA5)

```bash
python scripts/run_pipeline.py
```

This will:
- Open ARCO ERA5 lazily from public GCS (no credentials needed)
- Subset to CONUS, 2 weeks of hourly data
- Select `2m_temperature`, soil moisture layer 1, and high-vegetation LAI
- Write local zarr with ML-optimized chunking (`{time: 1, lat: -1, lon: -1}`)
- Regrid to 1° using xESMF bilinear interpolation #Unable to install 
- Compute Dask parallel temporal statistics
- Connect the zarr store to a PyTorch DataLoader

### 3. Run the northeast US water stress pipeline

```bash
python scripts/pipeline_2_arcgridAML_currenttooling.py
```

This will:
- Load ERA5 surface variables from the local zarr store written by the main pipeline
  (`data/era5_subset.zarr`)
- Build a simple synthetic DEM placeholder on the ERA5 grid
- Compute a toy water stress index from soil moisture and temperature:
  `stress = max(0, -(soil_moisture - 1.2 × temperature_c))`
- Save result to `data/water_stress.nc`

---

### 4. Train (single GPU)

```bash
torchrun --nproc_per_node=1 scripts/train.py --epochs 10
```

### 5. Train (multi-GPU)

```bash
torchrun --nproc_per_node=2 scripts/train.py --epochs 10
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

Results on A100 (single GPU, synthetic data):

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
| `volumetric_soil_water_layer_1` | Top-layer soil moisture | m3 m-3 | Real ERA5 preprocessing |
| `leaf_area_index_high_vegetation` | High-vegetation leaf area index | m2 m-2 | Real ERA5 preprocessing |
| `z500`, `t850`, `u10`, `v10` | Synthetic ERA5-like benchmark variables | mixed | Synthetic data generator |

The synthetic generator in `pipeline/data/synthetic.py` uses the more classical
weather-forecasting variables (`z500`, `t850`, `u10`, `v10`), while the current
real-data pipeline is focused on surface and land variables.

---

## Background

What the ConvLSTM is good at:

- `Conv2d` layers learn local spatial structure such as fronts, gradients, and
  terrain-linked patterns
- recurrent state carries temporal information such as persistence and lagged response

What it is not optimized for:

- modern global weather models like FourCastNet and GraphCast generally scale
  better to long-range, global circulation dynamics than ConvLSTM
- ConvLSTM processes timesteps sequentially, which limits parallelism


---

## Related Work

- [FourCastNet](https://arxiv.org/abs/2202.11214) — Fourier Neural Operator for global weather
- [GraphCast](https://arxiv.org/abs/2212.12794) — Graph network weather forecasting (DeepMind)
- [Pangu-Weather](https://arxiv.org/abs/2211.02556) — Transformer-based NWP
- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) — Framework this work targets
- [NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio) — Inference and evaluation
