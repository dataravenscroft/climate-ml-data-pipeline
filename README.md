# ERA5 ConvLSTM — Spatiotemporal Weather Forecasting

PyTorch ConvLSTM trained on ERA5 reanalysis data to predict atmospheric
state one timestep ahead. Includes full distributed training (DDP/NCCL)
and a production climate data pipeline (xarray · zarr · dask · xESMF).

---

## What This Is

A neural surrogate for atmospheric timestep integration — the machine
learning equivalent of the physical simulation system I built in 2006
(GCM netCDF inputs, topographic downscaling, distributed ensemble runs,
published in *Ecological Applications*). This project connects that
physical modeling background to modern deep learning tooling.

The model learns to predict the next 6-hourly atmospheric state from
6 prior timesteps, operating on a global 32×64 grid of four ERA5
variables: **Z500** (geopotential height), **T850** (temperature),
**U10** and **V10** (surface winds).

---

## Architecture

```
Input: (B, T=6, C=4, H=32, W=64)   ← 6 weather maps, 4 variables, global grid
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
grid stays intact throughout, so the model learns that pressure gradients
move eastward and cold fronts have characteristic spatial shapes.

---

## Files

| File | Description |
|---|---|
| `train_distributed.py` | ConvLSTM model + PyTorch DDP distributed training |
| `era5_workflow.py` | Full ERA5 data pipeline: xarray → zarr → regrid → Dask → cloud |
| `DISTRIBUTED_TRAINING.md` | DDP setup, launch commands, speedup results |

---

## Quickstart

### 1. Install

```bash
pip install torch xarray zarr dask distributed xesmf \
            numpy netCDF4 s3fs gcsfs numcodecs
```

> xESMF may require conda on some systems:
> `conda install -c conda-forge xesmf`

### 2. Run the data pipeline

```bash
python era5_workflow.py
```

This will:
- Build a synthetic ERA5-like xarray Dataset (Z500, T850, U10, V10)
- Convert to chunked zarr with Blosc/zstd compression
- Regrid to multiple resolutions using xESMF (bilinear + conservative)
- Run a Dask parallel preprocessing pipeline
- Connect the zarr store to a PyTorch DataLoader

### 3. Train (single GPU)

```bash
torchrun --nproc_per_node=1 train_distributed.py --epochs 10
```

### 4. Train (multi-GPU)

```bash
torchrun --nproc_per_node=2 train_distributed.py --epochs 10
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

See [DISTRIBUTED_TRAINING.md](DISTRIBUTED_TRAINING.md) for multi-node
setup and full launch commands.

---
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline
python era5_pipeline.py

## Climate Data Pipeline

`era5_workflow.py` implements the standard Earth-2 / Pangeo toolchain:

**xarray** — labelled N-D arrays with CF-convention coordinates.
Supports `sel` / `isel` by coordinate value, weighted spatial means,
and direct zarr I/O.

**zarr** — chunked, compressed cloud-native storage. Chunks of
`{time: 1, lat: -1, lon: -1}` optimize for ML sample access.
Blosc/zstd compression achieves ~3–5x ratio on float32 atmospheric fields.

**xESMF** — conservative and bilinear regridding between grid
resolutions. Conservative method preserves area-averaged quantities
(critical for precipitation, flux fields). Used to move between ERA5
native 0.25°, FourCastNet 0.25°, and ClimaX 5.625° grids.

**Dask** — lazy parallel computation for terabyte-scale datasets.
Builds task graphs without loading data, executes chunk-by-chunk.
Enables full ERA5 preprocessing on a laptop.

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

## Variables

| Variable | Description | Level | Units |
|---|---|---|---|
| Z500 | Geopotential height | 500 hPa | m |
| T850 | Temperature | 850 hPa | K |
| U10 | Eastward wind | 10 m | m/s |
| V10 | Northward wind | 10 m | m/s |

These are the standard benchmark variables used in FourCastNet,
Pangu-Weather, and NVIDIA PhysicsNeMo evaluations.

---

## Background

In 2006 I built a gridded spatially explicit climate simulation system:
GCM netCDF inputs, topographic downscaling to 28.5m resolution across
two IPCC emissions scenarios (A2 and B2), distributed ensemble runs,
output summarization pipelines. Published in *Ecological Applications*.

This project is the neural surrogate version of that work — replacing
expensive physical timestep integration with a ConvLSTM trained on
observed atmospheric trajectories. Same spatial structure, same variables,
1000× faster at inference time.

---

## Related Work

- [FourCastNet](https://arxiv.org/abs/2202.11214) — Fourier Neural Operator for global weather
- [GraphCast](https://arxiv.org/abs/2212.12794) — Graph network weather forecasting (DeepMind)
- [Pangu-Weather](https://arxiv.org/abs/2211.02556) — Transformer-based NWP
- [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) — Framework this work targets
- [NVIDIA Earth2Studio](https://github.com/NVIDIA/earth2studio) — Inference and evaluation

---

## License

MIT
