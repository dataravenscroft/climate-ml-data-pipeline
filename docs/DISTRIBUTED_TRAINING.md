# Distributed Training

This repository supports multi-GPU training via **PyTorch DistributedDataParallel (DDP)**
with the **NCCL** backend.

## Architecture Overview

```
torchrun (launcher)
│
├── Rank 0 (GPU 0) ──┐
├── Rank 1 (GPU 1) ──┤── AllReduce gradient sync (NCCL) ──► averaged gradients
├── Rank 2 (GPU 2) ──┤
└── Rank 3 (GPU 3) ──┘
       │
       ▼
  Each rank trains on a non-overlapping shard of data (DistributedSampler)
  with a full copy of the model. Rank 0 logs metrics and saves checkpoints.
```

**Key design decisions:**
- **DDP over DataParallel**: DDP spawns one process per GPU (no GIL bottleneck), uses NCCL for gradient communication, and distributes memory load evenly. DataParallel is easier but slower and poorly suited for multi-node setups.
- **NCCL backend**: NVIDIA's optimized collective communication library. Uses NVLink (intra-node) and InfiniBand (inter-node) when available. Dramatically outperforms Gloo for GPU workloads.
- **DistributedSampler**: Ensures each rank receives a non-overlapping, full-coverage partition of the dataset. `set_epoch(epoch)` is called each epoch to re-shuffle differently across ranks.
- **Checkpoint portability**: Checkpoints save `model.module.state_dict()` (the underlying model, not the DDP wrapper), so they load cleanly in non-distributed inference.

## Quick Start

The training entry point supports two modes:

- `--data_mode synthetic` for fast offline testing
- `--data_mode real` for training from the local ERA5 Zarr store produced by `scripts/run_pipeline.py`

Backend selection is also configurable:

- `--backend auto` chooses `nccl` when CUDA is available and `gloo` otherwise
- `--backend gloo` is useful for CPU-only smoke tests

### Single Node, 2 GPUs (Colab Pro / Lambda / RunPod)

```bash
torchrun --nproc_per_node=2 scripts/train.py \
    --data_mode synthetic \
    --epochs 20 \
    --batch_size 16 \
    --hidden 64
```

Effective batch size = 16 × 2 = **32**

### Single Node, 1 GPU (development / testing)

```bash
torchrun --nproc_per_node=1 scripts/train.py \
    --data_mode synthetic \
    --backend gloo \
    --epochs 5
```

### Single Node, 1 GPU with real ERA5 data

```bash
torchrun --nproc_per_node=1 scripts/train.py \
    --data_mode real \
    --backend gloo \
    --zarr_path data/era5_subset.zarr \
    --variables 2m_temperature volumetric_soil_water_layer_1 leaf_area_index_high_vegetation \
    --epochs 10
```

### Multi-Node (e.g. 2 nodes × 4 GPUs = 8 GPUs total)

```bash
# Run on node 0 (master):
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=<node0_ip> \
    --master_port=29500 \
    scripts/train.py \
    --data_mode real \
    --zarr_path data/era5_subset.zarr \
    --variables 2m_temperature volumetric_soil_water_layer_1 leaf_area_index_high_vegetation

# Run on node 1:
torchrun \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=<node0_ip> \
    --master_port=29500 \
    scripts/train.py \
    --data_mode real \
    --zarr_path data/era5_subset.zarr \
    --variables 2m_temperature volumetric_soil_water_layer_1 leaf_area_index_high_vegetation
```

## Observed Speedup (2x T4, Colab Pro)

| Setup | Epoch Time | Throughput |
|---|---|---|
| 1 GPU (baseline) | ~18s | 44 samples/s |
| 2 GPU (DDP) | ~11s | 73 samples/s |
| Speedup | **1.6×** | — |

*Speedup < 2× due to communication overhead at this batch/model size. 
Larger models and batch sizes yield closer to linear scaling.*

## Batch Size and Learning Rate

When scaling to N GPUs, effective batch size increases N×. Apply the **linear scaling rule**:

```
lr_distributed = lr_single_gpu × world_size
```

With warmup (recommended for large effective batches):
```python
scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1/world_size, total_iters=5
)
```

## Real ERA5 Training Notes

When `--data_mode real` is used, `scripts/train.py`:

- opens the local Zarr store with `ERA5Dataset`
- normalizes each requested variable
- creates sliding windows of length `--seq_len`
- performs a chronological train/validation split using `--val_fraction`
- configures the ConvLSTM input and output channels from the selected variable list

The default real-data path is `data/era5_subset.zarr`, which matches the output
of `scripts/run_pipeline.py`.

## What AllReduce Does

After each `loss.backward()`, DDP hooks trigger an **AllReduce** operation:
1. Each rank computes gradients on its local data shard
2. NCCL AllReduce **sums** gradients across all ranks simultaneously
3. Each rank receives the **same averaged gradient** (`sum / world_size`)
4. `optimizer.step()` applies identical updates on all ranks → models stay in sync

This is more efficient than a parameter server because there's no single bottleneck node — all-to-all ring communication saturates available bandwidth.
