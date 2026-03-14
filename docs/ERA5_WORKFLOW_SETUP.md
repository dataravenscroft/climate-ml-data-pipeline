# ERA5 Workflow — Setup & Run Instructions

## Option A: Run Locally in VS Code

### 1. Install dependencies
Open VS Code terminal (Ctrl+` or Cmd+`) and run:

```bash
pip install xarray zarr dask distributed xesmf numpy netCDF4 \
            s3fs gcsfs numcodecs torch cartopy matplotlib
```

> **Note on xESMF:** If pip install fails, use conda instead:
> ```bash
> conda install -c conda-forge xesmf
> ```
> xESMF depends on ESMF (Earth System Modeling Framework) which
> is easier to install via conda than pip.

### 2. Run the script
```bash
python era5_workflow.py
```

### 3. What gets created locally
```
data/
  netcdf/era5_synthetic.nc        ← traditional netCDF file
  zarr/era5_synthetic.zarr/       ← chunked zarr store
  zarr/era5_compressed.zarr/      ← zarr with Blosc/zstd compression
  regrid/era5_5625.zarr/          ← regridded to 5.625° resolution
```

---

## Option B: Run in Colab

### 1. Install dependencies (Cell 1)
```python
!pip install xarray zarr dask distributed xesmf netCDF4 \
             s3fs gcsfs numcodecs torch -q
```

### 2. Upload the script (Cell 2)
```python
from google.colab import files
files.upload()  # select era5_workflow.py
```

### 3. Run it (Cell 3)
```python
!python era5_workflow.py
```

### OR run section by section
Paste each numbered section (Step 1 through Step 6) into
separate cells — easier to follow and re-run individual parts.

---

## Connecting to REAL ERA5 (free, public)

The Pangeo project hosts real ERA5 on Google Cloud Storage
as a public zarr store — no credentials needed:

```python
import xarray as xr

# 0.25° global ERA5, 1940-present, all pressure levels
ds_real = xr.open_zarr(
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    storage_options={"token": "anon"},
    chunks={"time": 1},
)
print(ds_real)
```

Then plug into ERA5ZarrDataset at the bottom of the script:
```python
dataset = ERA5ZarrDataset(
    zarr_path="gs://gcp-public-data-arco-era5/ar/...",
    variables=["z500", "t850", "u10", "v10"],
    seq_len=6,
)
```

---

## What each step teaches you

| Step | Tool | What you learn |
|---|---|---|
| 1 | xarray | Labelled dimensions, CF conventions, sel/isel, weighted mean |
| 2 | zarr | Chunking strategy, compression, cloud-native storage |
| 3 | xESMF | Bilinear vs conservative regridding, why it matters physically |
| 4 | Dask | Lazy evaluation, task graphs, parallel compute on large data |
| 5 | s3fs/gcsfs | Reading/writing zarr to S3 and GCS |
| 6 | PyTorch | Connecting zarr store directly to DataLoader |

---

## Interview answer once done

> "I built a full ERA5 preprocessing pipeline:
> loaded gridded atmospheric fields with xarray using CF-convention
> coordinates, converted to chunked zarr with Blosc compression,
> regridded between resolutions using xESMF conservative interpolation,
> parallelized preprocessing with Dask lazy evaluation, and connected
> the zarr store directly to a PyTorch DataLoader for ConvLSTM training.
> The same code works on local files or gs:// / s3:// cloud stores
> without modification — which is how the Earth-2 team operates."
