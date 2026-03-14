from __future__ import annotations

import os
import shutil
import time

import xarray as xr


def write_local_zarr(subset: xr.Dataset, local_zarr_path: str) -> xr.Dataset:
    """Write dataset to local Zarr with ML-optimized chunking.

    Chunk layout {time: 1, lat: -1, lon: -1} is optimized for ML training:
    each chunk is one full spatial snapshot (~0.1 MB per variable).
    Clears inherited remote encodings before writing — required to avoid
    codec conflicts between ARCO ERA5's remote store and local zarr v3.
    """
    print("\n── Step 2: Writing local Zarr ──")

    ML_CHUNKS = {"time": 1, "lat": -1, "lon": -1}

    subset_chunked = subset.drop_encoding().chunk(ML_CHUNKS)

    for v in list(subset_chunked.data_vars) + list(subset_chunked.coords):
        subset_chunked[v].encoding = {}

    if os.path.exists(local_zarr_path):
        shutil.rmtree(local_zarr_path)

    print(f"  Chunk layout: {ML_CHUNKS}")
    print(f"  Writing to {local_zarr_path} ...")

    t0 = time.time()
    subset_chunked.to_zarr(local_zarr_path, mode="w", consolidated=True)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    ds_local = xr.open_zarr(local_zarr_path, consolidated=True, chunks=ML_CHUNKS)
    print(f"  Local store dims: {dict(ds_local.sizes)}")
    return ds_local
