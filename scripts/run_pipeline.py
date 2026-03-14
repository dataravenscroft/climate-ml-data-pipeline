"""Run the full ERA5 data pipeline.

Usage:
    python scripts/run_pipeline.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.data.era5 import (
    LOCAL_ZARR_PATH,
    LOCAL_REGRID_PATH,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    compute_stats,
    make_dask_client,
    open_and_subset,
)
from pipeline.data.dataset import build_dataloader
from pipeline.data.regrid import regrid
from pipeline.data.zarr_store import write_local_zarr


def main() -> None:
    print("=" * 68)
    print("ERA5 Climate Pipeline")
    print("=" * 68)

    os.makedirs("data", exist_ok=True)

    make_dask_client()

    subset   = open_and_subset(None)
    ds_local = write_local_zarr(subset, LOCAL_ZARR_PATH)
    _        = regrid(ds_local, LOCAL_REGRID_PATH, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    compute_stats(ds_local)
    loader   = build_dataloader(LOCAL_ZARR_PATH)

    print("\n" + "=" * 68)
    print("Pipeline complete.")
    print(f"  Local Zarr:   {LOCAL_ZARR_PATH}")
    print(f"  Regrid Zarr:  {LOCAL_REGRID_PATH}")
    print("=" * 68)


if __name__ == "__main__":
    main()
