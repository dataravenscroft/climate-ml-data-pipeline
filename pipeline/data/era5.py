from __future__ import annotations

import time
import warnings

import dask
import xarray as xr

warnings.filterwarnings("ignore", category=UserWarning)

# ─── CONFIG ───────────────────────────────────────────────────────────────────

ARCO_ERA5_PATH = (
    "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
)

VARIABLES = [
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "leaf_area_index_high_vegetation",
]

TIME_START = "2020-06-01"
TIME_END   = "2020-06-14"   # 2 weeks = 336 hourly timesteps

# CONUS bounding box (ERA5 uses 0–360 longitude)
LAT_MAX =  50.0
LAT_MIN =  25.0
LON_MIN = 235.0   # ~125°W
LON_MAX = 295.0   # ~65°W

LOCAL_ZARR_PATH   = "data/era5_subset.zarr"
LOCAL_REGRID_PATH = "data/era5_subset_1deg.zarr"


# ─── DASK ──────────────────────────────────────────────────────────────────────

def make_dask_client():
    """Configure Dask threaded scheduler — tuned for M2/32GB.

    Threaded scheduler is preferred over LocalCluster for this workload:
    threads release the GIL for numpy/zarr I/O so parallelism works, and
    there's no subprocess/port overhead. 8 workers matches M2 performance
    cores. LocalCluster + dashboard (localhost:8787) is an option if you
    want task-graph visibility, but adds ~500MB overhead.
    """
    dask.config.set(scheduler="threads", num_workers=8)
    print("  Dask scheduler: threaded (8 workers, M2/32GB)")
    return None


# ─── STEP 1: OPEN & SUBSET ─────────────────────────────────────────────────────

def open_and_subset(client) -> xr.Dataset:
    """Open ARCO ERA5 lazily from GCS and subset to configured region/time.

    xr.open_zarr is lazy — nothing is downloaded until .compute() is called.
    ERA5 latitude is stored descending (90 → -90), so slice(lat_max, lat_min).
    """
    print("\n── Step 1: Opening ARCO ERA5 (lazy) ──")

    ds = xr.open_zarr(
        ARCO_ERA5_PATH,
        consolidated=True,
        storage_options={"token": "anon"},
        chunks={"time": 1},
    )

    print(f"  Remote vars available: {list(ds.data_vars)[:8]} ...")
    print(f"  Subsetting: {TIME_START} → {TIME_END}, lat [{LAT_MIN},{LAT_MAX}], lon [{LON_MIN},{LON_MAX}]")

    subset = ds[VARIABLES].sel(
        time=slice(TIME_START, TIME_END),
        latitude=slice(LAT_MAX, LAT_MIN),
        longitude=slice(LON_MIN, LON_MAX),
    )

    # Rename to lat/lon — xESMF and most ML libraries expect these names
    subset = subset.rename({"latitude": "lat", "longitude": "lon"})

    # Derived variable — stays lazy (no .compute() yet)
    subset["t2m_celsius"] = subset["2m_temperature"] - 273.15
    subset["t2m_celsius"].attrs = {"long_name": "2m temperature", "units": "degC"}

    print(f"  Subset dims: {dict(subset.sizes)}")
    _print_memory_estimate(subset)

    return subset


def _print_memory_estimate(ds: xr.Dataset) -> None:
    """Rough in-memory size if fully loaded — useful sanity check on M1."""
    n_cells = 1
    for v in ds.sizes.values():
        n_cells *= v
    mb = n_cells * len(ds.data_vars) * 4 / 1e6
    print(f"  Estimated full load size: {mb:.0f} MB across {len(ds.data_vars)} variables")
    if mb > 2000:
        print("  >2GB — consider narrowing TIME_END or bbox before computing")


# ─── STEP 4: DASK STATS ────────────────────────────────────────────────────────

def compute_stats(ds_local: xr.Dataset) -> None:
    """Build lazy computation graphs then compute all in one Dask pass.

    Building all graphs before computing lets Dask fuse operations across
    chunks, avoiding redundant reads.
    """
    print("\n── Step 4: Dask parallel stats ──")

    temp_mean = (ds_local["2m_temperature"] - 273.15).mean("time")
    soil_std  = ds_local["volumetric_soil_water_layer_1"].std("time")
    lai_max   = ds_local["leaf_area_index_high_vegetation"].max("time")

    print("  Lazy graphs built — computing now ...")
    t0 = time.time()
    temp_mean_val, soil_std_val, lai_max_val = dask.compute(
        temp_mean, soil_std, lai_max
    )
    elapsed = time.time() - t0

    print(f"  Computed in {elapsed:.2f}s")
    print(f"  Mean 2m temp (°C):          {float(temp_mean_val.mean()):.2f}")
    print(f"  Soil moisture temporal std:  {float(soil_std_val.mean()):.4f}")
    print(f"  LAI high veg max:            {float(lai_max_val.mean()):.4f}")
