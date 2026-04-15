from __future__ import annotations

import os
import shutil

import numpy as np
import xarray as xr


def regrid(
    ds_local: xr.Dataset,
    local_regrid_path: str,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> xr.Dataset | None:
    """Regrid dataset to 1° target grid using xESMF bilinear interpolation.

    Bilinear is appropriate for state variables (temperature, soil moisture, LAI).
    Use conservative regridding for flux/precipitation variables to preserve
    area integrals (mass/energy conservation).
    """
    print("\n── Step 3: Regridding with xESMF ──")

    try:
        import xesmf as xe
    except ImportError:
        print("  xESMF not installed — skipping regrid step.")
        print("  Install: conda install -c conda-forge xesmf")
        return None

    target_grid = xr.Dataset({
        "lat": (["lat"], np.arange(lat_min, lat_max + 1.0, 1.0)),
        "lon": (["lon"], np.arange(lon_min, lon_max + 1.0, 1.0)),
    })

    print(f"  Source: {ds_local.sizes['lat']} × {ds_local.sizes['lon']} (0.25°)")
    print(f"  Target: {len(target_grid.lat)} × {len(target_grid.lon)} (1.0°)")

    state_vars = [
        "geopotential_500", "temperature_850",
        "10m_u_component_of_wind", "10m_v_component_of_wind",
    ]
    state_vars = [v for v in state_vars if v in ds_local]

    regridder = xe.Regridder(
        ds_local[state_vars],
        target_grid,
        method="bilinear",
        periodic=False,
        reuse_weights=True,
    )

    ds_regrid = regridder(ds_local[state_vars])
    print(f"  Regridded dims: {dict(ds_regrid.sizes)}")

    if os.path.exists(local_regrid_path):
        shutil.rmtree(local_regrid_path)

    ds_regrid.chunk({"time": 1, "lat": -1, "lon": -1}).to_zarr(
        local_regrid_path, mode="w", consolidated=True
    )
    print(f"  Saved: {local_regrid_path}")
    return ds_regrid
