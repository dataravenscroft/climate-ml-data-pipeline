"""Pipeline 2 — Northeast US water stress from ERA5 zarr.

Loads local ERA5 zarr store (written by run_pipeline.py), computes a simple
water stress index from soil moisture and temperature, and saves to NetCDF.

Usage:
    python scripts/review/pipeline_2_arcgridAML_currenttooling.py

Output:
    data/water_stress.nc
"""

import numpy as np
import xarray as xr

# ── Load ERA5 zarr ─────────────────────────────────────────────────────────────
print("opening ERA5 zarr store …")
ds = xr.open_zarr("data/zarr/era5_real_subset.zarr", consolidated=False)

subset = ds[["2m_temperature", "volumetric_soil_water_layer_1", "leaf_area_index_high_vegetation"]].sel(
    time=slice("2020-06-01", "2020-06-14"),
    lat=slice(45, 25),
    lon=slice(235, 295),
)

ppt  = subset["volumetric_soil_water_layer_1"]  # soil moisture proxy  (time, lat, lon)
temp = subset["2m_temperature"] - 273.15        # temperature          (time, lat, lon) °C

# ── Synthetic DEM on ERA5 grid ─────────────────────────────────────────────────
# Placeholder — replace with real SRTM/ETOPO when available.
# Shape approximates Appalachian ridge (~800 m) tapering toward coast.
_lon2d, _lat2d = np.meshgrid(subset.lon.values, subset.lat.values)
_elev = np.clip(800 * np.exp(-0.3 * (_lon2d - 280) ** 2) + 20, 0, 1200)  # (lat, lon) metres
dem = xr.DataArray(
    _elev,
    coords={"lat": subset.lat, "lon": subset.lon},
    dims=["lat", "lon"],
    attrs={"units": "m", "note": "synthetic placeholder"},
)

# ── Regrid climate to DEM grid ─────────────────────────────────────────────────
# Grids are identical here so interp is a no-op — kept for pipeline generality.
ppt_on_dem  = ppt.interp(lat=dem.lat, lon=dem.lon, method="linear")   # (time, lat, lon)
temp_on_dem = temp.interp(lat=dem.lat, lon=dem.lon, method="linear")  # (time, lat, lon)

# ── Water stress index ─────────────────────────────────────────────────────────
# Stress is positive where (precip - 1.2 × temp) is negative.
heat_index    = temp_on_dem * 1.2
water_balance = ppt_on_dem - heat_index
stress        = xr.where(water_balance < 0, np.abs(water_balance), 0)  # (time, lat, lon)

stress.name = "water_stress"
stress.attrs["description"] = "Simple water stress proxy from soil moisture and temperature"

# ── Save ───────────────────────────────────────────────────────────────────────
stress.to_netcdf("data/water_stress.nc")
print("saved → data/water_stress.nc")
