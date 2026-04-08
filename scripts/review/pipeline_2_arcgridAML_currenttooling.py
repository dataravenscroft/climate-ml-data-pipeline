import xarray as xr
import numpy as np

# Open gridded datasets
# print("opening gridded datasets")

# # Bounding box — northeast US
# LAT  = slice(45.0, 40.0)          # gridMET stores lat descending
# LON  = slice(-75.0, -70.0)
# TIME = slice("2020-01-01", "2020-01-07")

# # gridMET public OPeNDAP — no auth required
# GRIDMET = "http://thredds.northwestknowledge.net:8080/thredds/dodsC"

# print("  fetching gridMET precipitation …")
# ppt  = xr.open_dataset(f"{GRIDMET}/agg_met_pr_1979_CurrentYear_CONUS.nc",
#                         engine="netcdf4")["precipitation_amount"].sel(
#                             lat=LAT, lon=LON, day=TIME)          # (time, lat, lon) mm/day

# print("  fetching gridMET temperature …")
# tmin = xr.open_dataset(f"{GRIDMET}/agg_met_tmmn_1979_CurrentYear_CONUS.nc",
#                         engine="netcdf4")["daily_minimum_temperature"].sel(
#                             lat=LAT, lon=LON, day=TIME)          # (time, lat, lon) K
# tmax = xr.open_dataset(f"{GRIDMET}/agg_met_tmmx_1979_CurrentYear_CONUS.nc",
#                         engine="netcdf4")["daily_maximum_temperature"].sel(
#                             lat=LAT, lon=LON, day=TIME)          # (time, lat, lon) K
# temp = ((tmin + tmax) / 2) - 273.15                              # (time, lat, lon) °C

# # DEM — synthetic placeholder on the ppt grid (replace with real SRTM/ETOPO when available)
# # Realistic range for northeast US: Appalachians ~0–1200 m, coast near 0
# print("  building synthetic DEM (placeholder) …")
# _lats = ppt.lat.values
# _lons = ppt.lon.values
# _lon2d, _lat2d = np.meshgrid(_lons, _lats)
# _elev = np.clip(800 * np.exp(-0.3 * (_lon2d + 74) ** 2) + 20, 0, 1200)  # (lat, lon) metres
# dem = xr.DataArray(_elev, coords={"lat": _lats, "lon": _lons},
#                    dims=["lat", "lon"], attrs={"units": "m", "note": "synthetic placeholder"})


# # Regrid or interpolate climate variables to DEM grid
# # In real workflows, xESMF is often used for proper regridding.
# ppt_on_dem = ppt.interp(
#     lat=dem.lat,
#     lon=dem.lon,
#     method="linear"
# )

# temp_on_dem = temp.interp(
#     lat=dem.lat,
#     lon=dem.lon,
#     method="linear"
# )

# # Example derived quantity
# heat_index = temp_on_dem * 1.2
# water_balance = ppt_on_dem - heat_index

# # Conditional raster logic: stress is positive where balance is negative
# stress = xr.where(water_balance < 0, np.abs(water_balance), 0)

# # Carry metadata
# stress.name = "water_stress"
# stress.attrs["description"] = "Simple water stress proxy from precip and temperature"

# # Save result
# stress.to_netcdf("data/water_stress.nc")

### Alternative — use local ERA5 zarr from run_pipeline.py
ds = xr.open_zarr("data/zarr/era5_real_subset.zarr", consolidated=False)

subset = ds[["2m_temperature", "volumetric_soil_water_layer_1", "leaf_area_index_high_vegetation"]].sel(
    time=slice("2020-06-01", "2020-06-14"),
    lat=slice(45, 25),
    lon=slice(235, 295),
)

ppt  = subset["volumetric_soil_water_layer_1"]   # soil moisture proxy  (time, lat, lon)
temp = subset["2m_temperature"] - 273.15         # temperature °C       (time, lat, lon)

# Synthetic DEM on the ERA5 grid (placeholder — replace with real SRTM/ETOPO)
import numpy as np
_lon2d, _lat2d = np.meshgrid(subset.lon.values, subset.lat.values)
_elev = np.clip(800 * np.exp(-0.3 * (_lon2d - 280) ** 2) + 20, 0, 1200)  # (lat, lon) metres
dem = xr.DataArray(_elev, coords={"lat": subset.lat, "lon": subset.lon},
                   dims=["lat", "lon"], attrs={"units": "m", "note": "synthetic placeholder"})

# Regrid climate to DEM grid (same grid here, interp is a no-op)
ppt_on_dem  = ppt.interp(lat=dem.lat, lon=dem.lon, method="linear")
temp_on_dem = temp.interp(lat=dem.lat, lon=dem.lon, method="linear")

# Derived quantity
heat_index    = temp_on_dem * 1.2
water_balance = ppt_on_dem - heat_index
stress        = xr.where(water_balance < 0, np.abs(water_balance), 0)

stress.name = "water_stress"
stress.attrs["description"] = "Simple water stress proxy from soil moisture and temperature"

stress.to_netcdf("data/water_stress.nc")
print("saved → data/water_stress.nc")