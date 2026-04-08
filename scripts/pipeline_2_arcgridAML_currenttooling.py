import xarray as xr
import numpy as np
#
import py3dep
import pydaymet as daymet
# echo "machine urs.earthdata.nasa.gov login cravenscroft password 
#   "5R#/nms&^Qctww3"" > ~/.netrc                                                 
#   chmod 600 ~/.netrc

# Open gridded datasets
print("opening gridded datasets")

BBOX  = (-75.0, 40.0, -70.0, 45.0)   # smaller test bbox — northeast US
DATES = ("2020-01-01", "2020-01-07")  # 1 week for fast test

# Precipitation + temperature via Daymet V4 (~1 km daily)
_climate = daymet.get_bygeom(BBOX, dates=DATES, variables=["prcp", "tmin", "tmax"])
ppt  = _climate["prcp"]                            # precipitation  (time, y, x)  mm/day
temp = (_climate["tmin"] + _climate["tmax"]) / 2  # mean temperature (time, y, x)  °C

# DEM / elevation via USGS 3DEP (30 m, static)
dem = py3dep.get_dem(BBOX, resolution=500)          # elevation  (y, x)  metres


# Regrid or interpolate climate variables to DEM grid
# In real workflows, xESMF is often used for proper regridding.
ppt_on_dem = ppt.interp(
    lat=dem.lat,
    lon=dem.lon,
    method="linear"
)

temp_on_dem = temp.interp(
    lat=dem.lat,
    lon=dem.lon,
    method="linear"
)

# Example derived quantity
heat_index = temp_on_dem * 1.2
water_balance = ppt_on_dem - heat_index

# Conditional raster logic: stress is positive where balance is negative
stress = xr.where(water_balance < 0, np.abs(water_balance), 0)

# Carry metadata
stress.name = "water_stress"
stress.attrs["description"] = "Simple water stress proxy from precip and temperature"

# Save result
stress.to_netcdf("water_stress.nc")