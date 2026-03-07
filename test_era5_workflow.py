# """
# era5_workflow.py
# ================
# 2026 Climate ML Toolchain — xarray · zarr · dask · xESMF · cloud storage

# Covers:
#   1. Generate synthetic ERA5-like data (or load real ERA5 via CDS API)
#   2. Load and explore with xarray
#   3. Convert to Zarr (chunked, compressed)
#   4. Regrid across resolutions with xESMF
#   5. Parallelize with Dask
#   6. Deploy to cloud object storage (S3 / GCS)

# HOW TO RUN
# ----------
# VS Code (local):
#   pip install xarray zarr dask xesmf numpy netCDF4 s3fs gcsfs cartopy matplotlib
#   python era5_workflow.py

# Colab:
#   !pip install xarray zarr dask xesmf netCDF4 s3fs gcsfs cartopy matplotlib -q
#   Then paste sections into cells, or:
#   !python era5_workflow.py

# REAL ERA5 (optional):
#   pip install cdsapi
#   Set up ~/.cdsapirc with your CDS key (free at cds.climate.copernicus.eu)
#   Uncomment the download_real_era5() section below
# """

# import numpy as np
# import xarray as xr
# import zarr
# import dask
# import dask.array as da
# from dask.distributed import Client
# import os
# import time

# # ─────────────────────────────────────────────────────────────────────────────
# # 0. SETUP
# # ─────────────────────────────────────────────────────────────────────────────

# print("=" * 60)
# print("ERA5 Climate Workflow — xarray · zarr · dask · xESMF")
# print("=" * 60)

# # Create local directories
# os.makedirs("data/netcdf",  exist_ok=True)
# os.makedirs("data/zarr",    exist_ok=True)
# os.makedirs("data/regrid",  exist_ok=True)


# # ─────────────────────────────────────────────────────────────────────────────
# # 1. CREATE SYNTHETIC ERA5-LIKE DATA
# #    (swap this for real ERA5 in production — see download_real_era5 below)
# # ─────────────────────────────────────────────────────────────────────────────

# print("\n── Step 1: Creating synthetic ERA5-like xarray Dataset ──")

# def make_era5_dataset(
#     n_times: int = 24,       # 24 timesteps = 6 days at 6-hourly
#     lat_res: float = 2.5,    # 2.5° resolution (~80km)
#     lon_res: float = 2.5,
# ) -> xr.Dataset:
#     """
#     Create a realistic ERA5-like xarray Dataset.

#     Dimensions:  time × lat × lon
#     Variables:   z500, t850, u10, v10
#     Coordinates: proper datetime + lat/lon axes with CF conventions

#     In production: replace with xr.open_dataset("era5.nc") or
#                    xr.open_zarr("gs://your-bucket/era5.zarr")
#     """
#     # Coordinate axes
#     times = xr.cftime_range("2020-01-01", periods=n_times, freq="6H")
#     lats  = np.arange(-90, 90 + lat_res, lat_res)
#     lons  = np.arange(0,  360,           lon_res)

#     n_lat, n_lon = len(lats), len(lons)
#     rng = np.random.default_rng(42)

#     # Spatial base fields — simulate planetary wave structure
#     lon_grid, lat_grid = np.meshgrid(
#         np.deg2rad(lons), np.deg2rad(lats)
#     )
#     base = np.sin(2 * lon_grid) * np.cos(lat_grid)  # wavenumber-2 structure

#     def make_field(mean, std, temporal_scale, spatial_scale=1.0):
#         """Generate realistic-looking atmospheric field."""
#         field = np.zeros((n_times, n_lat, n_lon))
#         for t in range(n_times):
#             field[t] = (
#                 mean
#                 + spatial_scale * base * std * np.cos(0.26 * t)  # temporal evolution
#                 + std * 0.3 * rng.standard_normal((n_lat, n_lon)) # noise
#             )
#         return field.astype(np.float32)

#     # Create the Dataset — mimicking real ERA5 variables
#     ds = xr.Dataset(
#         {
#             # Geopotential height at 500 hPa (m) — mid-tropospheric dynamics
#             "z500": xr.DataArray(
#                 make_field(mean=5500, std=200, temporal_scale=0.1),
#                 dims=["time", "lat", "lon"],
#                 attrs={
#                     "long_name": "Geopotential height at 500 hPa",
#                     "units": "m",
#                     "standard_name": "geopotential_height",
#                     "pressure_level": 500,
#                 }
#             ),
#             # Temperature at 850 hPa (K) — lower troposphere temperature
#             "t850": xr.DataArray(
#                 make_field(mean=280, std=15, temporal_scale=0.05),
#                 dims=["time", "lat", "lon"],
#                 attrs={
#                     "long_name": "Temperature at 850 hPa",
#                     "units": "K",
#                     "standard_name": "air_temperature",
#                     "pressure_level": 850,
#                 }
#             ),
#             # U-component of wind at 10m (m/s) — east-west surface wind
#             "u10": xr.DataArray(
#                 make_field(mean=0, std=8, temporal_scale=0.2),
#                 dims=["time", "lat", "lon"],
#                 attrs={
#                     "long_name": "10m U-component of wind",
#                     "units": "m s**-1",
#                     "standard_name": "eastward_wind",
#                 }
#             ),
#             # V-component of wind at 10m (m/s) — north-south surface wind
#             "v10": xr.DataArray(
#                 make_field(mean=0, std=6, temporal_scale=0.2),
#                 dims=["time", "lat", "lon"],
#                 attrs={
#                     "long_name": "10m V-component of wind",
#                     "units": "m s**-1",
#                     "standard_name": "northward_wind",
#                 }
#             ),
#         },
#         coords={
#             "time": times,
#             "lat":  xr.DataArray(lats, dims=["lat"],
#                         attrs={"units": "degrees_north", "standard_name": "latitude"}),
#             "lon":  xr.DataArray(lons, dims=["lon"],
#                         attrs={"units": "degrees_east", "standard_name": "longitude"}),
#         },
#         attrs={
#             "title":       "Synthetic ERA5-like reanalysis",
#             "source":      "Generated for ERA5 workflow demo",
#             "institution": "ERA5 Workflow Demo",
#             "history":     f"Created {np.datetime64('today')}",
#             "conventions": "CF-1.8",
#         }
#     )
#     return ds


# ds = make_era5_dataset()

# # ── Explore the dataset ──
# print(f"\nDataset overview:")
# print(ds)
# print(f"\nDimensions:  {dict(ds.dims)}")
# print(f"Variables:   {list(ds.data_vars)}")
# print(f"Time range:  {ds.time.values[0]} → {ds.time.values[-1]}")
# print(f"Lat range:   {float(ds.lat.min()):.1f}° → {float(ds.lat.max()):.1f}°")
# print(f"Lon range:   {float(ds.lon.min()):.1f}° → {float(ds.lon.max()):.1f}°")
# print(f"Memory:      {ds.nbytes / 1e6:.1f} MB")

# # ── Key xarray operations ──
# print("\n── xarray operations ──")

# # Select a single timestep
# snapshot = ds.isel(time=0)
# print(f"Single timestep z500: shape={snapshot.z500.shape}, "
#       f"mean={float(snapshot.z500.mean()):.1f} m")

# # Select by coordinate value (not index)
# subset = ds.sel(lat=slice(30, 60), lon=slice(0, 90))
# print(f"NH subset (30-60°N, 0-90°E): {dict(subset.dims)}")

# # Time selection
# jan = ds.sel(time=ds.time.dt.month == 1)
# print(f"January timesteps: {len(jan.time)}")

# # Compute wind speed from components
# ds["wind_speed"] = np.sqrt(ds.u10**2 + ds.v10**2)
# ds["wind_speed"].attrs = {
#     "long_name": "10m wind speed",
#     "units": "m s**-1"
# }
# print(f"Wind speed computed: max={float(ds.wind_speed.max()):.1f} m/s")

# # Spatial mean (area-weighted — important for global averages)
# weights     = np.cos(np.deg2rad(ds.lat))
# ds_weighted = ds.weighted(weights)
# global_mean_t850 = ds_weighted.mean(["lat", "lon"])["t850"]
# print(f"Global mean T850: {float(global_mean_t850.mean()):.1f} K")

# # Save as netCDF (traditional format)
# nc_path = "data/netcdf/era5_synthetic.nc"
# ds.to_netcdf(nc_path)
# print(f"\nSaved to netCDF: {nc_path} ({os.path.getsize(nc_path)/1e6:.1f} MB)")


# # ─────────────────────────────────────────────────────────────────────────────
# # 2. CONVERT TO ZARR
# #    Zarr = cloud-native chunked array storage
# #    Key advantage: read only the chunks you need — no full file download
# # ─────────────────────────────────────────────────────────────────────────────

# print("\n── Step 2: Converting to Zarr ──")

# """
# CHUNKING STRATEGY — critical for performance.

# Chunks define the unit of I/O. Choose based on your access pattern:

#   Time-series analysis  → chunk along time:  {"time": 1,  "lat": 73, "lon": 144}
#   Spatial analysis      → chunk along space: {"time": 24, "lat": 10, "lon": 10}
#   ML training (common)  → balanced:          {"time": 1,  "lat": 73, "lon": 144}

# Rule of thumb: chunks of ~10–100 MB each.
# Too small = too many I/O operations.
# Too large = loads more data than you need.
# """

# # Define chunks — 1 timestep at a time, full spatial extent
# # Good for ML: load one sample at a time
# chunks = {"time": 1, "lat": -1, "lon": -1}  # -1 = full dimension

# zarr_path = "data/zarr/era5_synthetic.zarr"

# t0 = time.time()
# ds.chunk(chunks).to_zarr(
#     zarr_path,
#     mode="w",         # overwrite if exists
#     consolidated=True # write consolidated metadata — faster remote reads
# )
# elapsed = time.time() - t0
# print(f"Written to Zarr: {zarr_path} ({elapsed:.2f}s)")

# # Read back and inspect
# ds_zarr = xr.open_zarr(zarr_path)
# print(f"Loaded from Zarr: {dict(ds_zarr.dims)}")
# print(f"Chunks: {dict(ds_zarr.z500.chunksizes)}")

# # Inspect the raw zarr store
# store    = zarr.open(zarr_path)
# z500_arr = store["z500"]
# print(f"\nZarr array info:")
# print(f"  Shape:       {z500_arr.shape}")
# print(f"  Chunks:      {z500_arr.chunks}")
# print(f"  Dtype:       {z500_arr.dtype}")
# # print(f"  Compressor:  {z500_arr.compressor}")
# # print(f"  Compression: {z500_arr.info.storage_ratio:.2f}x ratio")
# print(f"  Compressors: {z500_arr.compressors}")
# # ── Zarr compression options ──
# print("\n── Zarr compression options ──")
# """
# Zarr supports multiple compressors. For float32 climate data:

#   Blosc(cname='zstd')  → best compression ratio for climate fields
#   Blosc(cname='lz4')   → fastest, less compression
#   GZip(level=4)        → compatible with everything, slower

# Interview note: real ERA5 zarr stores use Blosc+zstd and achieve
# ~3-5x compression on float32 atmospheric fields.
# """
# from zarr.codecs import ZstdCodec

# zarr_path_compressed = "data/zarr/era5_compressed.zarr"

# encoding = {
#     var: {
#         "compressors": ZstdCodec(level=5),
#         "chunks": [1, len(ds.lat), len(ds.lon)],
#         "dtype": "float32",
#     }
#     for var in ds.data_vars
# }
# ds.chunk(chunks).to_zarr(zarr_path_compressed, mode="w", encoding=encoding, consolidated=True)
# size_nc   = sum(os.path.getsize(os.path.join(r,f)) for r,d,files in os.walk("data/netcdf") for f in files)
# size_zarr = sum(os.path.getsize(os.path.join(r,f)) for r,d,files in os.walk(zarr_path_compressed) for f in files)
# print(f"netCDF size:          {size_nc/1e6:.1f} MB")
# print(f"Zarr (zstd) size:     {size_zarr/1e6:.1f} MB")
# print(f"Compression ratio:    {size_nc/size_zarr:.1f}x")


# # ─────────────────────────────────────────────────────────────────────────────
# # 3. REGRIDDING WITH xESMF
# #    Convert between grid resolutions — critical for multi-model workflows
# # ─────────────────────────────────────────────────────────────────────────────

# print("\n── Step 3: Regridding with xESMF ──")

# try:
#     import xesmf as xe

#     """
#     REGRIDDING METHODS:

#     bilinear          → fast, smooth, good for most variables
#     conservative      → preserves area-average — use for precipitation, flux fields
#     nearest_s2d       → nearest neighbor — use for categorical/mask fields
#     patch             → higher-order, smooth — good for wind/pressure visualization

#     Interview note: conservative regridding is critical for physical consistency.
#     If you bilinearly interpolate precipitation, you can violate conservation of mass.
#     ERA5 is 0.25° (~28km). PhysicsNeMo FourCastNet uses 0.25°. ClimaX uses 5.625°.
#     Regridding between these is a daily operation on the Earth-2 team.
#     """

#     # Source grid: our 2.5° dataset
#     ds_source = ds[["z500", "t850", "u10", "v10"]]

#     # Target grid: 5.625° (FourCastNet / ClimaX resolution)
#     ds_target_coarse = xr.Dataset({
#         "lat": (["lat"], np.arange(-90, 90.1, 5.625)),
#         "lon": (["lon"], np.arange(0, 360, 5.625)),
#     })

#     # Target grid: 1.0° (higher resolution)
#     ds_target_fine = xr.Dataset({
#         "lat": (["lat"], np.arange(-90, 90.1, 1.0)),
#         "lon": (["lon"], np.arange(0, 360, 1.0)),
#     })

#     print(f"Source grid:      {len(ds.lat)}×{len(ds.lon)} (2.5°)")
#     print(f"Target coarse:    {len(ds_target_coarse.lat)}×{len(ds_target_coarse.lon)} (5.625°)")
#     print(f"Target fine:      {len(ds_target_fine.lat)}×{len(ds_target_fine.lon)} (1.0°)")

#     # Build regridder — this computes and caches interpolation weights
#     print("\nBuilding bilinear regridder (2.5° → 5.625°)...")
#     regridder_coarse = xe.Regridder(
#         ds_source, ds_target_coarse,
#         method="bilinear",
#         periodic=True,           # longitude is periodic (0° = 360°)
#         reuse_weights=False,
#     )

#     print("Building conservative regridder (2.5° → 1.0°)...")
#     regridder_fine = xe.Regridder(
#         ds_source, ds_target_fine,
#         method="conservative",
#         periodic=True,
#         reuse_weights=False,
#     )

#     # Apply regridding
#     ds_coarse = regridder_coarse(ds_source)
#     ds_fine   = regridder_fine(ds_source)

#     print(f"\nRegridded to 5.625°: {dict(ds_coarse.dims)}")
#     print(f"Regridded to 1.0°:   {dict(ds_fine.dims)}")

#     # Verify conservation — mean should be similar before/after conservative regrid
#     orig_mean  = float(ds_source["t850"].mean())
#     fine_mean  = float(ds_fine["t850"].mean())
#     coarse_mean = float(ds_coarse["t850"].mean())
#     print(f"\nT850 mean conservation check:")
#     print(f"  Original (2.5°):    {orig_mean:.4f} K")
#     print(f"  Conservative (1°):  {fine_mean:.4f} K  (diff: {abs(fine_mean-orig_mean):.4f})")
#     print(f"  Bilinear (5.625°):  {coarse_mean:.4f} K  (diff: {abs(coarse_mean-orig_mean):.4f})")

#     # Save regridded data as zarr
#     ds_coarse.chunk({"time": 1, "lat": -1, "lon": -1}).to_zarr(
#         "data/regrid/era5_5625.zarr", mode="w", consolidated=True
#     )
#     print(f"\nSaved 5.625° zarr: data/regrid/era5_5625.zarr")

# except ImportError:
#     print("xESMF not installed. Install with: pip install xesmf")
#     print("On some systems: conda install -c conda-forge xesmf")
#     print("(xESMF requires ESMF which is easier via conda)")


# # ─────────────────────────────────────────────────────────────────────────────
# # 4. DASK — LAZY PARALLEL COMPUTATION
# #    Real ERA5 is terabytes. Dask lets you describe computations on data
# #    that hasn't been loaded yet, then executes in parallel chunks.
# # ─────────────────────────────────────────────────────────────────────────────

# print("\n── Step 4: Dask parallel computation ──")

# """
# KEY CONCEPT: Lazy evaluation

# Without Dask:
#   ds = xr.open_dataset("huge_era5.nc")   # loads everything into RAM → crash
#   mean = ds.t850.mean()                  # computes on full array

# With Dask:
#   ds = xr.open_zarr("era5.zarr")         # loads nothing — just metadata
#   mean = ds.t850.mean()                  # builds a task graph — nothing computed yet
#   result = mean.compute()               # NOW it executes, chunk by chunk

# This is how you work with terabyte ERA5 datasets on a laptop.
# """

# # Start a local Dask cluster
# print("Starting local Dask cluster...")
# client = Client(n_workers=2, threads_per_worker=2, memory_limit="2GB")
# print(f"Dask dashboard: {client.dashboard_link}")

# # Load zarr as a Dask-backed xarray dataset
# ds_dask = xr.open_zarr(zarr_path, chunks={"time": 1, "lat": -1, "lon": -1})
# print(f"\nDask-backed dataset loaded (nothing in RAM yet)")
# print(f"z500 dask array: {ds_dask.z500.data}")

# # ── Lazy operations — build the task graph ──
# print("\nBuilding lazy computation graph...")

# # These operations are INSTANT — they just describe what to compute
# t850_anomaly = ds_dask.t850 - ds_dask.t850.mean("time")   # temperature anomaly
# z500_std      = ds_dask.z500.std("time")                   # temporal variability
# wind_speed    = np.sqrt(ds_dask.u10**2 + ds_dask.v10**2)  # wind magnitude

# print(f"t850_anomaly graph:  {t850_anomaly.data}")  # shows task graph, not values

# # ── Trigger computation ──
# print("\nComputing (executing task graph in parallel)...")
# t0 = time.time()

# # .compute() triggers actual execution
# anomaly_vals = t850_anomaly.compute()
# std_vals     = z500_std.compute()
# wind_vals    = wind_speed.compute()

# elapsed = time.time() - t0
# print(f"Computed in {elapsed:.2f}s")
# print(f"T850 anomaly range: {float(anomaly_vals.min()):.2f} to {float(anomaly_vals.max()):.2f} K")
# print(f"Z500 std (temporal variability): {float(std_vals.mean()):.2f} m")
# print(f"Max wind speed: {float(wind_vals.max()):.2f} m/s")

# # ── Dask for ML preprocessing ──
# print("\n── Dask ML preprocessing pipeline ──")

# def normalize_variable(da):
#     """Normalize to zero mean, unit variance — lazy operation."""
#     return (da - da.mean()) / da.std()

# # Build full preprocessing pipeline — all lazy
# ds_normalized = xr.Dataset({
#     var: normalize_variable(ds_dask[var])
#     for var in ["z500", "t850", "u10", "v10"]
# })

# # Compute and save normalized data
# print("Computing normalized dataset...")
# t0 = time.time()
# ds_normalized_computed = ds_normalized.compute()
# print(f"Normalized in {elapsed:.2f}s")
# print(f"Z500 mean after norm: {float(ds_normalized_computed.z500.mean()):.6f}  (should be ~0)")
# print(f"Z500 std after norm:  {float(ds_normalized_computed.z500.std()):.6f}   (should be ~1)")

# client.close()


# # ─────────────────────────────────────────────────────────────────────────────
# # 5. CLOUD OBJECT STORAGE
# #    Reading/writing ERA5 zarr directly from S3 or GCS
# # ─────────────────────────────────────────────────────────────────────────────

# print("\n── Step 5: Cloud object storage ──")

# """
# In production, ERA5 zarr stores live in cloud buckets.
# You read them exactly like local zarr — just swap the path for a cloud URI.

# PUBLIC ERA5 ZARR STORES (no credentials needed):
#   Google Cloud (Pangeo):
#     gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3

#   AWS (Pangeo):
#     s3://era5-pds/  (NetCDF, not zarr)

# CREDENTIALS:
#   GCS: gcloud auth application-default login
#        or set GOOGLE_APPLICATION_CREDENTIALS env var
#   S3:  aws configure
#        or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
# """

# # ── GCS example ──
# try:
#     import gcsfs
#     print("\nGCS read example (public ERA5 on Pangeo):")
#     print("  ds = xr.open_zarr(")
#     print("      'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3',")
#     print("      chunks={'time': 1},")
#     print("      storage_options={'token': 'anon'}  # public bucket")
#     print("  )")
#     print("  # This gives you REAL ERA5 at 0.25° resolution, 1940-present")

#     # Uncomment to actually connect (requires internet + gcsfs):
#     # fs = gcsfs.GCSFileSystem(token="anon")
#     # ds_era5_real = xr.open_zarr(
#     #     "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
#     #     storage_options={"token": "anon"},
#     #     chunks={"time": 1},
#     # )
#     # print(ds_era5_real)

# except ImportError:
#     print("gcsfs not installed: pip install gcsfs")

# # ── S3 example ──
# try:
#     import s3fs
#     print("\nS3 write example:")
#     print("  s3 = s3fs.S3FileSystem()")
#     print("  store = s3fs.S3Map('s3://your-bucket/era5_processed.zarr', s3=s3)")
#     print("  ds.chunk({'time':1}).to_zarr(store, mode='w', consolidated=True)")

#     # Uncomment to actually write (requires AWS credentials + bucket):
#     # s3 = s3fs.S3FileSystem()
#     # store = s3fs.S3Map("s3://your-bucket/era5_processed.zarr", s3=s3)
#     # ds.chunk({"time": 1, "lat": -1, "lon": -1}).to_zarr(
#     #     store, mode="w", consolidated=True
#     # )

# except ImportError:
#     print("s3fs not installed: pip install s3fs")

# # ── Local → Cloud pattern ──
# print("\n── Standard cloud deployment pattern ──")
# print("""
# 1. Process locally with Dask (normalize, regrid, quality control)
# 2. Write to local zarr to verify
# 3. Upload to cloud bucket:
#    ds.to_zarr("gs://your-bucket/era5_processed.zarr", mode="w")
# 4. From anywhere:
#    ds = xr.open_zarr("gs://your-bucket/era5_processed.zarr",
#                       chunks={"time": 1})
# 5. Feed directly into DataLoader:
#    sample = ds.isel(time=idx).to_array().values  # → numpy → torch tensor
# """)


# # ─────────────────────────────────────────────────────────────────────────────
# # 6. CONNECTING TO YOUR ConvLSTM DATALOADER
# # ─────────────────────────────────────────────────────────────────────────────

# print("\n── Step 6: Connecting zarr → PyTorch DataLoader ──")

# import torch
# from torch.utils.data import Dataset, DataLoader

# class ERA5ZarrDataset(Dataset):
#     """
#     Production ERA5 Dataset that reads directly from zarr.

#     Replaces the synthetic ERA5Dataset in train_distributed.py.
#     Works with local zarr or cloud zarr (gs://, s3://) — same code.
#     """

#     def __init__(self, zarr_path: str, variables: list, seq_len: int = 6):
#         """
#         Args:
#             zarr_path:  path to zarr store (local or gs:// or s3://)
#             variables:  list of variable names to load e.g. ["z500","t850","u10","v10"]
#             seq_len:    number of input timesteps
#         """
#         # Open lazily — nothing loaded into RAM yet
#         self.ds       = xr.open_zarr(zarr_path, consolidated=True)[variables]
#         self.variables = variables
#         self.seq_len  = seq_len
#         self.n_times  = len(self.ds.time)

#         # Compute normalization stats (mean/std per variable) — lazy then compute
#         print(f"  Computing normalization stats for {variables}...")
#         self.means = {v: float(self.ds[v].mean()) for v in variables}
#         self.stds  = {v: float(self.ds[v].std())  for v in variables}

#     def __len__(self):
#         return self.n_times - self.seq_len

#     def __getitem__(self, idx):
#         # Load seq_len+1 frames from zarr (only these chunks hit disk/network)
#         frames = self.ds.isel(time=slice(idx, idx + self.seq_len + 1))

#         # Stack variables into channel dimension → (T+1, C, H, W)
#         arr = np.stack([
#             (frames[v].values - self.means[v]) / self.stds[v]
#             for v in self.variables
#         ], axis=1).astype(np.float32)  # (T+1, C, H, W)

#         x = torch.from_numpy(arr[:self.seq_len])   # (T, C, H, W)
#         y = torch.from_numpy(arr[self.seq_len])    # (C, H, W)
#         return x, y


# # Test the dataset
# print("\nTesting ERA5ZarrDataset...")
# dataset = ERA5ZarrDataset(
#     zarr_path  = zarr_path,
#     variables  = ["z500", "t850", "u10", "v10"],
#     seq_len    = 6,
# )
# print(f"Dataset length: {len(dataset)} samples")
# x, y = dataset[0]
# print(f"Sample x shape: {x.shape}  (T=6, C=4, H, W)")
# print(f"Sample y shape: {y.shape}  (C=4, H, W)")
# print(f"x mean: {x.mean():.4f}  std: {x.std():.4f}  (should be ~0, ~1)")

# loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
# batch_x, batch_y = next(iter(loader))
# print(f"Batch x shape: {batch_x.shape}  (B=4, T=6, C=4, H, W)")
# print(f"Batch y shape: {batch_y.shape}  (B=4, C=4, H, W)")

# print("\n" + "="*60)
# print("Workflow complete.")
# print("="*60)
# print("""
# WHAT YOU BUILT:
#   ✓ xarray Dataset with proper CF conventions and metadata
#   ✓ Chunked zarr store with Blosc/zstd compression
#   ✓ Regridded to multiple resolutions (2.5° → 5.625°, 1.0°)
#   ✓ Dask parallel preprocessing pipeline
#   ✓ Cloud storage read/write pattern (S3 + GCS)
#   ✓ ERA5ZarrDataset connecting zarr → PyTorch DataLoader

# NEXT STEPS:
#   1. Point ERA5ZarrDataset at real ERA5:
#      zarr_path = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

#   2. Swap synthetic dataset in train_distributed.py:
#      train_dataset = ERA5ZarrDataset("data/zarr/era5_synthetic.zarr", [...])

#   3. Add to GitHub ERA5 repo — shows full production pipeline
# """)
# import xarray as xr

# ds = xr.open_zarr(
#     "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr",
#     consolidated=True,
#     storage_options={"token": "anon"},
# )

# print(ds)
# import xarray as xr

# path = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ds = xr.open_zarr(
#     path,
#     consolidated=True,
#     storage_options={"token": "anon"},
# )

# print(ds)
# print(list(ds.data_vars)[:10])


# import xarray as xr
# import zarr

# path = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ds = xr.open_zarr(
#     path,
#     consolidated=True,
#     storage_options={"token": "anon"},
# )

# print(ds.sizes)
# print(ds.coords)

import xarray as xr

path = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

ds = xr.open_zarr(
    path,
    consolidated=True,
    storage_options={"token": "anon"},
)

subset = ds[["2m_temperature"]].sel(
    time=slice("2020-01-01", "2020-01-03"),
    latitude=slice(50, 20),
    longitude=slice(230, 300),
)

print(subset)
print(subset["2m_temperature"].mean().compute())


pressure_vars = [v for v in ds.data_vars if "soil" in v or "leaf" in v or "vegetation" in v]
print(pressure_vars[:20])


# '100m_u_component_of_wind', '100m_v_component_of_wind', 
# '10m_u_component_of_neutral_wind', '10m_u_component_of_wind', 
# '10m_v_component_of_neutral_wind', '10m_v_component_of_wind', 
# '10m_wind_gust_since_previous_post_processing', 
# '2m_dewpoint_temperature', '2m_temperature', 
# 'air_density_over_the_oceans', 'angle_of_sub_gridscale_orography',
#  'anisotropy_of_sub_gridscale_orography', 'benjamin_feir_index', 
# 'boundary_layer_dissipation', 'boundary_layer_height', 'charnock', 
# 'clear_sky_direct_solar_radiation_at_surface', 'cloud_base_height', 
# 'coefficient_of_drag_with_waves', 'convective_available_potential_energy',
#  'convective_inhibition', 'convective_precipitation', 'convective_rain_rate', 
# 'convective_snowfall', 'convective_snowfall_rate_water_equivalent', 
# 'downward_uv_radiation_at_the_surface', 'duct_base_height', 
# 'eastward_gravity_wave_surface_stress', 'eastward_turbulent_surface_stress', 
# 'evaporation', 'forecast_albedo', 'forecast_logarithm_of_surface_roughness_for_heat',
#  'forecast_surface_roughness', 'fraction_of_cloud_cover', 
# 'free_convective_velocity_over_the_oceans', 'friction_velocity',
#  'geopotential', 'geopotential_at_surface', 'gravity_wave_dissipation', 
# 'high_cloud_cover', 'high_vegetation_cover', 'ice_temperature_layer_1', 
# 'ice_temperature_layer_2', 'ice_temperature_layer_3', 'ice_temperature_layer_4', 
# 'instantaneous_10m_wind_gust', 'instantaneous_eastward_turbulent_surface_stress', 
# 'instantaneous_large_scale_surface_precipitation_fraction', 
# 'instantaneous_moisture_flux', 'instantaneous_northward_turbulent_surface_stress', 
# 'instantaneous_surface_sensible_heat_flux', 'k_index', 'lake_bottom_temperature', 
# 'lake_cover', 'lake_depth', 'lake_ice_depth', 'lake_ice_temperature', 
# 'lake_mix_layer_depth', 'lake_mix_layer_temperature', 'lake_shape_factor', 
# 'lake_total_layer_temperature', 'land_sea_mask', 'large_scale_precipitation', 
# 'large_scale_precipitation_fraction', 'large_scale_rain_rate', 
# 'large_scale_snowfall', 'large_scale_snowfall_rate_water_equivalent', 
# 'leaf_area_index_high_vegetation', 'leaf_area_index_low_vegetation', 
# 'low_cloud_cover', 'low_vegetation_cover', 'maximum_2m_temperature_since_previous_post_processing', 
# 'maximum_individual_wave_height', 'maximum_total_precipitation_rate_since_previous_post_processing', 
# 'mean_boundary_layer_dissipation', 'mean_convective_precipitation_rate', 
# 'mean_convective_snowfall_rate', 'mean_direction_of_total_swell', 
# 'mean_direction_of_wind_waves', 'mean_eastward_gravity_wave_surface_stress', 
# 'mean_eastward_turbulent_surface_stress', 'mean_evaporation_rate', 
# 'mean_gravity_wave_dissipation', 'mean_large_scale_precipitation_fraction', 
# 'mean_large_scale_precipitation_rate', 'mean_large_scale_snowfall_rate', 
# 'mean_northward_gravity_wave_surface_stress', 'mean_northward_turbulent_surface_stress', 
# 'mean_period_of_total_swell', 'mean_period_of_wind_waves', 
# 'mean_potential_evaporation_rate', 'mean_runoff_rate', 'mean_sea_level_pressure', 
# 'mean_snow_evaporation_rate', 'mean_snowfall_rate', 'mean_snowmelt_rate', 
# 'mean_square_slope_of_waves', 'mean_sub_surface_runoff_rate', 
# 'mean_surface_direct_short_wave_radiation_flux', 'mean_surface_direct_short_wave_radiation_flux_clear_sky', 
# 'mean_surface_downward_long_wave_radiation_flux', 'mean_surface_downward_long_wave_radiation_flux_clear_sky', 
# 'mean_surface_downward_short_wave_radiation_flux', 'mean_surface_downward_short_wave_radiation_flux_clear_sky',
#  'mean_surface_downward_uv_radiation_flux', 'mean_surface_latent_heat_flux', 'mean_surface_net_long_wave_radiation_flux', 
# 'mean_surface_net_long_wave_radiation_flux_clear_sky', 'mean_surface_net_short_wave_radiation_flux', 
# 'mean_surface_net_short_wave_radiation_flux_clear_sky', 'mean_surface_runoff_rate', 'mean_surface_sensible_heat_flux', 
# 'mean_top_downward_short_wave_radiation_flux', 'mean_top_net_long_wave_radiation_flux', 'mean_top_net_long_wave_radiation_flux_clear_sky', 
# 'mean_top_net_short_wave_radiation_flux', 'mean_top_net_short_wave_radiation_flux_clear_sky', 
# 'mean_total_precipitation_rate', 'mean_vertical_gradient_of_refractivity_inside_trapping_layer', 'mean_vertically_integrated_moisture_divergence', 
# 'mean_wave_direction', 'mean_wave_direction_of_first_swell_partition', 'mean_wave_direction_of_second_swell_partition', 
# 'mean_wave_direction_of_third_swell_partition', 'mean_wave_period', 'mean_wave_period_based_on_first_moment', 
# 'mean_wave_period_based_on_first_moment_for_swell', 'mean_wave_period_based_on_first_moment_for_wind_waves', 
# 'mean_wave_period_based_on_second_moment_for_swell', 'mean_wave_period_based_on_second_moment_for_wind_waves', 
# 'mean_wave_period_of_first_swell_partition', 'mean_wave_period_of_second_swell_partition', 
# 'mean_wave_period_of_third_swell_partition', 'mean_zero_crossing_wave_period', 'medium_cloud_cover', 
# 'minimum_2m_temperature_since_previous_post_processing', 'minimum_total_precipitation_rate_since_previous_post_processing', 
# 'minimum_vertical_gradient_of_refractivity_inside_trapping_layer', 'model_bathymetry', 
# 'near_ir_albedo_for_diffuse_radiation', 'near_ir_albedo_for_direct_radiation', 
# 'normalized_energy_flux_into_ocean', 'normalized_energy_flux_into_waves', 
# 'normalized_stress_into_ocean', 'northward_gravity_wave_surface_stress', 
# 'northward_turbulent_surface_stress', 'ocean_surface_stress_equivalent_10m_neutral_wind_direction', 
# 'ocean_surface_stress_equivalent_10m_neutral_wind_speed', 'ozone_mass_mixing_ratio', 
# 'peak_wave_period', 'period_corresponding_to_maximum_individual_wave_height', 
# 'potential_evaporation', 'potential_vorticity', 'precipitation_type', 
# 'runoff', 'sea_ice_cover', 'sea_surface_temperature', 
# 'significant_height_of_combined_wind_waves_and_swell', 
# 'significant_height_of_total_swell', 'significant_height_of_wind_waves', 
# 'significant_wave_height_of_first_swell_partition', 'significant_wave_height_of_second_swell_partition', 
# 'significant_wave_height_of_third_swell_partition', 'skin_reservoir_content', 
# 'skin_temperature', 'slope_of_sub_gridscale_orography', 'snow_albedo', 
# 'snow_density', 'snow_depth', 'snow_evaporation', 'snowfall', 'snowmelt', 
# 'soil_temperature_level_1', 'soil_temperature_level_2', 'soil_temperature_level_3', 
# 'soil_temperature_level_4', 'soil_type', 
# 'specific_cloud_ice_water_content', 
# 'specific_cloud_liquid_water_content', 
# 'specific_humidity', 
# 'standard_deviation_of_filtered_subgrid_orography', 
# 'standard_deviation_of_orography', 'sub_surface_runoff', 
# 'surface_latent_heat_flux', 'surface_net_solar_radiation', 
# 'surface_net_solar_radiation_clear_sky', 'surface_net_thermal_radiation', 
# 'surface_net_thermal_radiation_clear_sky', 'surface_pressure', 
# 'surface_runoff', 'surface_sensible_heat_flux', 
# 'surface_solar_radiation_downward_clear_sky', 'surface_solar_radiation_downwards', 
# 'surface_thermal_radiation_downward_clear_sky', 'surface_thermal_radiation_downwards', 
# 'temperature', 'temperature_of_snow_layer', 'toa_incident_solar_radiation',
#  'top_net_solar_radiation', 'top_net_solar_radiation_clear_sky', 
# 'top_net_thermal_radiation', 'top_net_thermal_radiation_clear_sky', 
# 'total_cloud_cover', 'total_column_cloud_ice_water', 
# 'total_column_cloud_liquid_water', 'total_column_ozone', 
# 'total_column_rain_water', 'total_column_snow_water', 
# 'total_column_supercooled_liquid_water', 'total_column_water', 
# 'total_column_water_vapour', 'total_precipitation', 
# 'total_sky_direct_solar_radiation_at_surface', 'total_totals_index', 
# 'trapping_layer_base_height', 'trapping_layer_top_height', 
# 'type_of_high_vegetation', 'type_of_low_vegetation', 'u_component_of_wind', 
# 'u_component_stokes_drift', 'uv_visible_albedo_for_diffuse_radiation', 
# 'uv_visible_albedo_for_direct_radiation', 'v_component_of_wind', 
# 'v_component_stokes_drift', 'vertical_integral_of_divergence_of_cloud_frozen_water_flux', 
# 'vertical_integral_of_divergence_of_cloud_liquid_water_flux', 'vertical_integral_of_divergence_of_geopotential_flux', 
# 'vertical_integral_of_divergence_of_kinetic_energy_flux', 'vertical_integral_of_divergence_of_mass_flux', 
# 'vertical_integral_of_divergence_of_moisture_flux', 'vertical_integral_of_divergence_of_ozone_flux', 
# 'vertical_integral_of_divergence_of_thermal_energy_flux', 'vertical_integral_of_divergence_of_total_energy_flux', 
# 'vertical_integral_of_eastward_cloud_frozen_water_flux', 'vertical_integral_of_eastward_cloud_liquid_water_flux', 
# 'vertical_integral_of_eastward_geopotential_flux', 'vertical_integral_of_eastward_heat_flux', 
# 'vertical_integral_of_eastward_kinetic_energy_flux', 'vertical_integral_of_eastward_mass_flux', 
# 'vertical_integral_of_eastward_ozone_flux', 'vertical_integral_of_eastward_total_energy_flux', 
# 'vertical_integral_of_eastward_water_vapour_flux', 'vertical_integral_of_energy_conversion', 
# 'vertical_integral_of_kinetic_energy', 'vertical_integral_of_mass_of_atmosphere', 
# 'vertical_integral_of_mass_tendency', 'vertical_integral_of_northward_cloud_frozen_water_flux', 
# 'vertical_integral_of_northward_cloud_liquid_water_flux', 'vertical_integral_of_northward_geopotential_flux', 
# 'vertical_integral_of_northward_heat_flux', 'vertical_integral_of_northward_kinetic_energy_flux', 
# 'vertical_integral_of_northward_mass_flux', 'vertical_integral_of_northward_ozone_flux', 
# 'vertical_integral_of_northward_total_energy_flux', 'vertical_integral_of_northward_water_vapour_flux', 
# 'vertical_integral_of_potential_and_internal_energy', 'vertical_integral_of_potential_internal_and_latent_energy', 
# 'vertical_integral_of_temperature', 'vertical_integral_of_thermal_energy', 
# 'vertical_integral_of_total_energy', 'vertical_velocity', 'vertically_integrated_moisture_divergence', 
# 'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2', 'volumetric_soil_water_layer_3', 
# 'volumetric_soil_water_layer_4', 'wave_spectral_directional_width', 
# 'wave_spectral_directional_width_for_swell', 'wave_spectral_directional_width_for_wind_waves', 
# 'wave_spectral_kurtosis', 'wave_spectral_peakedness', 
# 'wave_spectral_skewness', 'zero_degree_level']