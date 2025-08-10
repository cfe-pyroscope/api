import xarray as xr
from pathlib import Path
import pandas as pd


zarr_path = Path("") # Use the output_zarr_path you used in the creation script
zarr_file = xr.open_zarr(zarr_path)

target_date = pd.Timestamp("2025-07-10")

lat_min, lat_max = 90, 70
lon_min, lon_max = -20, 10

"""
The zarr dataset has two time dims, init_time and valid_time. 
It creates all possible combinations of these two dims so when 
we filter for a specific date we need to drop all the nan values 
from the other time dim. Finally we can filter lat,lon
"""
# Filter for a specific init_time, dropna from valid_time
subset_init_time = zarr_file.sel(init_time=target_date).dropna(dim='valid_time', how='all').sel(lat=slice(lat_min, lat_max),lon=slice(lon_min, lon_max))

# Filter for a specific init_time, dropna from valid_time
subset_valid_time = zarr_file.sel(valid_time=target_date).dropna(dim='init_time', how='all').sel(lat=slice(lat_min, lat_max),lon=slice(lon_min, lon_max))