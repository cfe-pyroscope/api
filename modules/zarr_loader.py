import xarray as xr
import zarr
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/nc/fopi_2024120100.nc")
ZARR_STORE = Path("data/zarr/fopi_2024120100.zarr")


def clean_time(ds, time_min=0, time_max=1e5):
    """
    Clean the 'time' coordinate by removing or fixing invalid or extreme values.
    """
    time = ds['time']
    valid_time_mask = time.where(
        time.notnull() & np.isfinite(time) & (time >= time_min) & (time <= time_max),
        drop=True
    )
    ds_cleaned = ds.sel(time=valid_time_mask['time'])
    return ds_cleaned


def convert_nc_to_zarr(force=False):
    if not ZARR_STORE.exists() or force:
        # Load NetCDF without decoding time to prevent overflow
        ds = xr.open_dataset(DATA_PATH, chunks={"time": 1, "lat": 256, "lon": 256}, decode_times=False)

        # Clean invalid time entries before anything else
        ds = clean_time(ds)

        # Write to Zarr
        ds.to_zarr(ZARR_STORE, mode="w")
    return ZARR_STORE



def load_zarr():
    return xr.open_zarr(ZARR_STORE)
