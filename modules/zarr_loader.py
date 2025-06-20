import xarray as xr
import zarr
import numpy as np
from pathlib import Path

DATA_PATH = Path("data/nc/fopi/fopi_2024120100.nc")
ZARR_STORE = Path("data/zarr/fopi/fopi_2024120100.zarr")


def clean_time(ds, time_min=0, time_max=1e5):
    """
    Clean the 'time' coordinate in an xarray Dataset by filtering out invalid,
    missing, or extreme time values outside a specified range.

    This function removes time entries that are NaN, infinite, or fall outside
    the inclusive range defined by `time_min` and `time_max`. The Dataset is
    then subset to include only the valid times.

    Args:
        ds (xarray.Dataset): Input dataset containing a 'time' coordinate.
        time_min (float, optional): Minimum acceptable time value (inclusive).
                                    Defaults to 0.
        time_max (float, optional): Maximum acceptable time value (inclusive).
                                    Defaults to 100000.

    Returns:
        xarray.Dataset: A subset of the input dataset filtered to only valid times.
    """
    time = ds['time']
    valid_time_mask = time.where(
        time.notnull() & np.isfinite(time) & (time >= time_min) & (time <= time_max),
        drop=True
    )
    ds_cleaned = ds.sel(time=valid_time_mask['time'])
    return ds_cleaned


def convert_nc_to_zarr(force=False):
    """
    Convert a NetCDF dataset to Zarr format for optimized chunked access,
    cleaning invalid time coordinates before saving.

    This function checks if the Zarr store already exists and only performs
    the conversion if it does not exist or if `force=True` is specified.
    It disables automatic time decoding to avoid overflow errors, cleans
    the time coordinate, and then writes the cleaned dataset to Zarr format.

    Args:
        force (bool, optional): If True, forces conversion even if Zarr store
                                exists. Defaults to False.

    Returns:
        pathlib.Path: Path to the Zarr store directory.
    """
    if not ZARR_STORE.exists() or force:
        # Load NetCDF without decoding time to prevent overflow
        ds = xr.open_dataset(DATA_PATH, chunks={"time": 1, "lat": 256, "lon": 256}, decode_times=False)

        # Clean invalid time entries before anything else
        ds = clean_time(ds)

        # Write to Zarr
        ds.to_zarr(ZARR_STORE, mode="w")
    return ZARR_STORE



def load_zarr():
    """
    Load the dataset from the previously saved Zarr store.

    Opens the dataset saved in Zarr format for chunked and efficient
    access, suitable for large multidimensional data arrays.

    Returns:
        xarray.Dataset: The dataset loaded from the Zarr store.
    """
    return xr.open_zarr(ZARR_STORE)
