import xarray as xr
import numpy as np
from pathlib import Path
import re

BASE_NC_PATH = Path("data/nc")
BASE_ZARR_PATH = Path("data/zarr")

FILENAME_PATTERNS = {
    "fopi": re.compile(r"fopi_(\d{10})\.nc"),
    "pof": re.compile(r"POF_V2_(\d{4})_(\d{2})_(\d{2})_FC\.nc"),
}


def get_latest_nc_file(index: str) -> Path:
    """
    Find the most recent NetCDF file for a given dataset index based on filename patterns.

    Parameters:
        index (str): Dataset identifier, must match a key in FILENAME_PATTERNS.

    Returns:
        Path: Path object pointing to the latest NetCDF file in BASE_NC_PATH/index.

    Raises:
        ValueError: If no filename pattern is defined for the given index.
        FileNotFoundError: If no matching NetCDF files are found in the directory.
    """
    folder = BASE_NC_PATH / index
    pattern = FILENAME_PATTERNS.get(index)

    if not pattern:
        raise ValueError(f"No filename pattern defined for index '{index}'")

    files = list(folder.glob("*.nc"))
    matched = []

    for f in files:
        m = pattern.search(f.name)
        if m:
            if index == "fopi":
                timestamp = m.group(1)
            elif index == "pof":
                y, m_, d = m.groups()
                timestamp = f"{y}{m_}{d}00"
            else:
                continue
            matched.append((timestamp, f))

    if not matched:
        raise FileNotFoundError(f"No valid NetCDF files found for index '{index}'")

    # Sort by timestamp (string comparison) descending and return the latest file
    matched.sort(key=lambda x: x[0], reverse=True)
    return matched[0][1]


def clean_time(ds, time_min=0, time_max=1e5):
    """
    Filter Dataset to include only valid time entries within a specified range.

    Parameters:
        ds (xr.Dataset): Input xarray dataset containing a 'time' coordinate.
        time_min (float): Lower bound for valid time values (inclusive).
        time_max (float): Upper bound for valid time values (inclusive).

    Returns:
        xr.Dataset: Subset of the original dataset with time values that are finite,
                    non-null, and within the specified range.
    """
    time = ds['time']
    valid_time_mask = time.where(
        time.notnull() & np.isfinite(time) & (time >= time_min) & (time <= time_max),
        drop=True
    )
    return ds.sel(time=valid_time_mask['time'])


def convert_nc_to_zarr(index: str, force=False) -> Path:
    """
    Convert the latest NetCDF file of a dataset to Zarr format, optionally forcing overwrite.

    Parameters:
        index (str): Dataset identifier ('fopi' or 'pof').
        force (bool): If True, overwrite existing Zarr store even if it exists.

    Returns:
        Path: Path to the resulting Zarr store directory.

    Raises:
        ValueError: If the timestamp extraction logic is not implemented for the given index.
    """
    nc_path = get_latest_nc_file(index)

    # Determine timestamp string based on filename
    if index == "fopi":
        timestamp = re.search(r"fopi_(\d{10})", nc_path.name).group(1)
    elif index == "pof":
        y, m, d = re.search(r"POF_V2_(\d{4})_(\d{2})_(\d{2})", nc_path.name).groups()
        timestamp = f"{y}{m}{d}00"
    else:
        raise ValueError(f"Zarr timestamp logic not implemented for index '{index}'")

    zarr_store = BASE_ZARR_PATH / index / f"{index}_{timestamp}.zarr"

    if not zarr_store.exists() or force:
        ds = xr.open_dataset(nc_path, chunks={"time": 1, "lat": 256, "lon": 256}, decode_times=False)
        ds = clean_time(ds)
        zarr_store.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(zarr_store, mode="w", consolidated=False)

    return zarr_store


def load_zarr(index: str) -> xr.Dataset:
    """
    Load a Zarr store for a dataset, converting from NetCDF if necessary, and adjust longitude for 'fopi'.

    Parameters:
        index (str): Dataset identifier ('fopi' or 'pof').

    Returns:
        xr.Dataset: The loaded xarray dataset. For 'fopi', longitudes are converted
                    from [0, 360] range to [-180, 180] and sorted accordingly.
    """
    path = convert_nc_to_zarr(index)
    ds = xr.open_zarr(path)

    # ✅ For 'fopi', convert longitudes from 0–360 to -180–180 if needed
    if index == "fopi" and 'lon' in ds.coords and ds.lon.max() > 180:
        lons = ds.lon.values
        shifted_lons = (lons + 180) % 360 - 180
        sort_idx = np.argsort(shifted_lons)
        ds = ds.isel(lon=sort_idx)
        ds['lon'].values[:] = shifted_lons[sort_idx]

    return ds