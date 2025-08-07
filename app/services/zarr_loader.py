import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import re
from filelock import FileLock
import time
from app.api.config import settings
from sqlmodel import Session
from app.api.db.session import engine
import shutil
from datetime import datetime
from app.logging_config import logger
from app.api.crud.db_operations import get_records_by_datetime


BASE_NC_PATH = Path(settings.STORAGE_ROOT)
BASE_ZARR_PATH = Path(settings.STORAGE_ROOT).joinpath("zarr")
BASE_ZARR_PATH.mkdir(parents=True, exist_ok=True)

FILENAME_PATTERNS = {
    "fopi": re.compile(r"fopi_(\d{10})\.nc"),
    "pof": re.compile(r"POF_V2_(\d{4})_(\d{2})_(\d{2})_FC\.nc"),
}

def list_all_nc_files(index: str) -> list[Path]:
    """
    Return a list of all NetCDF files for a given index that match the expected filename pattern.

    Parameters:
        index (str): Dataset identifier, must be a key in FILENAME_PATTERNS.

    Returns:
        list[Path]: List of matching NetCDF file paths.

    Raises:
        ValueError: If the index is unsupported.
        FileNotFoundError: If no matching files are found.
    """
    folder = BASE_NC_PATH / index
    pattern = FILENAME_PATTERNS.get(index)

    if not pattern:
        raise ValueError(f"No filename pattern defined for index '{index}'")

    all_files = list(folder.glob("*.nc"))
    matching_files = [f for f in all_files if pattern.match(f.name)]

    if not matching_files:
        raise FileNotFoundError(f"No matching NetCDF files found for index '{index}'")

    return matching_files


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
        raise FileNotFoundError(f"No valid NetCDF files found for index '{index}' in {folder}")

    # Sort by timestamp (string comparison) descending and return the latest file
    matched.sort(key=lambda x: x[0], reverse=True)
    return matched[0][1]


def get_nc_file_for_date(index: str, base_date: str) -> Path:
    """
    Retrieve the NetCDF file path for a given index and base date.

    Args:
        index (str): The data index or variable name to query.
        base_date (str): The ISO-formatted date string (e.g., "2025-08-07")
                         used to filter the record.

    Returns:
        Path: Full path to the corresponding NetCDF (.nc) file.

    Raises:
        Any exceptions raised by `get_records_by_datetime` or missing files are not handled here.
    """
    with Session(engine) as session:
        data_path = get_records_by_datetime(session, index, datetime.fromisoformat(base_date))
    return BASE_NC_PATH.joinpath(data_path.filepath)


def get_nc_file_for_date_old(index: str, base_date: str) -> Path:
    """
    Find and return the NetCDF (.nc) file path for a given index and base date.

    This function searches in a subdirectory of BASE_NC_PATH based on the provided
    index ('fopi' or 'pof') and looks for a file whose name matches the date-specific
    naming pattern.

    Args:
        index (str): The dataset index, either 'fopi' or 'pof'.
        base_date (str): The base date in ISO format (e.g., '2025-08-07' or full ISO timestamp).

    Returns:
        Path: The path to the matching NetCDF file.

    Raises:
        ValueError: If the provided index is unsupported.
        FileNotFoundError: If no matching file is found for the given date and index.
    """
    folder = BASE_NC_PATH / index
    files = list(folder.glob("*.nc"))

    # Normalize input date
    date_str = base_date.split("T")[0].replace("-", "")

    if index == "fopi":
        pattern = re.compile(rf"fopi_{date_str}\d{{2}}\.nc")
    elif index == "pof":
        y, m, d = date_str[:4], date_str[4:6], date_str[6:8]
        pattern = re.compile(rf"POF_V2_{y}_{m}_{d}_FC\.nc")
    else:
        raise ValueError("Unsupported index")

    for f in files:
        if pattern.match(f.name):
            return f

    raise FileNotFoundError(f"No NetCDF file found for index '{index}' and base date '{base_date}'")


def clean_time(ds, time_min=0, time_max=1e5):
    """
    Filter Dataset to include only valid time entries within a specified range.

    For 'fopi': keep forecast hours as float (0, 3, 6, ...).
    For 'pof': skip filtering.

    Returns:
        xr.Dataset
    """
    time = ds['time']

    if np.issubdtype(time.dtype, np.number):
        valid_mask = (
            time.notnull() & np.isfinite(time) & (time >= time_min) & (time <= time_max)
        )
        ds = ds.sel(time=valid_mask)
        return ds

    # For datetime64 (e.g. pof), assume all times are valid
    return ds


def convert_nc_to_zarr(index: str, base_time: str, force=False) -> Path:
    """
    Convert a NetCDF file for a specific base date to Zarr format.

    Parameters:
        index (str): Dataset identifier ('fopi' or 'pof').
        base_time (str): ISO 8601 string (e.g. "2025-06-24T22:30:00Z").
        force (bool): If True, force regeneration even if Zarr exists.

    Returns:
        Path: Path to the Zarr store directory.
    """
    base_date = pd.to_datetime(base_time).date()
    folder = BASE_NC_PATH / index

    if index == "pof":
        matching_file = get_nc_file_for_date(index, base_time)
        timestamp = pd.to_datetime(base_time).strftime("%Y%m%d") + "00"

    elif index == "fopi":
        matching_file = get_nc_file_for_date(index, base_time)
        timestamp = re.search(r"fopi_(\d{10})", matching_file.name).group(1)

    else:
        raise ValueError(f"Unsupported index: {index}")

    zarr_store = BASE_ZARR_PATH / index / f"{index}_{timestamp}.zarr"
    lock_path = zarr_store.with_suffix(".lock")

    with FileLock(lock_path):
        if not zarr_store.exists() or force:
            if force and zarr_store.exists():
                logger.info(f"🧹 Removing existing Zarr store at {zarr_store}")
                shutil.rmtree(zarr_store)

            decode_times_flag = False if index == "fopi" else True
            ds = xr.open_dataset(
                matching_file,
                chunks={"time": 1, "lat": 256, "lon": 256},
                decode_times=decode_times_flag
            )

            ds = clean_time(ds)

            if index == "fopi":
                # Ensure time is numeric (already in forecast hours), skip conversion
                time_hours = ds.time.values.astype(float)
                ds['time'] = ("time", time_hours)
                logger.info(f"🔁 Final time dtype: {ds.time.dtype}, values: {ds.time.values[:5]}")

            zarr_store.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"💾 Writing Zarr store to {zarr_store}")
            ds.to_zarr(zarr_store, mode="w", consolidated=False)

            time.sleep(0.5)

    return zarr_store



def load_zarr(index: str, base_time: str) -> xr.Dataset:
    """
    Load a Zarr store for the NetCDF file closest to the given base_time.

    Parameters:
        index (str): Dataset identifier.
        base_time (str): ISO 8601 base timestamp string.

    Returns:
        xr.Dataset: The loaded xarray dataset.
    """
    retries = 3
    delay_sec = 0.5

    for attempt in range(retries):
        try:
            path = convert_nc_to_zarr(index, base_time)
            ds = xr.open_zarr(path)
            break
        except PermissionError as e:
            logger.warning(f"🔄 Zarr file in use for index '{index}', retrying ({attempt + 1}/{retries})...")
            if attempt == retries - 1:
                logger.error(f"❌ Failed to load Zarr store for index '{index}' after {retries} attempts.")
                raise
            time.sleep(delay_sec)

    if index == "fopi" and 'lon' in ds.coords and ds.lon.max() > 180:
        lons = ds.lon.values
        shifted_lons = (lons + 180) % 360 - 180
        sort_idx = np.argsort(shifted_lons)
        ds = ds.isel(lon=sort_idx)
        ds['lon'].values[:] = shifted_lons[sort_idx]

    return ds
