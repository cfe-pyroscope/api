import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import re
from filelock import FileLock
import time
from sqlmodel import Session
from db.db.session import engine
import shutil
from datetime import datetime
from logging_config import logger
from db.crud.db_operations import get_records_by_datetime, get_all_records
from config import settings


FILENAME_PATTERNS = {
    "fopi": re.compile(r"fopi_(\d{10})\.nc"),
    "pof": re.compile(r"POF_V2_(\d{4})_(\d{2})_(\d{2})_FC\.nc"),
}


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
    return settings.NC_PATH.joinpath(data_path.filepath)



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
    if index == "pof":
        matching_file = get_nc_file_for_date(index, base_time)
        timestamp = pd.to_datetime(base_time).strftime("%Y%m%d") + "00"

    elif index == "fopi":
        matching_file = get_nc_file_for_date(index, base_time)
        timestamp = re.search(r"fopi_(\d{10})", matching_file.name).group(1)

    else:
        raise ValueError(f"Unsupported index: {index}")

    zarr_store = ZARR_PATH / index / f"{index}_{timestamp}.zarr"
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


def convert_db_records_to_unique_zarr(index: str, session: Session, force: bool = False) -> Path:
    """
    Convert all NetCDF files listed in the DB for a dataset into a single Zarr store.

    Parameters:
        index (str): Dataset identifier ('fopi' or 'pof').
        session (Session): Active DB session to retrieve file paths.
        force (bool): If True, overwrite the existing Zarr store.

    Returns:
        Path: Path to the resulting Zarr store directory.
    """
    zarr_store = settings.ZARR_PATH / index / f"{index}.zarr"
    lock_path = zarr_store.with_suffix(".lock")

    # Step 1: Get all records for the dataset
    records = get_all_records(session, index)

    if not records:
        raise FileNotFoundError(f"No NetCDF records found in DB for dataset '{index}'")

    with FileLock(lock_path):
        if not zarr_store.exists() or force:
            if force and zarr_store.exists():
                logger.info(f"Removing existing Zarr store at {zarr_store}")
                shutil.rmtree(zarr_store)

            datasets = []

            # Step 2: Sort records by datetime if available
            sorted_records = sorted(records, key=lambda r: r.datetime)

            for record in sorted_records:
                path = Path(record.file_path)  # Adjust this attribute name if different
                if not path.exists():
                    logger.warning(f"File not found: {path}, skipping.")
                    continue

                decode_times_flag = False if index.lower() == "fopi" else True

                ds = xr.open_dataset(
                    path,
                    chunks={"time": 1, "lat": 256, "lon": 256},
                    decode_times=decode_times_flag
                )

                ds = clean_time(ds)

                if index.lower() == "fopi":
                    ds['time'] = ("time", ds.time.values.astype(float))

                datasets.append(ds)

            if not datasets:
                raise RuntimeError("No valid datasets to combine.")

            combined = xr.concat(datasets, dim="time")

            zarr_store.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing combined Zarr store to {zarr_store}")
            combined.to_zarr(zarr_store, mode="w", consolidated=True)

            time.sleep(0.5)

    return zarr_store

#convert_db_records_to_unique_zarr("fopi")
#convert_db_records_to_unique_zarr("pof")