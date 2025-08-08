import xarray as xr
import numpy as np
from pathlib import Path
import re
from filelock import FileLock
import time
from sqlmodel import Session
from db.db.session import engine
import shutil
from config.logging_config import logger
from db.crud.db_operations import get_all_records
from config.config import settings


FILENAME_PATTERNS = {
    "fopi": re.compile(r"fopi_(\d{10})\.nc"),
    "pof": re.compile(r"POF_V2_(\d{4})_(\d{2})_(\d{2})_FC\.nc"),
}




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
                path = Path(settings.STORAGE_ROOT) / record.filepath
                print("***PATH", path)
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
            combined.to_zarr(zarr_store, mode="w", consolidated=True) # UserWarning: Consolidated metadata is currently not part in the Zarr format 3

            time.sleep(0.5)

    return zarr_store


with Session(engine) as session:
    convert_db_records_to_unique_zarr("fopi", session, force=True)
