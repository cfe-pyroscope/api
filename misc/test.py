import xarray as xr
import numpy as np
import pandas as pd
from pathlib import Path
import re
from filelock import FileLock
import time
import shutil
from datetime import datetime
from sqlmodel import Session
from db.db.session import engine
from db.crud.db_operations import get_records_by_datetime, get_all_records
from config.config import settings
from config.logging_config import logger


def convert_db_records_to_unique_zarr(index: str, session: Session, force: bool = False) -> Path:
    """
    Converts all NetCDF records associated with a given dataset index from the database into a consolidated Zarr store.

    This function:
    - Retrieves all records associated with the provided `index` from the database.
    - Loads the corresponding NetCDF files from disk.
    - Optionally filters out invalid data (e.g., values not in the [0, 1] range).
    - Concatenates the datasets along the time dimension.
    - Writes the combined data to a Zarr store on disk.
    - Ensures safe concurrent access via a file lock.

    Args:
        index (str): The dataset identifier used to query the database and define storage paths.
        session (Session): An active SQLAlchemy session used to query the database.
        force (bool, optional): If True, overwrites the existing Zarr store if it exists. Defaults to False.

    Returns:
        Path: The path to the resulting Zarr store.

    Raises:
        FileNotFoundError: If no matching records are found in the database.
        RuntimeError: If no valid datasets could be processed or combined.
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

                # ds = clean_time(ds)

                if index.lower() == "fopi":
                    ds['time'] = ("time", ds.time.values.astype(float))

                # --- Filter: Keep only values between 0 and 1 (inclusive) ---
                var_name = settings.VAR_NAMES[index]
                if var_name in ds:
                    da = ds[var_name]
                    valid_mask = (da >= 0) & (da <= 1) & ~np.isnan(da)
                    ds[var_name] = da.where(valid_mask)

                    # drop all time slices that are now entirely NaN
                    ds = ds.dropna(dim="time", how="all", subset=[var_name])

                datasets.append(ds)

            if not datasets:
                raise RuntimeError("No valid datasets to combine.")

            combined = xr.concat(datasets, dim="time")

            zarr_store.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing filtered Zarr store to {zarr_store}")
            combined.to_zarr(zarr_store, mode="w", consolidated=True)

            time.sleep(0.5)

    return zarr_store


# with Session(engine) as session:
    # convert_db_records_to_unique_zarr("fopi", session, force=True)
    # convert_db_records_to_unique_zarr("pof", session, force=True)
