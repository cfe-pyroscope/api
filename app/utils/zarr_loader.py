import xarray as xr
import numpy as np
import pandas as pd
import time
from config.config import settings
from config.logging_config import logger


def _load_zarr(index: str, base_time: str) -> xr.Dataset:
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
            path = settings.ZARR_PATH / index / f"{index}.zarr"
            try:
                ds = xr.open_zarr(path, consolidated=True)
            except Exception:
                ds = xr.open_zarr(path, consolidated=False)
            break
        except PermissionError as e:
            logger.warning(f"🔄 Zarr file in use for index '{index}', retrying ({attempt + 1}/{retries})...")
            if attempt == retries - 1:
                logger.error(f"❌ Failed to load Zarr store for index '{index}' after {retries} attempts.")
                raise
            time.sleep(delay_sec)

    return ds

