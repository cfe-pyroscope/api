import xarray as xr
import pandas as pd
import time
from config.config import settings
from config.logging_config import logger


def _load_zarr(index: str) -> xr.Dataset:
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


def _slice_field(ds, param, matched_base, matched_fcst):
    """
    Return ds[param] at the exact (base_time, forecast_index) whose forecast_time equals matched_fcst.
    """
    ds_bt = ds.sel(base_time=matched_base)  # base_time is a coord
    fcst_vals = pd.to_datetime(ds_bt["forecast_time"].values)
    fcst_vals = [pd.Timestamp(v).tz_localize(None).replace(microsecond=0) for v in fcst_vals]
    idx = next(i for i, v in enumerate(fcst_vals) if v == matched_fcst)
    return ds_bt[param].isel(forecast_index=idx)


def _select_first_param(ds: xr.Dataset) -> str:
    """
    Return the name of the first data variable in an xarray Dataset.
    Excludes 'forecast_time' as it's metadata, not a parameter to visualize.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to inspect.

    Returns
    -------
    str
        The name of the first data variable according to `ds.data_vars` ordering,
        excluding 'forecast_time'.
    """
    # Get all data variables except 'forecast_time' which is metadata
    data_vars = [var for var in ds.data_vars.keys() if var != 'forecast_time']

    if not data_vars:
        raise ValueError("Dataset has no data variables (excluding 'forecast_time').")
    return data_vars[0]
