from fastapi import APIRouter, Path, HTTPException
from fastapi.responses import JSONResponse
import os
import pandas as pd
import xarray as xr
from app.utils.time_utils import _iso_utc
from config.config import settings
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/latest_date", response_model=dict)
def get_latest_date(
        index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    ):
    """
    Retrieve the most recent forecast initialization date from a consolidated Zarr file.

    This endpoint:
    - Opens the Zarr dataset located at `settings.ZARR_PATH/<index>/<index>.zarr`
      for the specified `index` (`"pof"` or `"fopi"`).
    - Reads the `base_time` coordinate (all forecast initialization times) and selects the latest.
    - Produces **two** serialized representations of that timestamp:
      1) `latest_date`: ISO 8601 **date-only** string `YYYY-MM-DD` (back-compat).
      2) `latest_date_utc`: full ISO 8601 **UTC** string (e.g., `2025-07-11T00:00:00Z`),
         created via `_iso_utc(...)` to normalize to UTC with a trailing `Z`.
    - Returns both in a dictionary.

    Args:
        index (str): Dataset identifier. Must be either `"fopi"` or `"pof"`.

    Returns:
        dict: A dictionary containing both the date-only and full-ISO UTC forms. Example:
            {
                "latest_date": "2025-07-11",
                "latest_date_utc": "2025-07-11T00:00:00Z"
            }

    Raises:
        HTTPException: If the Zarr file is missing or the `base_time` coordinate is absent.
        JSONResponse: If an unexpected error occurs while reading or processing the dataset.

    Notes:
    - `base_time` values are treated as naive UTC by convention; `_iso_utc` ensures
      stable UTC serialization (with `Z`) at second precision.
    """
    try:
        if index not in ("pof", "fopi"):
            raise HTTPException(status_code=404, detail="Index must be 'pof' or 'fopi'.")

        zarr_path = settings.ZARR_PATH / index / f"{index}.zarr"
        if not os.path.exists(zarr_path):
            raise HTTPException(status_code=404, detail=f"Zarr file not found: {zarr_path}")

        ds = xr.open_zarr(zarr_path, consolidated=True)
        if "base_time" not in ds.coords:
            raise HTTPException(status_code=400, detail="Coordinate 'base_time' not found in dataset.")

        base_times = pd.to_datetime(ds["base_time"].values)
        latest_ts = base_times.max()

        latest_date = latest_ts.strftime("%Y-%m-%d")  # old format
        latest_date_utc = _iso_utc(latest_ts)  # new UTC format

        logger.info(f"📅 Latest date for {index}: {latest_date} ({latest_date_utc})")
        return {
            "latest_date": latest_date,
            "latest_date_utc": latest_date_utc,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Failed to get latest date")
        return JSONResponse(status_code=400, content={"error": str(e)})