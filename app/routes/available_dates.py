from fastapi import APIRouter, Path, HTTPException
from fastapi.responses import JSONResponse
import xarray as xr
import os
import pandas as pd
from utils.time_utils import _iso_utc
from config.config import settings
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/available_dates", response_model=dict)
def fetch_available_dates(
    index: str = Path(..., description="Dataset index, e.g. 'fopi' or 'pof'."),
    ):
    """
    Retrieve all available forecast initialization dates from a consolidated Zarr file.

    This endpoint:
    - Reads the `base_time` coordinate from the specified Zarr dataset (`pof.zarr` or `fopi.zarr`)
      located under `settings.ZARR_PATH/<index>/<index>.zarr`.
    - Produces **two** serialized representations of those times:
      1) `available_dates`: ISO 8601 **date-only** strings in `YYYY-MM-DD` format (back-compat).
      2) `available_dates_utc`: full ISO 8601 **UTC** strings (e.g., `2025-07-01T00:00:00Z`),
         created via `_iso_utc(...)` to normalize to UTC with a trailing `Z`.
    - Returns them in a dictionary.

    Args:
        index (str): Dataset identifier. Must be either `"fopi"` or `"pof"`.

    Returns:
        dict: A dictionary with both date-only and full-ISO UTC lists. Example:
            {
                "available_dates": [
                    "2025-07-01",
                    "2025-07-02",
                    ...
                ],
                "available_dates_utc": [
                    "2025-07-01T00:00:00Z",
                    "2025-07-02T00:00:00Z",
                    ...
                ]
            }

    Raises:
        HTTPException: If the Zarr file is not found, or if the `base_time` coordinate is missing.
        JSONResponse: If any other error occurs while reading the file or processing the data.

    Notes:
    - `base_time` values are treated as naive UTC by convention; `_iso_utc` ensures
      stable UTC serialization (`...Z`) at second precision.
    """
    try:
        zarr_path = settings.ZARR_PATH / index / f"{index}.zarr"

        if not os.path.exists(zarr_path):
            raise HTTPException(status_code=404, detail=f"Zarr file not found: {zarr_path}")

        # Open Zarr and read base_time
        ds = xr.open_zarr(zarr_path, consolidated=True)
        if "base_time" not in ds.coords:
            raise HTTPException(status_code=400, detail="Coordinate 'base_time' not found in dataset.")

        # base_time is naive UTC by convention; normalize on serialization
        base_times = pd.to_datetime(ds["base_time"].values)

        # Back-compat list (date-only)
        dates_compact = base_times.strftime("%Y-%m-%d").tolist()

        # UTC-safe full ISO list
        dates_iso_utc = [_iso_utc(bt) for bt in base_times]

        logger.info(f"📅 Available dates for {index}: {dates_compact}")
        return {
            "available_dates": dates_compact,
            "available_dates_utc": dates_iso_utc,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Failed to get available dates")
        return JSONResponse(status_code=400, content={"error": str(e)})