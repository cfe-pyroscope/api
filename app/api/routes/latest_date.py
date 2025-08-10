from fastapi import APIRouter, Path, HTTPException
from fastapi.responses import JSONResponse
import os
import pandas as pd
import xarray as xr
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
      for the specified `index` (`pof` or `fopi`).
    - Reads the `base_time` coordinate, which contains all forecast initialization times.
    - Selects the latest date (max value) and converts it to an ISO 8601 date string
      in `YYYY-MM-DD` format.
    - Returns the latest date in a dictionary with the key `"latest_date"`.

    Args:
        index (str): Dataset identifier. Must be either `"fopi"` or `"pof"`.

    Returns:
        dict: A dictionary containing the latest available forecast initialization date.
        Example:
            {
                "latest_date": "2025-07-11"
            }

    Raises:
        HTTPException: If the Zarr file is missing or the `base_time` coordinate is absent.
        JSONResponse: If an unexpected error occurs while reading or processing the dataset.
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

        # Extract latest date and format as YYYY-MM-DD
        latest_date = pd.to_datetime(ds["base_time"].values).max().strftime("%Y-%m-%d")

        logger.info(f"📅 Latest date for {index}: {latest_date}")
        return {"latest_date": latest_date}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Failed to get latest date")
        return JSONResponse(status_code=400, content={"error": str(e)})
