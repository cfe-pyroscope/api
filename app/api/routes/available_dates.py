from fastapi import APIRouter, Path, HTTPException
from fastapi.responses import JSONResponse
import xarray as xr
import os
import pandas as pd
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
    - Converts the coordinate values to ISO 8601 date strings in `YYYY-MM-DD` format.
    - Returns them in a dictionary with the key `"available_dates"`.

    Args:
        index (str): Dataset identifier. Must be either `"fopi"` or `"pof"`.

    Returns:
        dict: A dictionary containing a list of ISO date strings representing available forecast
        initialization dates. Example:
            {
                "available_dates": [
                    "2025-07-01",
                    "2025-07-02",
                    ...
                ]
            }

    Raises:
        HTTPException: If the Zarr file is not found, or if the `base_time` coordinate is missing.
        JSONResponse: If any other error occurs while reading the file or processing the data.
    """
    try:
        zarr_path = settings.ZARR_PATH / index / f"{index}.zarr"

        if not os.path.exists(zarr_path):
            raise HTTPException(status_code=404, detail=f"Zarr file not found: {zarr_path}")

        # Open Zarr and read base_time
        ds = xr.open_zarr(zarr_path, consolidated=True)
        if "base_time" not in ds.coords:
            raise HTTPException(status_code=400, detail="Coordinate 'base_time' not found in dataset.")

        # Convert to ISO strings
        dates = pd.to_datetime(ds["base_time"].values).astype(str).tolist()
        logger.info(f"📅 Available dates for {index}: {dates}")
        return {"available_dates": dates}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Failed to get available dates")
        return JSONResponse(status_code=400, content={"error": str(e)})
