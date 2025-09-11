from fastapi import APIRouter, Path, HTTPException
from fastapi.responses import JSONResponse
import os
import pandas as pd
import xarray as xr
from utils.time_utils import _iso_utc_str, _iso_utc_ndarray
from config.config import settings
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/latest_date", response_model=dict)
def get_latest_date(
        index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    ):
    """
    Get the latest available date for a dataset.
    Reads the Zarr store for the given `index` and finds the maximum `base_time`.
    """
    zarr_path = settings.ZARR_PATH / index / f"{index}.zarr"
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
        base_times = _iso_utc_ndarray(ds["base_time"].values)
    except KeyError:
        raise HTTPException(status_code=400, detail="Coordinate 'base_time' not found in dataset.")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Zarr file not found: {zarr_path}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    latest_ts = base_times.max()
    return {
        "latest_date": latest_ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }