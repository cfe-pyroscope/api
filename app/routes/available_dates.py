from fastapi import APIRouter, Path, HTTPException
import xarray as xr
from utils.time_utils import _iso_utc_ndarray
from config.config import settings
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/available_dates", response_model=dict)
def fetch_available_dates(
    index: str = Path(..., description="Dataset index, e.g. 'fopi' or 'pof'."),
    ):
    """
    Get available dates for a dataset.
    Reads the Zarr store for the given dataset `index` and extracts the `base_time`
    coordinate. Returns both compact dates (`YYYY-MM-DD`) and full UTC timestamps
    (`YYYY-MM-DDTHH:MM:SSZ`).
    """
    zarr_path = settings.ZARR_PATH / index / f"{index}.zarr"
    try:
        ds = xr.open_zarr(zarr_path, consolidated=True)
        base_times = _iso_utc_ndarray(ds["base_time"].values)

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Zarr file not found: {zarr_path}")
    except KeyError:
        raise HTTPException(status_code=400, detail="Coordinate 'base_time' not found in dataset.")
    except Exception as e:
        logger.exception("❌ Failed to get available dates")
        raise HTTPException(status_code=400, detail=str(e))

    dates_compact = base_times.strftime("%Y-%m-%d").tolist()
    dates_iso_utc = base_times.strftime("%Y-%m-%dT%H:%M:%SZ").tolist()

    return {
        "available_dates": dates_compact,
        "available_dates_utc": dates_iso_utc
    }