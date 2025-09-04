from fastapi import APIRouter, Query, Path, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

from utils.zarr_handler import _load_zarr
from utils.time_utils import _iso_utc_str, _iso_utc_ndarray
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/by_date")
async def get_forecast_time(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(
        ...,
        description="Base time ISO8601 (e.g., '2025-09-02T00:00:00Z').",
    ),
) -> dict:
    """
    Get forecast times for a dataset at a given base time.
    Loads the Zarr store for the given `index` and finds forecast steps
    associated with the specified `base_time`. Returns the matched base
    time and a list of forecast times in ISO8601 UTC format.
    """
    try:
        ds = _load_zarr(index)
        base_times = _iso_utc_ndarray(ds["base_time"].values)

        matches = np.where(base_times == base_time)[0]
        idx = int(matches[0])
        matched_base = base_times[idx]

        # logger.info(f"base_time: {base_time}")
        # logger.info(f"base_times: {base_times}")
        # logger.info(f"matches: {matches}")
        # logger.info(f"matched_base: {matched_base}")

        # ---- Select the row of forecast_time for this base_time --------------
        ft_row = ds["forecast_time"].isel(base_time=idx)  # dims: (forecast_index,)
        ft_vals = _iso_utc_ndarray(ft_row)

        # ---- Build output list per dataset conventions -----------------------
        seen = set()
        ordered_unique = []
        for v in ft_vals:
            if v not in seen:
                seen.add(v)
                ordered_unique.append(v)
        out_times = ordered_unique

        # logger.info(f"out_times: {out_times}")

        forecast_time = [_iso_utc_str(t) for t in out_times]

        return {
            "index": index,
            "base_time": matched_base.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast_time": forecast_time,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Failed to retrieve forecast steps (by_date)")
        return JSONResponse(status_code=400, content={"error": str(e)})
