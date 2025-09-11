from fastapi import APIRouter, Query, Path, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

from utils.zarr_handler import _load_zarr
from utils.time_utils import _normalize_times, _match_base_time, _iso_utc_str, _iso_utc_ndarray
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

        # --- 1) Normalize the requested base_time and match the actual coord label ---
        req_base, _ = _normalize_times(base_time, base_time)
        matched_base = _match_base_time(ds, req_base)

        # --- 2) Select by coord label (not by positional index) ---
        ds_bt = ds.sel(base_time=matched_base)
        ft_vals_orig = pd.to_datetime(ds_bt["forecast_time"].values)              # original labels
        ft_vals_utc = pd.to_datetime(ds_bt["forecast_time"].values, utc=True)     # for comparison/sorting

        # --- 3) Group by date: prefer 00Z if present, else earliest hour ---
        by_date = {}
        for orig, as_utc in zip(ft_vals_orig, ft_vals_utc):
            d = as_utc.date()
            if d not in by_date:
                by_date[d] = (orig, as_utc)
            else:
                best_orig, best_utc = by_date[d]
                if as_utc.hour == 0 or as_utc < best_utc:
                    by_date[d] = (orig, as_utc)

        # --- 4) Sort by date and convert to ISO UTC strings ---
        out_times = []
        for d in sorted(by_date.keys()):
            chosen_orig, _ = by_date[d]
            chosen = pd.Timestamp(chosen_orig).tz_localize(None).replace(microsecond=0)
            out_times.append(chosen)

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

