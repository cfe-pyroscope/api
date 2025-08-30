from fastapi import APIRouter, Query, Path, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

from utils.zarr_handler import _load_zarr
from utils.time_utils import _iso_naive_utc, _iso_utc   # ⬅️ UTC-safe output
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/by_date")
async def get_forecast_time(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(
        ...,
        description="Base time ISO8601 (e.g., '2025-07-11T00:00:00Z').",
    ),
) -> dict:
    """
    Get the available forecast steps for a given dataset and initialization time.

    Path
    ----
    GET /{index}/by_date

    Summary
    -------
    Returns the matched initialization (`base_time`) and the forecast steps
    **for that run only**, obtained by selecting the corresponding row of the
    2D `forecast_time` variable using `forecast_index`.

    Data Model (expected)
    ---------------------
    Dimensions: (forecast_index: N, lon: ..., lat: ..., base_time: M)
    Coordinates: base_time (M), lat, lon, forecast_index (0..N-1)
    Data variables:
      - forecast_time (base_time, forecast_index) datetime64[ns]
      - MODEL_FIRE   (base_time, forecast_index, lat, lon) float64

    Behavior
    --------
    - Parses `base_time` to timezone-naive, second precision.
    - Exact match on `base_time` at second precision; if not found, tries a
      00Z↔12Z fallback on the same date. If still not found, returns 404 with
      hints (first few available runs that date), all serialized via `_iso_utc`.
    - Selects the **row** of `forecast_time` for the matched `base_time`, i.e.,
      `forecast_time[matched_base, :]` along `forecast_index`.
    - For **POF**: collapses to unique daily steps (preserving order), up to ~10 days.
    - For **FOPI** (and others): returns the steps in `forecast_index` order,
      de-duplicated but not re-sorted.
    - All returned times are serialized with `_iso_utc` ("...Z").

    Errors
    ------
    - 400: Missing expected coordinates/variables or other processing errors.
    - 404: No `base_time` match (after 00Z↔12Z fallback) for the requested date.
    """
    try:
        ds = _load_zarr(index)

        # ---- Basic schema checks (fail fast) ---------------------------------
        required_dims = {"base_time", "forecast_index"}
        if not required_dims.issubset(set(ds.dims)):
            raise HTTPException(
                status_code=400,
                detail=f"Dataset missing required dims {required_dims}. Found: {set(ds.dims)}",
            )
        for coord in ("lat", "lon"):
            if coord not in ds.coords:
                raise HTTPException(
                    status_code=400,
                    detail=f"Dataset missing coordinate '{coord}'.",
                )
        if "forecast_time" not in ds:
            raise HTTPException(
                status_code=400,
                detail="Dataset missing data variable 'forecast_time' (base_time, forecast_index).",
            )
        ft_var = ds["forecast_time"]
        if not {"base_time", "forecast_index"}.issubset(set(ft_var.dims)):
            raise HTTPException(
                status_code=400,
                detail=f"'forecast_time' must have dims ('base_time','forecast_index'). Found: {ft_var.dims}",
            )

        # ---- Parse requested base_time (accepts with/without Z) --------------
        requested = _iso_naive_utc(base_time)

        # All available base_time values (normalize to naive, second precision)
        base_vals = pd.to_datetime(ds["base_time"].values)
        base_vals = pd.to_datetime([
            pd.Timestamp(bv).tz_localize(None).replace(microsecond=0) for bv in base_vals
        ])

        # Exact match?
        matches = np.where(base_vals == requested)[0]

        # Fallback 00Z <-> 12Z on same calendar date (common multi-daily runs)
        if matches.size == 0:
            alt_hour = 12 if requested.hour == 0 else 0
            alt = requested.replace(hour=alt_hour)
            matches = np.where(base_vals == alt)[0]
            if matches.size == 0:
                same_day = [bv for bv in base_vals if (bv.date() == requested.date())]
                hint_isoz = ", ".join(_iso_utc(x) for x in same_day[:3])
                detail = (
                    f"base_time '{_iso_utc(requested)}' not found. "
                    f"Tried '{_iso_utc(alt)}' as well. "
                    + (f"Available that date: {hint_isoz}" if same_day else "No base_time on that date.")
                )
                raise HTTPException(status_code=404, detail=detail)
            requested = alt

        idx = int(matches[0])
        matched_base = base_vals[idx]

        # ---- Select the row of forecast_time for this base_time --------------
        # Using isel is robust against exact datetime equality quirks.
        ft_row = ds["forecast_time"].isel(base_time=idx)  # dims: (forecast_index,)
        # Convert to naive, second precision
        ft_vals = pd.to_datetime(ft_row.values)
        ft_vals = [pd.Timestamp(x).tz_localize(None).replace(microsecond=0) for x in ft_vals]

        # ---- Build output list per dataset conventions -----------------------
        if index.lower() == "pof":
            # Collapse to unique days (preserving index order), typically ~10
            seen = set()
            daily = []
            for v in ft_vals:
                d = v.floor("D")
                if d not in seen:
                    seen.add(d)
                    daily.append(d)
                if len(daily) >= 10:
                    break
            out_times = daily
        else:
            # Keep steps in forecast_index order; deduplicate while preserving order
            seen = set()
            ordered_unique = []
            for v in ft_vals:
                if v not in seen:
                    seen.add(v)
                    ordered_unique.append(v)
            out_times = ordered_unique

        forecast_time = [_iso_utc(t) for t in out_times]

        return {
            "index": index,
            "base_time": _iso_utc(matched_base),
            "forecast_time": forecast_time,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Failed to retrieve forecast steps (by_date)")
        return JSONResponse(status_code=400, content={"error": str(e)})
