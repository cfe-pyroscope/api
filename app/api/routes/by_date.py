from fastapi import APIRouter, Query, Path, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np

from app.utils.zarr_loader import _load_zarr
from app.utils.time_utils import _parse_naive, _iso_utc   # ⬅️ use UTC-safe output
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/by_date")
async def get_forecast_steps(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(
        ...,
        description="Base time ISO8601 (e.g., '2025-07-11T00:00:00' or '2025-07-11T00:00:00Z').",
    ),
) -> dict:
    """
    Get the available forecast steps for a given dataset and initialization time.

    Path
    ----
    GET /{index}/by_date

    Summary
    -------
    Returns the grid center location, the matched initialization (`base_time`), and
    a list of forecast step times, all serialized as ISO 8601 UTC strings via
    `_iso_utc` (e.g., "2025-07-11T00:00:00Z").

    Parameters
    ----------
    index : str (path)
        Dataset identifier, e.g. "fopi" or "pof".
    base_time : str (query)
        Requested initialization time in ISO 8601 form, with or without trailing
        'Z' (e.g., "2025-07-11T00:00:00" or "2025-07-11T00:00:00Z").
        Parsed with `_parse_naive` to a timezone-naive, second-precision timestamp.

    Behavior
    --------
    - Validates required coordinates: `base_time`, `forecast_time`, `lat`, `lon`.
    - Matches `base_time` exactly at second precision; if not found, tries a
      00Z↔12Z fallback on the same calendar date. If still not found, returns 404
      with a hint (a few available times for that date), all shown as `_iso_utc`.
    - For **POF**: collapses `forecast_time` to unique daily steps (preserving order),
      typically ~10 days.
    - For **FOPI** (and others): returns all available steps (3-hourly), sorted and unique.
    - All returned times (`base_time` and each entry in `forecast_steps`) are
      serialized with `_iso_utc`, ensuring explicit UTC with trailing 'Z'.

    Returns
    -------
    dict
        Example:
        {
          "index": dataset name (fopi or pof),
          "base_time": "2025-07-11T00:00:00Z",
          "forecast_steps": [
            "2025-07-11T00:00:00Z",
            "2025-07-11T03:00:00Z",
            ...
          ]
        }

    Errors
    ------
    - 400 Bad Request:
      - Missing required coordinates in the Zarr dataset.
      - Other processing errors (returned as JSON with an "error" message).
    - 404 Not Found:
      - No `base_time` match (after attempting 00Z↔12Z fallback) for the requested date.

    Notes
    -----
    - The dataset is loaded via `_load_zarr(index, base_time)`.
    - Coordinates are normalized to timezone-naive, second precision for comparison;
      `_iso_utc` is used only for output serialization.
    """
    try:
        ds = _load_zarr(index)

        # Parse requested base_time (accepts with/without Z; normalize to naive seconds)
        requested = _parse_naive(base_time)

        # All available base_time values (naive, second precision)
        base_vals = pd.to_datetime(ds["base_time"].values)
        base_vals = pd.to_datetime([
            pd.Timestamp(bv).tz_localize(None).replace(microsecond=0) for bv in base_vals
        ])

        # Exact match?
        matches = np.where(base_vals == requested)[0]

        # Fallback 00Z <-> 12Z (common for multiple daily runs)
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

        matched_base = base_vals[matches[0]]

        # Slice at the selected base_time
        ds_bt = ds.sel(base_time=matched_base)

        # Collect forecast_time values (naive, second precision)
        ft_vals = pd.to_datetime(ds_bt["forecast_time"].values)
        ft_vals = [pd.Timestamp(x).tz_localize(None).replace(microsecond=0) for x in ft_vals]

        if index.lower() == "pof":
            # Collapse to unique days (preserving order), typically 10
            seen = set()
            daily = []
            for v in ft_vals:
                d = v.floor("D")
                if d not in seen:
                    seen.add(d)
                    daily.append(d)
                if len(daily) >= 10:  # cap at 10 days if desired
                    break
            out_times = daily
        else:
            # FOPI: keep all 3-hourly steps, sorted & unique
            out_times = sorted(set(ft_vals))

        forecast_steps = [_iso_utc(t) for t in out_times]

        return {
            "index": index,
            "base_time": _iso_utc(matched_base),
            "forecast_steps": forecast_steps,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Failed to retrieve forecast steps (by_date)")
        return JSONResponse(status_code=400, content={"error": str(e)})
