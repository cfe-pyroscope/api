from fastapi import APIRouter, Query, Path, Depends, HTTPException
from fastapi.responses import JSONResponse
from datetime import timedelta
import pandas as pd

from utils.zarr_handler import _load_zarr
from utils.time_utils import (
    _iso_utc_str,
    _iso_naive_utc,
    _normalize_times,
    _match_base_time,
    _match_forecast_time,   # ← add this import
)
from config.logging_config import logger

router = APIRouter()

@router.get("/{index}/by_forecast")
def get_forecast_evolution(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(
        ..., description="Verification time ISO8601 (e.g. '2025-09-02T00:00:00Z')."
    ),
) -> dict:
    """
    For the requested verification date, return the list of base_time dates
    (verification−9 … verification) that *contain* that verification date in
    their forecast_time values (date-only match).
    """
    try:
        ds = _load_zarr(index)

        # 1) Normalize and match request to an exact base_time coord from the dataset
        req_base, _ = _normalize_times(base_time, base_time)
        matched_base = _match_base_time(ds, req_base)

        # 2) Work with date-only (UTC) for the verification day
        verification_midnight = _iso_naive_utc(base_time)     # tz-naive UTC midnight
        verification_date = pd.to_datetime(verification_midnight, utc=True).date()

        # 3) Map each UTC date → the dataset's original base_time label
        bt_vals_orig = pd.to_datetime(ds["base_time"].values)          # original labels
        bt_vals_utc  = pd.to_datetime(ds["base_time"].values, utc=True)  # for comparisons
        date_to_bt_orig = {utc_ts.date(): orig for orig, utc_ts in zip(bt_vals_orig, bt_vals_utc)}

        # 4) Build the 10-day date window [verification−9 … verification] in ascending order
        window_dates = [verification_date - timedelta(days=d) for d in range(9, -1, -1)]

        # 5) Keep only those base_time dates whose row contains the verification DATE
        out_times: list[str] = []
        for d in window_dates:
            bt_orig = date_to_bt_orig.get(d)
            if bt_orig is None:
                continue  # (you noted dates always exist, but guard anyway)

            # Ensure the selector matches the dataset's coord type (tz-naive, second precision)
            bt_label = pd.Timestamp(bt_orig).tz_localize(None).replace(microsecond=0)

            # match a forecast date for this base_time; if not present, skip
            try:
                _ = _match_forecast_time(ds, bt_label, verification_midnight)
                out_times.append(_iso_utc_str(bt_label))  # normalize to 00:00Z for output
            except Exception:
                # No forecast for the verification date at this base_time
                continue

        logger.info(
            "FORECAST steps (by_forecast): first=%s last=%s count=%d",
            out_times[0] if out_times else "∅",
            out_times[-1] if out_times else "∅",
            len(out_times),
        )
        return {
            "index": index,
            "base_time": matched_base.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "forecast_time": out_times,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Failed to get forecast evolution steps (by_forecast)")
        return JSONResponse(status_code=400, content={"error": str(e)})
