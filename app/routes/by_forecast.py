from fastapi import APIRouter, Query, Path, Depends, HTTPException
from fastapi.responses import JSONResponse
from datetime import timedelta
import pandas as pd

from utils.zarr_handler import _load_zarr
from utils.time_utils import _iso_utc_str, _iso_naive_utc
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
    Get forecast evolution leading up to a verification time (inclusive).
    Uses the dataset's actual coordinates for the provided base_time to build
    daily steps from verification-9 days to verification. Ensures results are:
      - selected by coord label (not position),
      - unique per date (prefer 00Z, else earliest hour),
      - sorted by date ascending,
      - filtered to the 10-day window ending at verification.
    """
    try:
        ds = _load_zarr(index)

        # Normalize request and match the exact base_time coord label from the dataset
        from utils.time_utils import _normalize_times, _match_base_time
        req_base, _ = _normalize_times(base_time, base_time)
        matched_base = _match_base_time(ds, req_base)

        # Select this base by COORD (robust to unsorted base_time)
        ds_bt = ds.sel(base_time=matched_base)

        # Pull forecast_time labels (keep originals for returning exact coord labels)
        ft_vals_orig = pd.to_datetime(ds_bt["forecast_time"].values)               # originals
        ft_vals_utc  = pd.to_datetime(ds_bt["forecast_time"].values, utc=True)     # for compare/sort

        # Group by DATE → prefer 00Z if present, else earliest hour
        by_date: dict[date, tuple[pd.Timestamp, pd.Timestamp]] = {}
        for orig, as_utc in zip(ft_vals_orig, ft_vals_utc):
            d = as_utc.date()
            if d not in by_date:
                by_date[d] = (orig, as_utc)
            else:
                best_orig, best_utc = by_date[d]
                if as_utc.hour == 0 or as_utc < best_utc:
                    by_date[d] = (orig, as_utc)

        # Build ordered list (ascending by date)
        ordered = [(d, by_date[d][0]) for d in sorted(by_date.keys())]

        # Define the 10-day window ending at verification date (inclusive)
        verification = _iso_naive_utc(base_time)
        end_d = pd.to_datetime(verification, utc=True).date()
        start_d = (pd.to_datetime(verification, utc=True) - pd.Timedelta(days=9)).date()

        # Filter to window and return exact coord labels as ISO UTC strings
        out_times: list[str] = []
        for d, orig in ordered:
            if start_d <= d <= end_d:
                chosen = pd.Timestamp(orig).tz_localize(None).replace(microsecond=0)
                out_times.append(_iso_utc_str(chosen))

        # Log & respond
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

