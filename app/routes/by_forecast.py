from fastapi import APIRouter, Query, Path, Depends, HTTPException
from fastapi.responses import JSONResponse
from datetime import timedelta
import pandas as pd

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
    Get forecast evolution leading up to a verification time.
    Builds daily forecast steps from 9 days before the given `base_time`
    (verification time) up to the verification date itself. Returns the
    dataset index, the verification time, and the list of forecast steps
    in ISO8601 UTC format.
    """
    try:
        verification = _iso_naive_utc(base_time)
        start_date = verification - timedelta(days=9)
        logger.info(
            f"Selected verification (target) time: {_iso_utc_str(verification)} — building steps in [{start_date}, {verification}]"
        )

        start_d = pd.Timestamp(start_date.date())
        end_d = pd.Timestamp(verification.date())
        rng = pd.date_range(start=start_d, end=end_d, freq="D")
        forecast_time = [_iso_utc_str(ts.to_pydatetime()) for ts in rng]

        logger.info(
            "FORECAST steps (by_forecast): first=%s last=%s count=%d",
            forecast_time[0], forecast_time[-1], len(forecast_time)
        )
        return {
            "index": index,
            "base_time": _iso_utc_str(verification),
            "forecast_time": forecast_time,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Failed to get forecast evolution steps (by_forecast)")
        return JSONResponse(status_code=400, content={"error": str(e)})
