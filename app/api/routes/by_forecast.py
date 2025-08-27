from fastapi import APIRouter, Query, Path, Depends, HTTPException
from fastapi.responses import JSONResponse
from datetime import timedelta
import pandas as pd

from app.utils.zarr_loader import _load_zarr
from app.utils.time_utils import _parse_naive, _iso_utc
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/by_forecast")
def get_forecast_evolution(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(
        ..., description="Verification time ISO8601 (e.g., '2025-07-11T00:00:00' or '2025-07-11T00:00:00Z')."
    ),
) -> dict:
    """Retrieve the temporal sequence of forecast targets for a given verification time.

    This endpoint generates the list of forecast time steps covering the 9 days leading
    up to a selected **verification time** (``base_time`` in the request). The temporal
    resolution depends on the dataset:

    - **POF**: daily targets (one per day).
    - **FOPI**: 3‑hourly targets (eight per day).

    Parameters
    ----------
    index : str
        Dataset identifier, either ``"pof"`` or ``"fopi"``.
    base_time : str
        Verification time in ISO8601 format (e.g., ``"2025-07-11T00:00:00Z"``).
    session : Session
        Database session (injected by FastAPI, currently unused).

    Returns
    -------
    dict
        Dictionary with:
        - ``index``: dataset name (fopi or pof).
        - ``base_time``: selected verification time in ISO8601 UTC.
        - ``forecast_time``: list of ISO8601 UTC timestamps representing forecast
          targets within the 9‑day lookback window at the dataset's native temporal resolution.

    Raises
    ------
    HTTPException
        400 if the index is unsupported or required coordinates are missing.
        404 if no forecast steps are found in the requested range.
    """
    try:
        ds = _load_zarr(index)

        # Parse the requested verification time to timezone‑naive, second precision (UTC)
        verification = _parse_naive(base_time)
        start_date = verification - timedelta(days=9)
        logger.info(
            f"🔍 Selected verification (target) time: {_iso_utc(verification)} — building steps in [{start_date}, {verification}]"
        )

        name = index.lower()
        if name == "pof":
            # POF is daily: generate daily targets from start_date..verification (inclusive)
            start_d = pd.Timestamp(start_date.date())
            end_d = pd.Timestamp(verification.date())
            rng = pd.date_range(start=start_d, end=end_d, freq="D")
            forecast_time = [_iso_utc(ts.to_pydatetime()) for ts in rng]
        elif name == "fopi":
            # FOPI is 3‑hourly: align start to the nearest 3‑hour boundary <= start_date
            start_ts = pd.Timestamp(start_date)
            aligned = start_ts.floor("3h")
            rng = pd.date_range(start=aligned, end=pd.Timestamp(verification), freq="3H")
            forecast_time = [_iso_utc(ts.to_pydatetime()) for ts in rng]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported index '{index}'. Use 'pof' or 'fopi'.")

        if not forecast_time:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No forecast targets could be generated in the range {start_date.isoformat()} to "
                    f"{verification.isoformat()} for index '{index}'."
                ),
            )

        logger.info(
            "🥎🥎🥎 FORECAST steps (by_forecast): first=%s last=%s count=%d",
            forecast_time[0], forecast_time[-1], len(forecast_time),
        )
        return {
            "index": index,
            "base_time": _iso_utc(verification),
            "forecast_time": forecast_time,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Failed to get forecast evolution steps (by_forecast)")
        return JSONResponse(status_code=400, content={"error": str(e)})


