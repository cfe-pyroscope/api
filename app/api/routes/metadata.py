from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime, timedelta

from app.services.zarr_loader import load_zarr
from app.services.time_utils import extract_base_time_from_encoding
from app.logging_config import logger

router = APIRouter()


@router.get("/{index}/metadata")
async def get_index_metadata(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(..., description="Base time in ISO 8601 format (e.g., '2025-06-20T00:00:00Z')."),
    lead_hours: int = Query(..., description="Lead time in hours to add to the base time.")
) -> dict:
    """
    Retrieve forecast metadata for a given dataset and forecast time.

    This endpoint loads the Zarr dataset corresponding to the provided index and base time,
    then calculates:
      - The valid timestamp (base_time + lead_hours)
      - The geographic center of the grid
      - All available forecast steps (with timestamps and lead hours)

    Args:
        index (str): Dataset identifier ("fopi" or "pof").
        base_time (str): Forecast initialization time in ISO 8601 format.
        lead_hours (int): Lead time (in hours) added to base_time to compute the valid time.

    Returns:
        dict: A dictionary containing:
            - `location`: List [latitude, longitude] for the grid center.
            - `valid_time`: Forecast valid time in ISO format.
            - `forecast_steps`: List of steps with:
                - `time` (str): Forecast time in ISO format
                - `lead_hours` (int): Lead time in hours from base_time

    Raises:
        400 Bad Request: If Zarr loading or time parsing fails.
    """
    try:
        ds = load_zarr(index, base_time)
        logger.info(f"{index} → time dtype: {ds.time.dtype}, values: {ds.time.values[:5]}")

        base_dt = datetime.fromisoformat(base_time.replace("Z", ""))
        valid_dt = base_dt + timedelta(hours=lead_hours)

        lat_center = float(ds.lat.mean())
        lon_center = float(ds.lon.mean())

        file_base_time = extract_base_time_from_encoding(ds, index)

        # ⏱ Construct forecast steps: list of {time, lead_hours}
        if index == "fopi":
            forecast_steps = []
            for t in ds.time.values:
                try:
                    t_val = float(t)
                    forecast_steps.append({
                        "time": (file_base_time + timedelta(hours=t_val)).isoformat(),
                        "lead_hours": int(t_val)
                    })
                except Exception as e:
                    logger.warning(f"⚠️ Skipping invalid time value in fopi: {t} ({e})")
        else:  # for pof
            forecast_steps = [
                {
                    "time": pd.to_datetime(t).isoformat(),
                    "lead_hours": int((pd.to_datetime(t) - file_base_time).total_seconds() // 3600)
                }
                for t in ds.time.values
            ]
        logger.info(f"✅ Forecast steps count: {len(forecast_steps)}")
        logger.info(f"📅 File base time: {file_base_time}")
        logger.info(f"📤 Forecast steps returned: {[step['time'] for step in forecast_steps]}")

        return {
            "location": [lat_center, lon_center],
            "valid_time": valid_dt.isoformat(),
            "forecast_steps": forecast_steps
        }

    except Exception as e:
        logger.exception("❌ Failed to retrieve index metadata")
        return JSONResponse(status_code=400, content={"error": str(e)})

