from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime, timedelta

from app.services.zarr_loader import load_zarr
from app.services.time_utils import extract_base_time_from_encoding
from app.logging_config import logger

router = APIRouter()


@router.get("/{index}")
async def get_index_metadata(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(..., description="Base time in ISO 8601 format (e.g., '2025-06-20T00:00:00Z')."),
    lead_hours: int = Query(..., description="Lead time in hours to add to the base time.")
) -> dict:
    """
    Retrieve metadata for a given dataset index including geographic center,
    valid timestamp, and forecast steps with lead times.
    """
    try:
        ds = load_zarr(index)
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

        return {
            "location": [lat_center, lon_center],
            "valid_time": valid_dt.isoformat(),
            "forecast_steps": forecast_steps
        }

    except Exception as e:
        logger.exception("❌ Failed to retrieve index metadata")
        return JSONResponse(status_code=400, content={"error": str(e)})
