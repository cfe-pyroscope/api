from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime, timedelta

from app.services.zarr_loader import load_zarr
from app.services.time_utils import extract_base_time_from_encoding
from app.logging_config import logger

router = APIRouter()


@router.get("/metadata/{index}")
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


# @router.get("/latest-date")
# async def get_latest_date(index: str = Query("pof", description="Dataset index: 'pof' or 'fopi'")):
#     """
#     Return the most recent available date for a given dataset index.
#     """
#     from app.services.zarr_loader import get_latest_nc_file
#     from datetime import datetime
#
#     try:
#         nc_path = get_latest_nc_file(index)
#
#         # Extract date from filename
#         if index == "fopi":
#             # Filename like: fopi_20250624.nc
#             import re
#             match = re.search(r"fopi_(\d{8})", nc_path.name)
#             if not match:
#                 raise ValueError("Filename pattern did not match")
#             date_str = match.group(1)
#         elif index == "pof":
#             # Filename like: POF_V2_2025_06_24_FC.nc
#             import re
#             match = re.search(r"POF_V2_(\d{4})_(\d{2})_(\d{2})", nc_path.name)
#             if not match:
#                 raise ValueError("Filename pattern did not match")
#             date_str = f"{match.group(1)}{match.group(2)}{match.group(3)}"
#         else:
#             raise ValueError("Unsupported index")
#
#         latest_date = datetime.strptime(date_str, "%Y%m%d").date().isoformat()
#         return {"latest_date": latest_date}
#
#     except Exception as e:
#         logger.exception("❌ Failed to get latest date")
#         return JSONResponse(status_code=400, content={"error": str(e)})
