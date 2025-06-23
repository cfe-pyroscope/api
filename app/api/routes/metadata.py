from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse
from app.services.zarr_loader import load_zarr
from datetime import datetime, timedelta
from app.logging_config import logger

router = APIRouter()


@router.get("/{index}")
async def get_index_metadata(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(..., description="Base time in ISO 8601 format (e.g., '2025-06-20T00:00:00Z')."),
    lead_hours: int = Query(..., description="Lead time in hours to add to the base time.")
) -> dict:
    """
    Retrieve metadata for a given dataset index including geographic center and valid timestamp.

    Parameters:
        index (str): Dataset identifier.
        base_time (str): ISO 8601 base timestamp.
        lead_hours (int): Number of hours to add to base time.

    Returns:
        dict: Geographic center and valid forecast time.
    """
    try:
        ds = load_zarr(index)
        base_dt = datetime.fromisoformat(base_time.replace("Z", ""))
        valid_dt = base_dt + timedelta(hours=lead_hours)

        lat_center = float(ds.lat.mean())
        lon_center = float(ds.lon.mean())

        return {
            "location": [lat_center, lon_center],
            "valid_time": valid_dt.isoformat()
        }
    except Exception as e:
        logger.exception("❌ Failed to retrieve index metadata")
        return JSONResponse(status_code=400, content={"error": str(e)})