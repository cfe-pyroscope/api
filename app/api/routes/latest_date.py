from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from datetime import datetime
import re

from app.services.zarr_loader import get_latest_nc_file
from app.logging_config import logger

router = APIRouter()


@router.get("/latest-date")
async def get_latest_date(index: str = Query("pof", description="Dataset index: 'pof' or 'fopi'")):
    """
    Retrieve the most recent available forecast date for the specified dataset index.

    This endpoint inspects the latest available NetCDF file for the given dataset (`fopi` or `pof`),
    extracts the date from the filename using a known pattern, and returns it in ISO 8601 format.

    Args:
        index (str): Dataset identifier. Defaults to "pof". Must be either "pof" or "fopi".

    Returns:
        dict: A dictionary containing:
            - `latest_date` (str): The most recent available forecast date in ISO format (YYYY-MM-DD).

    Raises:
        400 Bad Request: If the file is not found, filename doesn't match expected patterns,
        or another error occurs during processing.
    """
    logger.info(f"📅 latest-date endpoint hit with index={index}")
    try:
        nc_path = get_latest_nc_file(index)

        if index == "fopi":
            match = re.search(r"fopi_(\d{8})", nc_path.name)
            if not match:
                raise ValueError("Filename pattern did not match")
            date_str = match.group(1)

        elif index == "pof":
            match = re.search(r"POF_V2_(\d{4})_(\d{2})_(\d{2})", nc_path.name)
            if not match:
                raise ValueError("Filename pattern did not match")
            date_str = f"{match.group(1)}{match.group(2)}{match.group(3)}"

        else:
            raise ValueError("Unsupported index")

        latest_date = datetime.strptime(date_str, "%Y%m%d").date().isoformat()
        return {"latest_date": latest_date}

    except Exception as e:
        logger.exception("❌ Failed to get latest date")
        return JSONResponse(status_code=400, content={"error": str(e)})
