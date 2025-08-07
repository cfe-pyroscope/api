from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from datetime import datetime
import re

from app.services.zarr_loader import list_all_nc_files
from app.logging_config import logger

router = APIRouter()

@router.get("/available-dates")
async def get_available_dates(index: str = Query("pof", description="Dataset index: 'pof' or 'fopi'")):
    """
    Retrieve a list of all available forecast dates for the specified dataset.

    This endpoint scans the storage for NetCDF (.nc) files associated with the given dataset
    (`fopi` or `pof`), extracts valid dates from the filenames, and returns them in
    chronological order using ISO format (YYYY-MM-DD).

    Args:
        index (str): Dataset identifier. Defaults to "pof". Must be either "pof" or "fopi".

    Returns:
        dict: A dictionary containing:
            - `available_dates` (List[str]): Sorted list of available forecast dates in ISO format.

    Raises:
        400 Bad Request: If the directory scan or date parsing fails for any reason.
    """
    from app.services.zarr_loader import list_all_nc_files

    logger.info(f"📅 available-dates endpoint hit with index={index}")
    try:
        nc_files = list_all_nc_files(index)

        dates = []
        for nc_path in nc_files:
            if index == "fopi":
                match = re.search(r"fopi_(\d{8})", nc_path.name)
                if match:
                    dates.append(datetime.strptime(match.group(1), "%Y%m%d").date().isoformat())
            elif index == "pof":
                match = re.search(r"POF_V2_(\d{4})_(\d{2})_(\d{2})", nc_path.name)
                if match:
                    dates.append(datetime.strptime(f"{match.group(1)}{match.group(2)}{match.group(3)}", "%Y%m%d").date().isoformat())

        dates.sort()
        return {"available_dates": dates}

    except Exception as e:
        logger.exception("❌ Failed to get available dates")
        return JSONResponse(status_code=400, content={"error": str(e)})
