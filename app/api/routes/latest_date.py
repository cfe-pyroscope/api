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
    Return the most recent available date for a given dataset index.
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
