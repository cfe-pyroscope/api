from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from datetime import datetime
import re

from app.services.zarr_loader import get_latest_nc_file
from app.logging_config import logger

router = APIRouter()

# 
@router.get("/available-dates")
async def get_available_dates(index: str = Query("pof", description="Dataset index: 'pof' or 'fopi'")):
    logger.info(f"📅 available-dates endpoint hit with index")

    return {"available_dates": ['2024-05-01', '2024-09-03', '2024-05-17', '2024-01-09']}