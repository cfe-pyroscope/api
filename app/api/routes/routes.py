from fastapi import APIRouter, Depends, Query
from sqlmodel import Session
from typing import List
from app.api.db.session import get_session
from fastapi.responses import JSONResponse
from app.api.models.tables import Fopi, Pof
from datetime import datetime
from app.logging_config import logger
from app.api.crud import db_operations


index_table_map = {
    "fopi": Fopi,
    "pof": Pof,
}

router = APIRouter()

@router.get("/available_dates", response_model=List[datetime])
def get_pof_dates(index: str = Query("pof"),session: Session = Depends(get_session)):
    return db_operations.get_available_dates(session, index)

@router.get('/latest_date', response_model=dict)
def get_latest_date(index: str = Query("pof"), session: Session = Depends(get_session)):
    logger.info(f"📅 latest-date endpoint hit with index={index}")
    try:
        return db_operations.get_latest_datetime(session, index)
    except Exception as e:
        logger.exception("❌ Failed to get latest date")
        return JSONResponse(status_code=400, content={"error": str(e)})
