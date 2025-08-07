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


from fastapi.responses import JSONResponse

@router.get("/available_dates", response_model=dict)
def fetch_available_dates(index: str = Query("pof"), session: Session = Depends(get_session)):
    """
    Retrieve all available forecast initialization dates from the database for the specified dataset.

    Args:
        index (str): Dataset identifier. Defaults to "pof".
        session (Session): Database session injected via dependency.

    Returns:
        dict: A dictionary containing a list of available initialization timestamps under the key 'available_dates'.
    """
    try:
        dates = db_operations.get_available_dates(session, index)
        return {"available_dates": dates}
    except Exception as e:
        logger.exception("❌ Failed to fetch available dates")
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.get('/latest_date', response_model=dict)
def get_latest_date(index: str = Query("pof"), session: Session = Depends(get_session)):
    """
        Retrieve the most recent forecast initialization date for the specified dataset.

        Args:
            index (str): Dataset identifier. Defaults to "pof".
            session (Session): Database session injected via dependency.

        Returns:
            dict: A dictionary containing the latest available initialization date (e.g., {"latest_date": "2025-08-07T00:00:00Z"}).

        Raises:
            400 Bad Request: If the latest date cannot be retrieved due to a processing or database error.
    """
    logger.info(f"📅 latest-date endpoint hit with index={index}")
    try:
        return db_operations.get_latest_datetime(session, index)
    except Exception as e:
        logger.exception("❌ Failed to get latest date")
        return JSONResponse(status_code=400, content={"error": str(e)})
