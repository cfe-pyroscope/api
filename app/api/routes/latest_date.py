from fastapi import APIRouter, Depends, Path
from sqlmodel import Session
from db.db.session import get_session
from models.db_tables import Fopi, Pof
from logging_config import logger
from db.crud import db_operations


index_table_map = {
    "fopi": Fopi,
    "pof": Pof,
}

router = APIRouter()


from fastapi.responses import JSONResponse

@router.get("/{index}/latest_date", response_model=dict)
def get_latest_date(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    session: Session = Depends(get_session),
):
    """
    Retrieve the most recent forecast initialization date for the specified dataset.

    Args:
        index (str): Dataset identifier provided as a path parameter.
        session (Session): Database session injected via dependency.

    Returns:
        dict: A dictionary containing the latest available initialization date
              (e.g., {"latest_date": "2025-08-07T00:00:00Z"}).

    Raises:
        400 Bad Request: If the latest date cannot be retrieved due to a processing or database error.
    """
    logger.info(f"📅 latest-date endpoint hit with index={index}")
    try:
        return db_operations.get_latest_datetime(session, index)
    except Exception as e:
        logger.exception("❌ Failed to get latest date")
        return JSONResponse(status_code=400, content={"error": str(e)})
