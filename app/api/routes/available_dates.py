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


@router.get("/{index}/available_dates", response_model=dict)
def fetch_available_dates(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    session: Session = Depends(get_session),
):
    """
    Retrieve all available forecast initialization dates from the database for the specified dataset.

    Args:
        index (str): Dataset identifier.
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
