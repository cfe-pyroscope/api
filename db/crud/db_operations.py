from sqlmodel import Session, select, func
from typing import List, Optional
from datetime import datetime, timedelta
from models.db_tables import Fopi, Pof
from logging_config import logger

DATASET_MODELS = {
    "fopi": Fopi,
    "pof": Pof,
}

def get_available_dates(session: Session, dataset: str) -> List[datetime]:
    """
    Retrieve a list of available forecast initialization dates for a given dataset.

    Queries the database for distinct `datetime` values from the table associated
    with the specified dataset. The dates are returned in ascending order.

    Args:
        session (Session): An active SQLModel database session.
        dataset (str): Name of the dataset (e.g., "fopi" or "pof").

    Returns:
        List[datetime]: A list of available forecast dates for the dataset.

    Raises:
        ValueError: If the provided dataset name is not recognized.
    """
    model = DATASET_MODELS.get(dataset.lower())
    if not model:
        raise ValueError(f"Unknown dataset: {dataset}")
    statement = select(model.datetime).distinct().order_by(model.datetime)
    return session.exec(statement).all()


def get_latest_datetime(session: Session, index: str) -> Optional[dict]:
    """
    Retrieve the most recent forecast initialization date for the specified dataset index.

    This function queries the database for the maximum datetime value in the table
    associated with the given dataset (`index`). The result is returned as an ISO-formatted string.

    Args:
        session (Session): An active SQLModel database session.
        index (str): Dataset identifier (e.g., "fopi" or "pof").

    Returns:
        dict: A dictionary containing the latest available forecast date in ISO format:
              e.g., {"latest_date": "2025-08-07"}

    Raises:
        Exception: If the provided index is not recognized or the query fails.
    """
    table_cls = DATASET_MODELS.get(index.lower())
    if table_cls is None:
        logger.exception("❌ Failed to get latest date")
        raise Exception(f"Index {index} not valid")
    statement = select(func.max(table_cls.datetime))
    result = session.exec(statement).one()
    latest_date = result.date().isoformat()
    return {"latest_date": latest_date}


def get_records_by_datetime(session: Session, dataset: str, target_time: datetime):
    """
    Retrieve a single forecast record from the database for a given dataset and date.

    This function selects a record from the table associated with the given `dataset` where
    the `datetime` field matches the same calendar day as `target_time`.

    It returns one record whose datetime is within the range:
        target_time.date() < datetime < target_time.date() + 1 day

    Args:
        session (Session): An active SQLModel database session.
        dataset (str): Dataset name (e.g., "fopi" or "pof").
        target_time (datetime): The date (and optionally time) to filter records by.

    Returns:
        SQLModel instance: The single matching record from the corresponding dataset table.

    Raises:
        ValueError: If the dataset is unknown.
        sqlmodel.exc.NoResultFound: If no matching record is found.
        sqlmodel.exc.MultipleResultsFound: If more than one matching record exists.
    """
    model = DATASET_MODELS.get(dataset.lower())
    if not model:
        raise ValueError(f"Unknown dataset: {dataset}")
    statement = select(model).where(model.datetime > target_time.date(), model.datetime < target_time.date() + timedelta(days=1))
    return session.exec(statement).one()


def get_all_records(session: Session, dataset: str):
    """
    Retrieve all records from the table corresponding to the dataset.

    Args:
        session (Session): An active SQLModel database session.
        dataset (str): Dataset name (e.g., 'fopi' or 'pof').

    Returns:
        List[SQLModel]: All records from the relevant table.
    """
    model = DATASET_MODELS.get(dataset.lower())
    if not model:
        raise ValueError(f"Unknown dataset: {dataset}")

    statement = select(model)
    return session.exec(statement).all()