from sqlmodel import Session, select, func
from typing import List, Optional
from datetime import datetime, timedelta
from app.api.models.tables import Fopi, Pof
from app.logging_config import logger

DATASET_MODELS = {
    "fopi": Fopi,
    "pof": Pof,
}

def get_available_dates(session: Session, dataset: str) -> List[datetime]:
    model = DATASET_MODELS.get(dataset.lower())
    if not model:
        raise ValueError(f"Unknown dataset: {dataset}")
    statement = select(model.datetime).distinct().order_by(model.datetime)
    return session.exec(statement).all()

def get_latest_datetime(session: Session, index: str) -> Optional[dict]:
    table_cls = DATASET_MODELS.get(index.lower())
    if table_cls is None:
        logger.exception("❌ Failed to get latest date")
        raise Exception(f"Index {index} not valid")
    statement = select(func.max(table_cls.datetime))
    result = session.exec(statement).one()
    latest_date = result.date().isoformat()
    return {"latest_date": latest_date}

def get_records_by_datetime(session: Session, dataset: str, target_time: datetime):
    model = DATASET_MODELS.get(dataset.lower())
    if not model:
        raise ValueError(f"Unknown dataset: {dataset}")
    statement = select(model).where(model.datetime > target_time.date(), model.datetime < target_time.date() + timedelta(days=1))
    return session.exec(statement).one()