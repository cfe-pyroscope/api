from sqlmodel import SQLModel
from app.api.db.session import engine
from app.api.models.tables import Fopi, Pof


def init_db():
    SQLModel.metadata.create_all(engine)