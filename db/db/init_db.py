from sqlmodel import SQLModel
from db.db.session import engine


def init_db():
    """
    Initialize the database by creating all tables defined in the SQLModel metadata.

    This function uses the configured `engine` to create any missing tables based
    on the models registered with SQLModel. It does not drop or modify existing tables.

    Used during application startup or development to ensure the schema is in place.
    Database will be automatically created on first start up of the api app.
    """
    SQLModel.metadata.create_all(engine)