from sqlmodel import create_engine, Session

DATABASE_URL = "sqlite:///./db/app.db"
engine = create_engine(DATABASE_URL, echo=True)


def get_session():
    """
    Dependency function that provides a database session.

    This generator yields a new SQLModel session using the configured `engine`.
    It ensures the session is properly opened and closed using a context manager.

    Returns:
        Generator[Session, None, None]: A SQLModel `Session` object for database access.
    """
    with Session(engine) as session:
        yield session
