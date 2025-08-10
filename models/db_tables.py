from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class BaseEntry(SQLModel):
    """
    Base model for forecast data entries.

    Represents common fields shared by all datasets stored in the database, including:
        - `datetime`: The initialization or valid time of the forecast.
        - `dataset`: Identifier for the dataset (e.g., 'fopi' or 'pof').
        - `filepath`: Relative or absolute path to the associated NetCDF file.
    """
    datetime: datetime
    dataset: str
    filepath: str


class Fopi(BaseEntry, table=True):
    """
    Database table model for the FOPI dataset.

    Inherits from `BaseEntry` and adds an auto-incrementing primary key field (`id`).
    This model is mapped to a database table via SQLModel's `table=True`.
    """
    id: Optional[int] = Field(default=None, primary_key=True)


class Pof(BaseEntry, table=True):
    """
    Database table model for the POF dataset.

    Inherits from `BaseEntry` and adds an auto-incrementing primary key field (`id`).
    This model is mapped to a database table via SQLModel's `table=True`.
    """
    id: Optional[int] = Field(default=None, primary_key=True)