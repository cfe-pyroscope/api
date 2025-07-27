from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime


class BaseEntry(SQLModel):
    datetime: datetime
    dataset: str
    filepath: str


class Fopi(BaseEntry, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)


class Pof(BaseEntry, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
