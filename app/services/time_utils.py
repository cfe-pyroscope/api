import re
from datetime import datetime
import numpy as np
import pandas as pd
from app.logging_config import logger


def calculate_time_index(ds, index: str, base_time: str, lead_hours: int) -> int:
    """
    Calculate the index of the time step in the dataset that is closest to the requested forecast time.

    Parameters:
        ds (xr.Dataset): The dataset containing a 'time' coordinate.
        index (str): Dataset identifier used to extract base time from filename encoding.
        base_time (str): Base time in ISO 8601 format (e.g. "2025-06-20T00:00:00Z").
        lead_hours (int): Lead time in hours to add to the base time.

    Returns:
        int: Index of the closest time step in the dataset.
    """
    base_file_time = extract_base_time_from_encoding(ds, index)
    valid_hours = compute_valid_hour(base_file_time, base_time, lead_hours)
    logger.info(f"ℹ️ index in calculate_time_index: {index}")
    if index == "fopi":
        time_in_hours = ds.time.values  # already float
    else:
        time_in_hours = np.array([
            (pd.to_datetime(t).to_pydatetime() - base_file_time).total_seconds() / 3600
            for t in ds.time.values
        ])

    time_index = int(np.argmin(np.abs(time_in_hours - valid_hours)))
    logger.info(f"🧭 Closest time index: {time_index}, Available times: {time_in_hours.tolist()}")
    return time_index


def extract_base_time_from_encoding(ds, index: str) -> datetime:
    """
    Extract the base timestamp from the dataset's source filename.

    Parameters:
        ds (xr.Dataset): Dataset with encoding metadata containing the filename.
        index (str): Dataset identifier used to select the matching regex pattern.

    Returns:
        datetime: Parsed base file time from the filename (e.g. 2024120100 → datetime object).
    """
    encoding_source = str(ds.encoding.get("source", ""))
    match = re.search(rf"{index}_(\d{{10}})", encoding_source)
    base_time_str = match.group(1) if match else "2024120100"
    file_base_time = datetime.strptime(base_time_str, "%Y%m%d%H")
    logger.info(f"📆 Base file time: {file_base_time.isoformat()}")
    return file_base_time


def compute_valid_hour(file_base_time: datetime, base_time_str: str, lead_hours: int) -> float:
    """
    Compute the number of forecast hours relative to the file's base time.

    Parameters:
        file_base_time (datetime): Base time parsed from the dataset filename.
        base_time_str (str): Requested base time in ISO 8601 format.
        lead_hours (int): Lead time in hours to add.

    Returns:
        float: Valid forecast hour (including offset from file base time).
    """
    base_dt = datetime.fromisoformat(base_time_str.replace("Z", ""))
    valid_hours = (base_dt - file_base_time).total_seconds() / 3600 + lead_hours
    logger.info(f"🕐 Requested valid hours from base: {valid_hours}")
    return valid_hours
