import os
import re
from datetime import datetime
from typing import List, Tuple


def parse_filename(filename: str) -> Tuple[str, datetime]:
    """
    Parse a NetCDF (.nc) filename and extract the dataset name and timestamp.

    Recognizes two specific filename patterns:
      1. fopi_YYYYMMDDHH.nc  → returns ("Fopi", datetime)
      2. POF_V2_YYYY_MM_DD_FC.nc → returns ("Pof", datetime)

    Args:
        filename (str): The name of the .nc file to parse.

    Returns:
        Tuple[str, datetime]: A tuple containing the dataset name ("Fopi" or "Pof") and the parsed datetime object.

    Raises:
        ValueError: If the filename does not match any known patterns.
    """
    # Pattern 1: fopi_YYYYMMDDHH.nc → dataset = "Fopi"
    match1 = re.match(r"fopi_(\d{10})\.nc$", filename, re.IGNORECASE)
    if match1:
        dt = datetime.strptime(match1.group(1), "%Y%m%d%H")
        return "Fopi", dt

    # Pattern 2: POF_V2_YYYY_MM_DD_FC.nc → dataset = "Pof"
    match2 = re.match(r"POF_V2_(\d{4})_(\d{2})_(\d{2})_FC\.nc$", filename, re.IGNORECASE)
    if match2:
        year, month, day = map(int, match2.groups())
        dt = datetime(year, month, day)
        return "Pof", dt

    raise ValueError(f"Unrecognized filename format: {filename}")


def scan_storage_files(directory: str) -> List[Tuple[str, datetime, str]]:
    """
    Scan a directory for NetCDF (.nc) files, parse their filenames to extract metadata,
    and return a list of parsed entries.

    For each valid .nc file, the filename is parsed to extract:
      - Dataset name (e.g., "Fopi", "Pof")
      - Corresponding datetime
      - Full file path

    Files that do not match known patterns are skipped with a warning.

    Args:
        directory (str): Path to the directory containing .nc files.

    Returns:
        List[Tuple[str, datetime, str]]: A list of tuples containing the dataset name,
        parsed datetime, and full file path for each recognized file.
    """
    entries = []
    for filename in os.listdir(directory):
        if filename.endswith(".nc"):
            try:
                dataset, dt = parse_filename(filename)
                filepath = os.path.join(directory, filename)
                entries.append((dataset, dt, filepath))
            except Exception as e:
                print(f"Skipping file {filename}: {e}")
    return entries
