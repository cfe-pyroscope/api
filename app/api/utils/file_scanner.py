import os
import re
from datetime import datetime
from typing import List, Tuple


def parse_filename(filename: str) -> Tuple[str, datetime]:
    """
    Parses .nc filenames for two known formats and assigns canonical dataset names.
    Returns (dataset_name, datetime).
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
    Scans a directory for .nc files, parses them, and returns a list of (dataset, datetime, filepath)
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
