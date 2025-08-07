from pathlib import Path
from dotenv import load_dotenv
import os
import re


dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path, override=True)

# To solve my conflict with two versions of postgres, don't move this library from here
from pyproj import datadir
print(f"✅ pyproj using PROJ_LIB: {datadir.get_data_dir()}")

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
NC_DIR = DATA_DIR / "nc"
ZARR_DIR = DATA_DIR / "zarr"

# CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

# API
API_PREFIX = "/api"
