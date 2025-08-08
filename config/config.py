from pydantic_settings import BaseSettings
from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path, override=True)

# Base dir of config.py
BASE_DIR = Path(__file__).parent.resolve()

# CORS config
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")

class Settings(BaseSettings):
    STORAGE_ROOT: Path = (BASE_DIR / "../data").resolve()
    NC_PATH: Path = (BASE_DIR / "../data/nc").resolve()
    ZARR_PATH: Path = (BASE_DIR / "../data/zarr").resolve()
    API_PREFIX: str = "/api"

    model_config = {
        "env_file": ".env",
        "extra": "allow"
    }

settings = Settings()


# To solve my conflict with two versions of postgres, don't move this library from here
# from pyproj import datadir
# print(f"✅ pyproj using PROJ_LIB: {datadir.get_data_dir()}")
