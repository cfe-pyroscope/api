from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path
from dotenv import load_dotenv
import os
import re
from typing import Dict, Pattern


# Load .env
dotenv_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path, override=True)

# CORS config
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173").split(",")


# Base dir of config.py
BASE_DIR = Path(__file__).parent.resolve()


class Settings(BaseSettings):
    STORAGE_ROOT: Path = (BASE_DIR / "../../data").resolve()
    NC_PATH: Path = (BASE_DIR / "../../data/nc").resolve()
    ZARR_PATH: Path = (BASE_DIR / "../../data/zarr").resolve()
    API_PREFIX: str = "/api"

    FILENAME_PATTERNS: Dict[str, Pattern] = Field(default_factory=lambda: {
        "fopi": re.compile(r"fopi_(\d{10})\.nc"),
        "pof": re.compile(r"POF_V2_(\d{4})_(\d{2})_(\d{2})_FC\.nc"),
    })

    VAR_NAMES: Dict[str, str] = Field(default_factory=lambda: {
        "fopi": "param100.128.192",
        "pof": "MODEL_FIRE",
    })

    model_config = {
        "env_file": ".env",
        "extra": "allow"
    }

settings = Settings()

