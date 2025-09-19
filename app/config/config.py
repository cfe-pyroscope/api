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

""" official palette 5 """
COLORS = ["#00000000", "#fff7ec", "#fdbb84",
          "#ef6548", "#d7301f", "#7f0000"]

""" official palette 10
COLORS = ["#00000000", "#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84",
          "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#7f0000"]"""


""" echarts palette
COLORS = ["#00000000", "#E3E8DA", "#C2DBC0", "#FFBF00", "#CC9A03",
          "#C45B2C", "#AD3822", "#951517", "#3A072C", "#0F0A0A"]"""

RANGE = {
    "pof": [0.0, 0.0025, 0.0075, 0.015, 0.030, 0.050],
    "fopi": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "default": [0.0, 1.0]
}

THRESHOLDS_ = {
    "pof": [0.0, 0.0025, 0.0075, 0.015, 0.030, 0.050],
    "fopi": [0.0, 0.2, 0.4, 0.6, 0.8]
}

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

