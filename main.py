import os
from pathlib import Path
from config.config import settings
from db.db.init_db import init_db
from db.db.bootstrap import sync_dataset
from models.db_tables import Fopi, Pof
import uvicorn

# To solve my conflict with two versions of postgres (DON'T REMOVE OR COMMENT IT)
os.environ["PROJ_LIB"] = str(
   Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages" / "pyproj" / "proj_dir" / "share" / "proj"
)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import available_dates, forecast, heatmap, latest_date, metadata



app = FastAPI(
    title="Fire Front Radar API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Extent-3857", "X-Scale-Min", "X-Scale-Max"],
)

@app.get("/")
def root():
    return {"message": "FastAPI is working"}


@app.on_event("startup")
def on_startup():
    """
    Perform application startup tasks:
      1. Initializes the database and creates tables if they don't exist.
      2. Synchronizes NetCDF files from local storage directories into the database
         for both "Fopi" and "Pof" datasets, based on the configured `STORAGE_ROOT`.

    This ensures that the database is ready and reflects the current state of the file system.
    """
    init_db()
    root = settings.NC_PATH
    # Sync each dataset from its folder into its table
    sync_dataset("Fopi", f"{root}/fopi", Fopi)
    sync_dataset("Pof", f"{root}/pof", Pof)


API = settings.API_PREFIX
app.include_router(available_dates.router, prefix=API)
app.include_router(forecast.router, prefix=API)
app.include_router(heatmap.router, prefix=API)
app.include_router(metadata.router, prefix=API)
app.include_router(latest_date.router, prefix=API)


# This enables debug mode through VSCode and PyCharm
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8090, reload=True, log_level="info")
