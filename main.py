import os
from pathlib import Path

# To solve my conflict with two versions of postgres
os.environ["PROJ_LIB"] = str(
    Path(__file__).resolve().parent / ".venv" / "Lib" / "site-packages" / "pyproj" / "proj_dir" / "share" / "proj"
)


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import latest_date, metadata, heatmap

from app.config import ALLOWED_ORIGINS, API_PREFIX


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
    expose_headers=["X-Extent-3857"],
)

app.include_router(latest_date.router, prefix=API_PREFIX)
app.include_router(metadata.router, prefix=API_PREFIX)
app.include_router(heatmap.router, prefix=API_PREFIX)
