import io
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from scipy.ndimage import zoom
import matplotlib
import matplotlib.pyplot as plt

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.responses import StreamingResponse
from modules.zarr_loader import convert_nc_to_zarr, load_zarr


matplotlib.use('Agg')

app = FastAPI(
    title="Fire Front Radar API",
    description="",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    contact={
    "name": "Fire Front Radar Team",
    "email": "info@firefrontradar.org",  # <- or just remove
    "url": "https://firefrontradar.org"  # <- or just remove
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_methods=["*"],
    allow_headers=["*"],
)

convert_nc_to_zarr()
ds = load_zarr()

@app.get("/")
async def index():
    return {"message": "Fire Front Radar API"}

# Base time reference
BASE_TIME_STR = "2024-12-01T00:00:00"
BASE_TIME = datetime.fromisoformat(BASE_TIME_STR)

@app.get("/")
async def index():
    return {"message": "Fire Front Radar API"}

@app.get("/fopi")
async def get_fopi(base: str = Query(...), lead: int = Query(...)):
    try:
        base_dt = datetime.fromisoformat(base.replace("Z", ""))
        valid_dt = base_dt + timedelta(hours=lead)

        lat_center = 38.5
        lon_center = -120.2

        return {
            "location": [lat_center, lon_center],
            "valid_time": valid_dt.isoformat()
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/bounds")
def get_bounds():
    return {
        "lat_min": float(ds.lat.min()),
        "lat_max": float(ds.lat.max()),
        "lon_min": float(ds.lon.min()),
        "lon_max": float(ds.lon.max()),
    }

@app.get("/heatmap/image")
def get_heatmap_image(
    base: str = Query(...),
    lead: int = Query(...),
    bbox: str = Query(None)
):
    try:
        base_dt = datetime.fromisoformat(base.replace("Z", ""))
        valid_hours = (base_dt - BASE_TIME).total_seconds() / 3600 + lead

        # Convert all temporal values in "hours from BASE_TIME"
        time_in_hours = np.array([
            (pd.to_datetime(t).to_pydatetime() - BASE_TIME).total_seconds() / 3600
            for t in ds.time.values
        ])

        # Calculate time index closest to the requested value
        time_index = int(np.argmin(np.abs(time_in_hours - valid_hours)))

        param = list(ds.data_vars.keys())[0]
        data = ds[param].isel(time=time_index).values
        data = np.nan_to_num(data, nan=0.0)

        data = zoom(data, 0.2)

        if bbox:
            try:
                lat_min, lon_min, lat_max, lon_max = map(float, bbox.split(','))
            except ValueError:
                return JSONResponse(status_code=400, content={"error": "Invalid bbox format"})
        else:
            lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())
            lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.axis("off")

        ax.imshow(
            data,
            cmap="inferno",
            extent=[lon_min, lon_max, lat_min, lat_max],
            origin="lower"
        )

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        print("Error in /heatmap/image:", str(e))
        return JSONResponse(status_code=400, content={"error": str(e)})
