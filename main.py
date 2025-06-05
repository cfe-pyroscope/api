from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from zarr_loader import convert_nc_to_zarr, load_zarr

app = FastAPI(
    title="Fire Front Radar API",
    description="",
    version="1.0.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/api/v1/docs",
    redoc_url="/api/v1/redoc",
    contact={
    "name": "Fire Front Radar Team",
    "email": "",
    "url": ""
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_methods=["*"],
    allow_headers=["*"],
)

convert_nc_to_zarr()  # Ensure Zarr store exists
ds = load_zarr()


@app.get("/metadata")
def get_metadata():
    return {
        "dims": dict(ds.dims),
        "coords": {
            "lat": [float(ds.lat.min()), float(ds.lat.max())],
            "lon": [float(ds.lon.min()), float(ds.lon.max())],
            "time": ds.time.values.tolist()
        },
        "variables": list(ds.data_vars)
    }


@app.get("/heatmap")
def get_heatmap(time_index: int = Query(0)):
    param = list(ds.data_vars.keys())[0]  # e.g., 'param100.128.192'
    arr = ds[param].isel(time=time_index).values
    return {
        "time": float(ds.time[time_index]),
        "heatmap": arr.tolist()  # consider compression in real API
    }


@app.get("/timeseries")
def get_timeseries(lat: float, lon: float):
    param = list(ds.data_vars.keys())[0]
    nearest_lat = ds.lat.sel(lat=lat, method="nearest")
    nearest_lon = ds.lon.sel(lon=lon, method="nearest")
    ts = ds[param].sel(lat=nearest_lat, lon=nearest_lon).values
    return {
        "lat": float(nearest_lat),
        "lon": float(nearest_lon),
        "timeseries": ts.tolist()
    }
