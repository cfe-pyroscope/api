from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Response
from modules.zarr_loader import convert_nc_to_zarr, load_zarr
import numpy as np
from scipy.ndimage import zoom

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


@app.get("/metadata")
def get_metadata():
    """
    Retrieve basic metadata from the loaded dataset.

    Returns:
        dict: A dictionary containing:
            - "dims": A dictionary of dataset dimensions and their sizes.
            - "coords": A dictionary with coordinate ranges:
                - "lat": Minimum and maximum latitude values as floats.
                - "lon": Minimum and maximum longitude values as floats.
                - "time": List of time values from the dataset (raw format).
            - "variables": A list of variable names (data fields) in the dataset.
    """
    return {
        "dims": dict(ds.dims),
        "coords": {
            "lat": [float(ds.lat.min()), float(ds.lat.max())],
            "lon": [float(ds.lon.min()), float(ds.lon.max())],
            "time": ds.time.values.tolist()
        },
        "variables": list(ds.data_vars)
    }


@app.get("/heatmap/summary")
async def get_heatmap_summary(time_index: int = Query(0)):
    """
    Returns summary statistics of the heatmap at the given time index.

    This endpoint provides basic metadata (min, max, shape, number of NaNs)
    about the 2D heatmap for a specific time step. Useful for diagnostic
    dashboards or validating data ranges before requesting full arrays.

    Args:
        time_index (int): Index along the time dimension to extract the data from.

    Returns:
        dict: A dictionary containing:
            - "time": the time value at the requested index,
            - "shape": shape of the 2D array,
            - "min": minimum value (ignoring NaNs),
            - "max": maximum value (ignoring NaNs),
            - "nan_count": total number of NaN values.
    """
    param = list(ds.data_vars.keys())[0]
    arr = ds[param].isel(time=time_index).values
    arr = np.nan_to_num(arr, nan=np.nan)

    return {
        "time": float(ds.time[time_index]),
        "shape": arr.shape,
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "nan_count": int(np.isnan(arr).sum())
    }


@app.get("/heatmap/full")
async def get_full_heatmap(time_index: int = Query(0)):
    """
    Returns the full heatmap array at the specified time index.

    This endpoint is intended for frontend apps that require the full
    2D array of data for visualization or computation. It replaces all
    invalid values (NaNs and infinities) with sentinel values.

    Args:
        time_index (int): Index along the time dimension to extract the data from.

    Returns:
        JSONResponse: A JSON object containing:
            - "time": the time value at the requested index,
            - "heatmap": the full 2D array as a nested list (with cleaned values).

    Note:
        🚫 Do not test this from Swagger — it's huge. Use only from frontend JS fetch.
    """
    param = list(ds.data_vars.keys())[0]
    arr = ds[param].isel(time=time_index).values
    arr = np.nan_to_num(arr, nan=-9999.0, posinf=9999.0, neginf=-9999.0)

    return JSONResponse({
        "time": float(ds.time[time_index]),
        "heatmap": arr.tolist()
    })


@app.get("/heatmap/tile")
async def get_heatmap_tile(
    time_index: int = Query(0),
    scale: float = Query(0.05, ge=0.01, le=1.0)
):
    """
    Returns a downsampled (tiled) version of the heatmap for map display.

    Useful for visualizing heatmaps in a web map component like Leaflet,
    this endpoint reduces the array resolution using bilinear interpolation.

    Args:
        time_index (int): Index along the time dimension to extract the data from.
        scale (float): Factor to downsample the heatmap by (0.01–1.0).
                       For example, 0.05 returns ~5% of the original resolution.

    Returns:
        dict: A dictionary containing:
            - "time": the time value at the requested index,
            - "heatmap": the downsampled 2D array as a nested list,
            - "scale": the downsampling factor used.
    """
    param = list(ds.data_vars.keys())[0]
    arr = ds[param].isel(time=time_index).values
    arr = np.nan_to_num(arr, nan=-9999.0, posinf=9999.0, neginf=-9999.0)

    # Downsample (e.g., 5% of original resolution)
    downsampled = zoom(arr, zoom=scale)

    return {
        "time": float(ds.time[time_index]),
        "heatmap": downsampled.tolist(),
        "scale": scale
    }


@app.get("/timeseries")
def get_timeseries(lat: float, lon: float):
    """
        Retrieve the time series of a variable at the nearest grid point to the specified latitude and longitude.

        Parameters:
            lat (float): Latitude coordinate in degrees. The nearest available value in the dataset will be used.
            lon (float): Longitude coordinate in degrees. The nearest available value in the dataset will be used.

        Returns:
            dict: A dictionary containing:
                - "lat": The actual latitude value from the dataset closest to the input.
                - "lon": The actual longitude value from the dataset closest to the input.
                - "timeseries": A list of values over time for the selected variable at the specified location.

        Notes:
            - The time series is extracted from the first variable in the dataset.
            - The time values themselves are not included in the response, but could be added if needed.
        """
    param = list(ds.data_vars.keys())[0]
    nearest_lat = ds.lat.sel(lat=lat, method="nearest")
    nearest_lon = ds.lon.sel(lon=lon, method="nearest")
    ts = ds[param].sel(lat=nearest_lat, lon=nearest_lon).values
    return {
        "lat": float(nearest_lat),
        "lon": float(nearest_lon),
        "timeseries": ts.tolist()
    }
