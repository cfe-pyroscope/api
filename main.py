from pathlib import Path
from dotenv import load_dotenv

# Load .env file and override existing environment variables
dotenv_path = Path(__file__).with_name(".env")
load_dotenv(dotenv_path, override=True)


from fastapi import FastAPI, Query, Path
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from modules.zarr_loader import load_zarr
from modules.bounds import *
from datetime import datetime, timedelta
import io
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import matplotlib
import rioxarray
import rasterio
import logging
from pyproj import Transformer

matplotlib.use("Agg")

logger = logging.getLogger("uvicorn")

app = FastAPI(
    title="Fire Front Radar API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json"
)

# Configure CORS middleware to allow the front-end application to access our API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Front-end development server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Expose custom header so the front-end can read it
    expose_headers=["X-Extent-3857"],
)


@app.get("/api/{index}")
async def get_index_metadata(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base: str = Query(..., description="Base time in ISO 8601 format (e.g., '2025-06-20T00:00:00Z')."),
    lead: int = Query(..., description="Lead time in hours to add to the base time.")
) -> dict:
    """
    Retrieve metadata for a given dataset index including geographic center and valid timestamp.

    This endpoint:
      - Loads the Zarr-backed dataset for the specified index.
      - Parses the base time string and adds the lead time (in hours) to compute a valid time.
      - Calculates the spatial center (mean latitude and longitude) of the dataset.

    Parameters:
        index (str): Dataset identifier, must match one of the supported indices.
        base (str): Base timestamp in ISO 8601 format.
        lead (int): Number of hours to add to the base time to get the valid time.

    Returns:
        dict: JSON containing:
            - "location": [latitude_center, longitude_center]
            - "valid_time": Valid time as an ISO 8601 string.

    Raises:
        JSONResponse 400: Returns a JSON error message if processing fails.
    """
    try:
        ds = load_zarr(index)
        base_dt = datetime.fromisoformat(base.replace("Z", ""))
        valid_dt = base_dt + timedelta(hours=lead)

        lat_center = float(ds.lat.mean())
        lon_center = float(ds.lon.mean())

        return {
            "location": [lat_center, lon_center],
            "valid_time": valid_dt.isoformat()
        }
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})


@app.get("/api/{index}/heatmap/image")
def get_heatmap_image(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base: str = Query(..., description="Base time in ISO 8601 format (e.g., '2025-06-20T00:00:00Z')."),
    lead: int = Query(..., description="Lead time in hours to add to the base time."),
    bbox: str = Query(None, description="Bounding box in Web-Mercator meters as 'x_min,y_min,x_max,y_max'.")
) -> StreamingResponse:
    """
    Generate and return a heatmap image for a dataset at a specified valid time and spatial subset.

    This endpoint:
      - Loads the Zarr-backed dataset for the specified index.
      - Determines the base file timestamp from the source encoding.
      - Computes the requested valid time offset from the base timestamp.
      - Identifies the closest available time slice in the dataset.
      - Subsets the data by the provided Web-Mercator bbox or uses the full domain.
      - Reprojects the subset to EPSG:3857 and renders a heatmap image.
      - Returns the image as a PNG stream with an 'X-Extent-3857' response header.

    Parameters:
        index (str): Dataset identifier, must match one of the supported indices.
        base (str): Base timestamp in ISO 8601 format.
        lead (int): Number of hours relative to the base time for the valid slice.
        bbox (str, optional): Subset bounding box in Web-Mercator meters ('x_min,y_min,x_max,y_max').

    Returns:
        StreamingResponse: PNG image stream of the heatmap. The response header 'X-Extent-3857'
                          contains the image extent in Web-Mercator meters as 'left,right,bottom,top'.

    Raises:
        JSONResponse 400: Returns a JSON error message if processing or plotting fails.
    """
    try:
        ds = load_zarr(index)
        param = list(ds.data_vars.keys())[0]
        logger.info(f"📥 Loaded index: {index}, param: {param}")

        encoding_source = str(ds.encoding.get("source", ""))
        match = re.search(rf"{index}_(\d{{10}})", encoding_source)
        base_time_str = match.group(1) if match else "2024120100"
        base_time = datetime.strptime(base_time_str, "%Y%m%d%H")
        logger.info(f"📆 Base file time: {base_time.isoformat()}")

        base_dt = datetime.fromisoformat(base.replace("Z", ""))
        valid_hours = (base_dt - base_time).total_seconds() / 3600 + lead
        logger.info(f"🕐 Requested valid hours from base: {valid_hours}")

        time_in_hours = np.array([
            (pd.to_datetime(t).to_pydatetime() - base_time).total_seconds() / 3600
            for t in ds.time.values
        ])
        time_index = int(np.argmin(np.abs(time_in_hours - valid_hours)))
        logger.info(f"🧭 Closest time index: {time_index}, Available times: {time_in_hours.tolist()}")

        if bbox:
            # Order: x_min,y_min,x_max,y_max (meters Web-Mercator)
            x_min, y_min, x_max, y_max = map(float, bbox.split(','))
            logger.info(f"📥 bbox 3857 received: {bbox}")

            # Convert Web-Mercator corner coordinates to lat/lon (EPSG:4326) for subsetting
            transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
            lon_min, lat_min = transformer.transform(x_min, y_min)  # SW
            lon_max, lat_max = transformer.transform(x_max, y_max)  # NE

            lat_min, lon_min, lat_max, lon_max = normalize_bbox(
                lat_min, lon_min, lat_max, lon_max
            )
        else:
            lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())
            lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
            lat_min, lon_min, lat_max, lon_max = normalize_bbox(
                lat_min, lon_min, lat_max, lon_max
            )
            logger.info("🛑 No bbox provided – using full domain")

        lat_mask = (ds.lat >= lat_min) & (ds.lat <= lat_max)
        lon_mask = (ds.lon >= lon_min) & (ds.lon <= lon_max)

        subset = ds[param].isel(time=time_index).where(lat_mask & lon_mask, drop=True)
        logger.info(f"🗺️ Subset lat range: {float(subset.lat.min())} → {float(subset.lat.max())}")
        logger.info(f"🗺️ Subset lon range: {float(subset.lon.min())} → {float(subset.lon.max())}")

        # Reproject the subset to EPSG:3857 (Web-Mercator)
        subset_rio = (
            subset
            .rio.write_crs(4326)  # lat/lon
            .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        )

        subset_3857 = subset_rio.rio.reproject(
            "EPSG:3857",
            resampling=rasterio.enums.Resampling.bilinear
        )

        data = np.nan_to_num(subset_3857.values, nan=0.0)

        # coordinates in meters
        x = subset_3857.x.values
        y = subset_3857.y.values

        # Flip vertically so that origin='upper' matches the data orientation
        if y[0] < y[-1]:
            data = np.flipud(data)

        # extent (left, right, bottom, top) in meters
        x_res = abs(x[1] - x[0])
        y_res = abs(y[1] - y[0])
        extent = [
            x.min() - x_res / 2,
            x.max() + x_res / 2,
            y.min() - y_res / 2,
            y.max() + y_res / 2,
        ]

        # Calculate figure proportions (meters east-west and north-south)
        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        aspect_ratio = x_range / y_range if y_range else 1
        height = 5
        width = height * aspect_ratio

        fig, ax = plt.subplots(figsize=(width, height), dpi=150)
        ax.axis("off")
        ax.set_aspect("auto")

        masked_data = np.ma.masked_invalid(data)
        cmap = plt.get_cmap("YlOrRd").copy()
        cmap.set_bad(color=(0, 0, 0, 0))

        logger.info(f"🖼️ Final extent used in imshow (EPSG:3857): {extent}")

        ax.imshow(masked_data, extent=extent, origin="upper")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        logger.info(f"✅ Heatmap image generated for {index} [lead={lead}, bbox={bbox}]")
        response = StreamingResponse(buf, media_type="image/png")
        response.headers["X-Extent-3857"] = ",".join(map(str, extent))  # left,right,bottom,top
        return response

    except Exception as e:
        logger.exception("🔥 Heatmap generation failed")
        return JSONResponse(status_code=400, content={"error": str(e)})

