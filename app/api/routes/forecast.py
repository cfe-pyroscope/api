# app/api/routes/forecast.py

from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime, timedelta
from app.services.zarr_loader import convert_nc_to_zarr, load_zarr
from app.services.time_utils import extract_base_time_from_encoding
from app.services.heatmap_generator import generate_heatmap_image
from app.logging_config import logger

router = APIRouter()


@router.get("/{index}/forecast")
async def get_forecast_init_steps(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    forecast_init: str = Query(..., description="Forecast initialization time (ISO 8601, e.g. 2025-07-05T00:00:00Z)")
):
    """
    Retrieve available forecast steps for a given dataset and initialization time.

    This endpoint loads forecast data for the specified dataset (`index`) and forecast
    initialization time (`forecast_init`), ensures the underlying Zarr store is prepared
    (converted from NetCDF if needed), and returns a list of time steps with corresponding
    lead times (in hours). It's used by the frontend to populate the forecast time slider.

    Args:
        index (str): Dataset identifier, e.g. "fopi" or "pof".
        forecast_init (str): Forecast run initialization time in ISO 8601 format.

    Returns:
        dict: A dictionary containing:
            - `forecast_init`: The initialization time requested.
            - `location`: [lat, lon] center of the forecast grid.
            - `forecast_steps`: A list of steps, each with:
                - `time`: Forecast time in ISO format.
                - `lead_hours`: Hours since initialization.

    Raises:
        400 Bad Request: If data loading or processing fails for any reason.
    """
    try:
        # Always use only the run specified in forecast_init (frontend gets available runs from /available-dates)
        # Ensure Zarr exists, convert from NC if needed
        ds = load_zarr(index, forecast_init)
        file_base_time = extract_base_time_from_encoding(ds, index)
        lat_center = float(ds.lat.mean())
        lon_center = float(ds.lon.mean())

        # Prepare the forecast steps array (compatible with ForecastSlider)
        forecast_steps = []
        if index == "fopi":
            for t in ds.time.values:
                try:
                    t_val = float(t)
                    step_time = file_base_time + timedelta(hours=t_val)
                    forecast_steps.append({
                        "time": step_time.isoformat() + "Z",
                        "lead_hours": int(t_val)
                    })
                except Exception as e:
                    logger.warning(f"Skipping invalid time value in fopi: {t} ({e})")
        else:  # pof
            for t in ds.time.values:
                step_time = str(t)
                t_dt = datetime.fromisoformat(str(t).replace("Z", ""))
                lead_hours = int((t_dt - file_base_time).total_seconds() // 3600)
                forecast_steps.append({
                    "time": t_dt.isoformat() + "Z",
                    "lead_hours": lead_hours
                })

        return {
            "forecast_init": forecast_init,
            "location": [lat_center, lon_center],
            "forecast_steps": forecast_steps
        }

    except Exception as e:
        logger.exception("Failed to get forecast steps for run")
        return JSONResponse(status_code=400, content={"error": str(e)})


@router.get("/{index}/forecast/heatmap/image")
def get_forecast_heatmap_image(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    forecast_init: str = Query(..., description="Forecast initialization time (ISO 8601, e.g. 2025-07-05T00:00:00Z)"),
    step: int = Query(..., description="Forecast lead step in hours (e.g., 0, 3, ..., 240 for FOPI; 24, ..., 240 for POF)"),
    bbox: str = Query(None, description="Bounding box in EPSG:3857 as 'x_min,y_min,x_max,y_max' (optional).")
):
    """
    Generate and return a forecast heatmap image as a PNG.

    This endpoint renders a heatmap based on forecast data for a given dataset (`index`),
    initialization time, and lead time (`step`). An optional bounding box (`bbox`) can
    be used to crop the output image spatially.

    Args:
        index (str): Dataset identifier ('fopi' or 'pof').
        forecast_init (str): ISO 8601 forecast initialization time.
        step (int): Forecast lead time in hours.
        bbox (str, optional): Optional bounding box in EPSG:3857 format (x_min,y_min,x_max,y_max).

    Returns:
        StreamingResponse: PNG image with additional headers:
            - X-Extent-3857: The spatial extent of the image in EPSG:3857.
            - X-Scale-Min: Minimum value of the data used for scaling.
            - X-Scale-Max: Maximum value of the data used for scaling.

    Raises:
        400 Bad Request: If the image generation fails for any reason.
    """
    try:
        # Under the hood, everything else (bbox, projection, scaling) works as in current solution
        image_stream, extent, vmin, vmax = generate_heatmap_image(
            index, forecast_init, step, bbox
        )
        response = StreamingResponse(image_stream, media_type="image/png")
        response.headers["X-Extent-3857"] = ",".join(map(str, extent))
        response.headers["X-Scale-Min"] = str(vmin)
        response.headers["X-Scale-Max"] = str(vmax)
        logger.info(f"✅ Heatmap image generated for {index} [{forecast_init}, step={step}, bbox={bbox}]")
        return response

    except Exception as e:
        logger.exception("Heatmap generation failed")
        return JSONResponse(status_code=400, content={"error": str(e)})
