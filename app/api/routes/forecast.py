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
    Returns all forecast steps for a given forecast initialization time (run).
    Ensures Zarr is prepared for this run.
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
    Generate a heatmap image for a specific forecast initialization and step (lead_hours).
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
