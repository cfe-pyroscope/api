from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime, timedelta
from app.services.zarr_loader import load_zarr
from app.services.time_utils import calculate_valid_times
from app.services.file_scanner import scan_storage_files
from app.logging_config import logger
from app.services.heatmap_generator import generate_heatmap_image
from config import settings

router = APIRouter()


@router.get("/{index}/forecast")
def get_forecast_steps(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    forecast_init: str = Query(..., description="ISO format base forecast date (e.g., 2025-07-11T00:00:00Z)")
):
    try:
        forecast_init_dt = datetime.fromisoformat(forecast_init.replace("Z", ""))
        start_date = forecast_init_dt - timedelta(days=9)

        logger.info(f"🔍 Selected forecast_init: {forecast_init_dt.isoformat()} — scanning from {start_date.date()}")

        scan_path = f"{settings.ZARR_PATH}/{index}"
        all_files = scan_storage_files(scan_path)
        relevant_files = [
            (ds_name, dt, path)
            for ds_name, dt, path in all_files
            if ds_name.lower() == index.lower()
            and start_date.date() <= dt.date() <= forecast_init_dt.date()
        ]

        logger.info(f"📂 Found {len(relevant_files)} matching files for index '{index}'")

        all_steps = []

        for _, file_base_time, _ in sorted(relevant_files, key=lambda x: x[1]):
            base_time_iso = file_base_time.isoformat() + "Z"
            ds = load_zarr(index, base_time_iso)
            steps = calculate_valid_times(ds, index, file_base_time, forecast_init_dt)

            for step in steps:
                all_steps.append({
                    "base_time": base_time_iso,
                    "lead_hours": step["lead_hours"],
                    "time": step["valid_time"].isoformat() + "Z"
                })

        all_steps.sort(key=lambda x: x["lead_hours"])
        return {"forecast_steps": all_steps}

    except Exception as e:
        logger.exception("❌ Failed to get forecast steps")
        return JSONResponse(status_code=500, content={"error": str(e)})


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
