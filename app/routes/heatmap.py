from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse, StreamingResponse
from utils.heatmap_generator import generate_heatmap_image
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/heatmap/image")
def get_heatmap_image(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(..., description="Base time in ISO 8601 format (e.g., '2025-09-02T00:00:00Z')."),
    forecast_time: str = Query(..., description="Forecast time in ISO 8601 format (e.g., '2025-09-05T00:00:00Z')."),
    bbox: str = Query(None, description="EPSG:3857 bbox as 'x_min,y_min,x_max,y_max' (e.g., '1033428.6224155831%2C4259682.712276304%2C2100489.537276644%2C4770282.061221281')")
) -> StreamingResponse:
    """
    Retrieve a generated heatmap image for a given dataset, time range, and spatial subset.
    This endpoint produces a PNG heatmap image representing data for the specified
    `index` at the given `base_time` and `forecast_time`. The response includes metadata
    in custom headers for the map extent and the data scale range.

    Example:
        GET /fopi/heatmap/image?base_time=2025-09-02T00:00:00&forecast_time=2025-09-05T00:00:00&bbox=-8237642,4970351,-8235642,4972351
    """
    try:
        image_stream, extent, vmin, vmax = generate_heatmap_image(index, base_time, forecast_time, bbox)

        response = StreamingResponse(image_stream, media_type="image/png")
        response.headers["X-Extent-3857"] = ",".join(map(str, extent))
        response.headers["X-Scale-Min"] = str(vmin)
        response.headers["X-Scale-Max"] = str(vmax)

        logger.info(f"✅  Heatmap image generated for {index} [base_time={base_time}, forecast_time={forecast_time}, bbox={bbox}]")
        return response

    except Exception as e:
        logger.exception("❌ Heatmap generation failed")
        return JSONResponse(status_code=400, content={"error": str(e)})