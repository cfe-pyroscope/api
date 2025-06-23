from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse, StreamingResponse
from app.services.heatmap_generator import generate_heatmap_image
from app.logging_config import logger

router = APIRouter()


@router.get("/{index}/heatmap/image")
def get_heatmap_image(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(..., description="Base time in ISO 8601 format (e.g., '2025-06-20T00:00:00Z')."),
    lead_hours: int = Query(..., description="Lead time in hours to add to the base time."),
    bbox: str = Query(None, description="Bounding box in Web-Mercator meters as 'x_min,y_min,x_max,y_max'.")
) -> StreamingResponse:
    """
    Generate and return a heatmap image for a dataset at a specified valid time and spatial subset.

    Parameters:
        index (str): Dataset identifier.
        base_time (str): ISO 8601 base timestamp.
        lead_hours (int): Lead time in hours.
        bbox (str, optional): Bounding box in Web-Mercator meters.

    Returns:
        StreamingResponse: PNG image stream with header 'X-Extent-3857'.
    """
    try:
        image_stream, extent = generate_heatmap_image(index, base_time, lead_hours, bbox)

        response = StreamingResponse(image_stream, media_type="image/png")
        response.headers["X-Extent-3857"] = ",".join(map(str, extent))
        logger.info(f"✅ Heatmap image generated for {index} [lead={lead_hours}, bbox={bbox}]")
        return response

    except Exception as e:
        logger.exception("🔥 Heatmap generation failed")
        return JSONResponse(status_code=400, content={"error": str(e)})
