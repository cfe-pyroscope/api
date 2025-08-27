from typing import List
from fastapi import APIRouter, Path, Query, HTTPException
from fastapi.responses import StreamingResponse
from config.logging_config import logger

router = APIRouter()

@router.get("/{index}/heatmap_series/images")
def get_heatmap_series(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    forecast_time: List[str] = Query(..., description="Repeat `forecast_time` per value"),
    bbox: str = Query(..., description="EPSG:3857 bbox as 'x_min,y_min,x_max,y_max'")
) -> StreamingResponse:

    # also support accidental comma-joined single item
    if len(forecast_time) == 1 and "," in forecast_time[0]:
        forecast_time = [s.strip() for s in forecast_time[0].split(",") if s.strip()]

    logger.info("⚽⚽⚽ forecast_time: %s", forecast_time)



