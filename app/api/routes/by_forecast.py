from fastapi import APIRouter, Query, Path, Depends, HTTPException
from fastapi.responses import JSONResponse
from datetime import timedelta
import pandas as pd

from app.utils.zarr_loader import _load_zarr
from app.utils.time_utils import _parse_naive, _iso_utc
from config.logging_config import logger

router = APIRouter()

@router.get("/{index}/by_forecast")
def get_forecast_evolution_steps(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(
        ..., description="Verification time ISO8601 (e.g., '2025-07-11T00:00:00' or '2025-07-11T00:00:00Z')."
    ),
) -> dict:
    """Retrieve the temporal sequence of forecast targets for a given verification time.

    This endpoint generates the list of forecast time steps covering the 9 days leading
    up to a selected **verification time** (``base_time`` in the request). The temporal
    resolution depends on the dataset:

    - **POF**: daily targets (one per day).
    - **FOPI**: 3‑hourly targets (eight per day).

    Parameters
    ----------
    index : str
        Dataset identifier, either ``"pof"`` or ``"fopi"``.
    base_time : str
        Verification time in ISO8601 format (e.g., ``"2025-07-11T00:00:00Z"``).
    session : Session
        Database session (injected by FastAPI, currently unused).

    Returns
    -------
    dict
        Dictionary with:
        - ``index``: dataset name (fopi or pof).
        - ``base_time``: selected verification time in ISO8601 UTC.
        - ``forecast_steps``: list of ISO8601 UTC timestamps representing forecast
          targets within the 9‑day lookback window at the dataset's native temporal resolution.

    Raises
    ------
    HTTPException
        400 if the index is unsupported or required coordinates are missing.
        404 if no forecast steps are found in the requested range.
    """
    try:
        ds = _load_zarr(index, base_time)

        # Parse the requested verification time to timezone‑naive, second precision (UTC)
        verification = _parse_naive(base_time)
        start_date = verification - timedelta(days=9)
        logger.info(
            f"🔍 Selected verification (target) time: {_iso_utc(verification)} — building steps in [{start_date}, {verification}]"
        )

        name = index.lower()
        if name == "pof":
            # POF is daily: generate daily targets from start_date..verification (inclusive)
            start_d = pd.Timestamp(start_date.date())
            end_d = pd.Timestamp(verification.date())
            rng = pd.date_range(start=start_d, end=end_d, freq="D")
            forecast_steps = [_iso_utc(ts.to_pydatetime()) for ts in rng]
        elif name == "fopi":
            # FOPI is 3‑hourly: align start to the nearest 3‑hour boundary <= start_date
            start_ts = pd.Timestamp(start_date)
            aligned = start_ts.floor("3H")
            rng = pd.date_range(start=aligned, end=pd.Timestamp(verification), freq="3H")
            forecast_steps = [_iso_utc(ts.to_pydatetime()) for ts in rng]
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported index '{index}'. Use 'pof' or 'fopi'.")

        if not forecast_steps:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No forecast targets could be generated in the range {start_date.isoformat()} to "
                    f"{verification.isoformat()} for index '{index}'."
                ),
            )

        logger.info(
            "🌋🌋🌋 FORECAST steps (by_forecast): first=%s last=%s count=%d",
            forecast_steps[0], forecast_steps[-1], len(forecast_steps),
        )
        return {
            "index": index,
            "base_time": _iso_utc(verification),
            "forecast_steps": forecast_steps,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("❌ Failed to get forecast evolution steps (by_forecast)")
        return JSONResponse(status_code=400, content={"error": str(e)})



@router.get("/{index}/by_forecast/heatmap/image")
def get_forecast_heatmap_image(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(..., description="Forecast initialization time (ISO 8601, e.g. 2025-07-05T00:00:00Z)"),
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
        base_time (str): ISO 8601 forecast initialization time.
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

    """
    TEMPORARY: Return an empty PNG image for all forecast heatmap requests.
    """
    from io import BytesIO
    from PIL import Image
    try:
        # Generate an empty white image (e.g., 512x512)
        width, height = 512, 512
        image = Image.new("RGBA", (width, height), (255, 255, 255, 0))  # Transparent background
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        response = StreamingResponse(buffer, media_type="image/png")
        response.headers["X-Extent-3857"] = "0,0,0,0"
        response.headers["X-Scale-Min"] = "0"
        response.headers["X-Scale-Max"] = "0"
        logger.info(f"⚠️ Returned empty image for {index} [{base_time}, step={step}, bbox={bbox}]")
        return response

    except Exception as e:
        logger.exception("Failed to return empty heatmap image")
        return JSONResponse(status_code=400, content={"error": str(e)})
    """try:
        # Under the hood, everything else (bbox, projection, scaling) works as in current solution
        image_stream, extent, vmin, vmax = generate_heatmap_image(
            index, base_time, step, bbox
        )
        response = StreamingResponse(image_stream, media_type="image/png")
        response.headers["X-Extent-3857"] = ",".join(map(str, extent))
        response.headers["X-Scale-Min"] = str(vmin)
        response.headers["X-Scale-Max"] = str(vmax)
        logger.info(f"✅  Heatmap image generated for {index} [{base_time}, step={step}, bbox={bbox}]")
        return response

    except Exception as e:
        logger.exception("Heatmap generation failed")
        return JSONResponse(status_code=400, content={"error": str(e)})"""
