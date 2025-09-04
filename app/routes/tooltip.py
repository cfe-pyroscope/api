from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse
from pyproj import Transformer
import numpy as np
import pandas as pd

from config.logging_config import logger

from utils.zarr_handler import _load_zarr, _select_first_param
from utils.bounds_utils import _parse_coords
from utils.time_utils import _normalize_times, _naive_utc_ndarray

router = APIRouter()


@router.get("/{index}/tooltip")
def get_tooltip_data(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(..., description="Base time ISO 8601 (e.g., '2025-09-02T00:00:00Z')."),
    forecast_time: str = Query(..., description="Forecast time ISO 8601 (e.g., '2025-09-05T00:00:00Z')."),
    coords: str = Query(..., description="EPSG:3857 as 'x,y' (e.g., '2617356.7225410054, -990776.760632454')")
) -> JSONResponse:
    """
    Get tooltip data for a dataset at a specific location and time.
    Converts input coordinates from EPSG:3857 to lon/lat, loads the Zarr
    dataset, and retrieves the nearest grid value for the first parameter
    (excluding `forecast_time`) at the requested `base_time` and
    `forecast_time`.
    """
    try:
        # coords -> lon/lat
        x3857, y3857 = _parse_coords(coords)
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x3857, y3857)

        # load dataset and pick variable
        ds = _load_zarr(index)
        param = _select_first_param(ds)  # excludes 'forecast_time' by design

        # normalize requested times to tz-naive UTC
        req_base, req_fcst = _normalize_times(base_time, forecast_time)

        logger.info(f"🍋🍋🍋 req_base: {req_base}")
        logger.info(f"🍋🍋🍋 req_fcst: {req_fcst}")

        ds = ds.assign_coords(
            base_time=_naive_utc_ndarray(ds["base_time"].values)
        )

        # if forecast_time is stored as a data variable (common), normalize after slicing base_time
        ds_bt = ds.sel(base_time=req_base)

        fcst_vals = _naive_utc_ndarray(ds_bt["forecast_time"].values)

        # 3) exact match on forecast_time (no nearest)
        matches = np.where(fcst_vals == req_fcst)[0]
        if matches.size == 0:
            raise KeyError(f"forecast_time {req_fcst} not found. "
                           f"Available: {fcst_vals.min()} .. {fcst_vals.max()} count={len(fcst_vals)}")
        fcst_idx = int(matches[0])

        da = ds_bt[param].isel(forecast_index=fcst_idx)
        picked = da.sel(lon=lon, lat=lat, method="nearest")

        value = picked.values
        if isinstance(value, np.ndarray):
            value = value.item()
        val = None if (value is None or (isinstance(value, float) and np.isnan(value))) else float(value)

        return JSONResponse(
            status_code=200,
            content={
                "index": index,
                "param": param,
                "value": val,
                "point": {
                    "input_epsg3857": {"x": float(x3857), "y": float(y3857)},
                    "lon": float(lon),
                    "lat": float(lat),
                    "nearest_grid": {
                        "lon": float(picked["lon"].values),
                        "lat": float(picked["lat"].values),
                    },
                },
                "time": {
                    # serialize as ISO Z when returning
                    "base_time": pd.Timestamp(req_base, tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "forecast_time": pd.Timestamp(req_fcst, tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            },
        )
    except Exception as e:
        logger.exception("❌ Tooltip data retrieval failed")
        return JSONResponse(status_code=400, content={"error": str(e)})

