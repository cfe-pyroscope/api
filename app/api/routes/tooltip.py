from fastapi import APIRouter, Query, Path
from fastapi.responses import JSONResponse
from pyproj import Transformer
import numpy as np
import pandas as pd

from config.logging_config import logger

from app.utils.zarr_handler import _load_zarr, _select_first_param
from app.utils.bounds_utils import _parse_coords
from app.utils.time_utils import _normalize_times, _iso_utc

router = APIRouter()


@router.get("/{index}/tooltip")
def get_tooltip_data(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    base_time: str = Query(..., description="Base time ISO 8601 (e.g., '2025-07-11T00:00:00Z')."),
    forecast_time: str = Query(..., description="Forecast time ISO 8601 (e.g., '2025-07-14T00:00:00Z')."),
    coords: str = Query(..., description="EPSG:3857 as 'x,y' (e.g., '2617356.7225410054, -990776.760632454')")
) -> JSONResponse:
    """
    Retrieve tooltip data (single-point sample) for a given dataset/time/coords.

    Strict behavior: the provided times must exactly exist in the dataset.

    Example:
        GET /api/fopi/tooltip?base_time=2025-07-11T00:00:00Z&forecast_time=2025-07-14T00:00:00Z&coords=2617356.7225410054, -990776.760632454
    """
    try:
        # coords -> lon/lat
        x3857, y3857 = _parse_coords(coords)
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(x3857, y3857)

        # load dataset and pick variable
        ds = _load_zarr(index)
        param = _select_first_param(ds)  # excludes 'forecast_time' by design

        # normalize times (naive UTC, second precision)
        req_base, req_fcst = _normalize_times(base_time, forecast_time)

        # slice EXACT base_time then find EXACT forecast_index by matching forecast_time data var
        ds_bt = ds.sel(base_time=req_base)  # base_time IS a coordinate

        # normalize dataset forecast_time values to naive UTC seconds
        fcst_vals = pd.to_datetime(ds_bt["forecast_time"].values)
        fcst_vals = [pd.Timestamp(v).tz_localize(None).replace(microsecond=0) for v in fcst_vals]

        # take the index
        fcst_idx = next(i for i, v in enumerate(fcst_vals) if v == req_fcst)

        # extract the 2D field
        da = ds_bt[param].isel(forecast_index=fcst_idx)

        # sample nearest grid point
        picked = da.sel(lon=lon, lat=lat, method="nearest")
        value = picked.values
        if isinstance(value, np.ndarray):
            value = value.item()
        val = None if (value is None or (isinstance(value, float) and np.isnan(value))) else float(value)

        logger.info(f"Tooltip value: {val}")

        return JSONResponse(status_code=200, content={
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
                "base_time": _iso_utc(req_base),
                "forecast_time": _iso_utc(req_fcst),
            },
        })
    except Exception as e:
        logger.exception("🛑 Tooltip data retrieval failed")
        return JSONResponse(status_code=400, content={"error": str(e)})

