from fastapi import APIRouter, Query, Path, HTTPException, Response
from fastapi.responses import JSONResponse
from typing import Optional
from urllib.parse import unquote
import pandas as pd
import numpy as np

from utils.zarr_handler import _load_zarr
from utils.time_utils import _iso_drop_tz, _iso_utc_str
from utils.bounds_utils import _extract_spatial_subset, _bbox_to_latlon

from config.config import settings
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/difference_map")
async def difference_map(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    bbox: Optional[str] = Query(None, description="EPSG:3857 bbox as 'x_min,y_min,x_max,y_max'"),
    base_time_start: str = Query(..., description="Start base_time ISO8601 (e.g. '2025-09-01T00:00:00Z')."),
    base_time_end: str = Query(..., description="End base_time ISO8601 (e.g. '2025-09-02T00:00:00Z')."),
):
    """
    Compute the per-gridcell difference (end - start) in the selected index.
    Matching is done by DATE (ignoring time-of-day); the actual dataset timestamps
    corresponding to those dates are used for selection.
    Returns lat, lon, and delta arrays for mapping difference of fire risk.
    """
    try:
        ds = _load_zarr(index)
        var_name = settings.VAR_NAMES[index]
        da = ds[var_name]

        start_date = pd.Timestamp(_iso_drop_tz(base_time_start)).tz_localize(None).normalize()
        end_date   = pd.Timestamp(_iso_drop_tz(base_time_end)).tz_localize(None).normalize()

        base_vals = pd.to_datetime(ds["base_time"].values)
        base_vals = [pd.Timestamp(bv).tz_localize(None).replace(microsecond=0) for bv in base_vals]
        base_dates = pd.DatetimeIndex(base_vals).normalize()

        # Map each date to its (unique) actual dataset timestamp
        date_to_bt = {}
        for bt, d in zip(base_vals, base_dates):
            # If duplicates per day ever appear, this keeps the first occurrence.
            if d not in date_to_bt:
                date_to_bt[d] = bt

        start_bt = date_to_bt[start_date]
        end_bt   = date_to_bt[end_date]

        da_pair = da.sel(base_time=[start_bt, end_bt])

        da_pair = _extract_spatial_subset(da_pair, bbox=bbox)
        da_start = da_pair.sel(base_time=start_bt)
        da_end   = da_pair.sel(base_time=end_bt)
        da_diff  = da_end - da_start

        lats_arr = da_diff["lat"].values
        lons_arr = da_diff["lon"].values

        lat_obj = np.asarray(lats_arr).astype(object)
        lon_obj = np.asarray(lons_arr).astype(object)
        lats = lat_obj.tolist()
        lons = lon_obj.tolist()

        delta_arr = da_diff.values
        delta_arr = np.asarray(delta_arr)
        if delta_arr.ndim > 2:
            delta_arr = np.squeeze(delta_arr)

        mask = ~np.isfinite(delta_arr.astype(float, copy=False))
        delta_obj = delta_arr.astype(object, copy=True)
        if mask.any():
            delta_obj[mask] = None
        delta_list = delta_obj.tolist()

        if bbox:
            lon_min, lat_min, lon_max, lat_max = _bbox_to_latlon(bbox)
            bbox_epsg4326 = (lat_min, lon_min, lat_max, lon_max)
            bbox_epsg3857 = unquote(bbox)
        else:
            bbox_epsg4326 = None
            bbox_epsg3857 = None

        # ---- Response ----
        response = {
            "index": index.lower(),
            "mode": "difference_map",
            "base_time_start": _iso_utc_str(pd.Timestamp(start_bt)),
            "base_time_end": _iso_utc_str(pd.Timestamp(end_bt)),
            "bbox_epsg3857": bbox_epsg3857,
            "bbox_epsg4326": bbox_epsg4326,  # (lat_min, lon_min, lat_max, lon_max)
            "lats": lats,
            "lons": lons,
            "delta": delta_list,
            "notes": (
                "Each grid cell value represents the difference between the selected index "
                "at base_time_end minus its value at base_time_start. "
                "Inputs are matched by calendar date (ignoring time-of-day), and only the "
                "grid cells within the provided bounding box are considered. "
                "Non-finite values (NaN/Inf) are returned as null in JSON output."
            ),
        }

        logger.info(
            "Difference map response: index=%s requested_dates=(%s,%s) actual_bt=(%s,%s) shape=%s",
            index.lower(), start_date.date(), end_date.date(), start_bt, end_bt, np.shape(delta_list)
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to build difference map")
        return JSONResponse(status_code=400, content={"error": str(e)})
