
from fastapi import APIRouter, Query, Path, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from urllib.parse import unquote

import xarray as xr
from utils.zarr_handler import _load_zarr
from utils.bounds_utils import _extract_spatial_subset, _bbox_to_latlon
from utils.time_utils import _iso_utc_str


from config.config import settings
from config.logging_config import logger

from pyproj import Transformer
transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

router = APIRouter()

@router.get("/forecast_horizon")
async def forecast_horizon( 
    bbox: str = Query(None, description="EPSG:3857 bbox as 'x_min,y_min,x_max,y_max' (e.g., '-489196.9810251281,1565430.33928041,117407.27544603075,1898084.2863774975')"),
    start_base: Optional[str] = Query(None, description="Filter runs from this base_time (inclusive). Base time ISO8601 (e.g., '2025-09-01T00:00:00Z')."),
):
    try:
        ds_fopi = _load_zarr('fopi')
        ds_pof = _load_zarr('pof')

        # Get projected coordinates
        bbox_split = bbox.split(',')
        min_lon, min_lat = transformer.transform(bbox_split[0], bbox_split[1])
        max_lon, max_lat = transformer.transform(bbox_split[2], bbox_split[3])

        subset_pof = ds_pof.sel(
            lon=slice(min_lon, max_lon),
            lat=slice(max_lat, min_lat)
        )

        subset_fopi = ds_fopi.sel(
            lon=slice(min_lon, max_lon),
            lat=slice(max_lat, min_lat)
        )

        # print(subset_fopi)

        # collapse lat/lon for regional averages
        subset_mean_pof = subset_pof.mean(dim=["lat", "lon"])
        subset_mean_fopi = subset_fopi.mean(dim=["lat", "lon"])
  
        # most recent
        latest_run_pof = subset_mean_pof.sel(base_time=subset_mean_pof.base_time.max())
        latest_run_fopi = subset_mean_fopi.sel(base_time=subset_mean_fopi.base_time.max())
        
        # as list for forecast_index
        pof_values = latest_run_pof["MODEL_FIRE"]
        pof_values_list =[round(v,5) for v in pof_values.compute().values.tolist()]

        fopi_values = latest_run_fopi["param100.128.192"]
        fopi_values_list =[round(v,5) for v in fopi_values.compute().values.tolist()]

        # find max/min of values for setting the axes in front end + 10% margin
        def get_axis_values(list):
            min_list = min(list)
            max_list = max(list)
            margin = 0.1*(max_list- min_list)
         
            axis_min = min_list - margin
            axis_max = max_list + margin
            return [axis_min, axis_max]

        pof_axes = get_axis_values(pof_values_list)
        fopi_axes = get_axis_values(fopi_values_list)

        if bbox:
            lon_min, lat_min, lon_max, lat_max = _bbox_to_latlon(bbox)
            bbox_latlon_flat = (lat_min, lon_min, lat_max, lon_max)
        else:
            bbox_latlon_flat = None

        if start_base:
            base_date = start_base
        else:
            max_bt = subset_mean_pof.base_time.max().values
            base_date = _iso_utc_str(max_bt)

        response = {
            "base_date": base_date,
            "bbox_epsg3857": unquote(bbox) if bbox else None,
            "bbox_epsg4326": bbox_latlon_flat,
             "pof_forecast": pof_values_list,
             "fopi_forecast": fopi_values_list,
             "axes_pof": pof_axes,
             "axes_fopi": fopi_axes,
        }

        logger.info(f"Forecast horizon: {response}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        print(e)
        logger.exception("Failed to build run-to-run series")
        return JSONResponse(status_code=400, content={"error": str(e)})
