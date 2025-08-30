
from fastapi import APIRouter, Query, Path, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from urllib.parse import unquote
import pandas as pd

from utils.zarr_handler import _load_zarr
from utils.time_utils import (
    _iso_drop_tz,
    _iso_utc,
)
from utils.stats import _agg_mean_median
from utils.bounds_utils import _extract_spatial_subset

from config.config import settings
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/time_series")
async def time_series(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    bbox: str = Query(None, description="EPSG:3857 bbox as 'x_min,y_min,x_max,y_max' (e.g., '1033428.6224155831%2C4259682.712276304%2C2100489.537276644%2C4770282.061221281')"),
    start_base: Optional[str] = Query(None, description="Filter runs from this base_time (inclusive). Base time ISO8601 (e.g., '2025-07-03T00:00:00Z')."),
    end_base: Optional[str]   = Query(None, description="Filter runs up to this base_time (inclusive). Base time ISO8601 (e.g., '2025-07-07T00:00:00Z')."),
):
    """
    Retrieve run-to-run summary statistics for a given fire danger index.

    Path
    ----
    GET /{index}/time_series

    This endpoint loads a time series from the Zarr dataset for the specified
    `index` (e.g., "pof" or "fopi"), optionally filters runs by a spatial
    bounding box in EPSG:3857 and/or a base time range, and computes the
    mean and median for each selected run.

    Parameters
    ----------
    index : str
        Dataset identifier, such as "fopi" or "pof".
    bbox : str, optional
        Bounding box in EPSG:3857 coordinates, formatted as
        `"x_min,y_min,x_max,y_max"`. May be URL-encoded
        (commas as `%2C`). If provided, only data within this
        spatial extent is included in the statistics.
    start_base : str, optional
        ISO8601 datetime string (e.g., `"2025-07-03T00:00:00Z"`)
        for the earliest `base_time` to include (inclusive).
    end_base : str, optional
        ISO8601 datetime string for the latest `base_time` to
        include (inclusive).

    Returns
    -------
    dict
        JSON object containing:
          - `index` (str): Lowercase dataset identifier.
          - `mode` (str): Always `"by_base_time"`.
          - `stat` (list[str]): List of statistics included, `["mean", "median"]`.
          - `bbox_epsg3857` (str | None): Human-readable bounding box or `null`.
          - `timestamps` (list[str]): Base time run timestamps in UTC ISO format.
          - `mean` (list[float | None]): Mean values per run, `None` if unavailable.
          - `median` (list[float | None]): Median values per run, `None` if unavailable.

    Notes
    -----
    - This function computes statistics eagerly (fully loading the data into memory).
    - NaN-only runs produce `None` for both mean and median.
    - The output is designed for charting, with the x-axis representing `base_time`.
    """
    try:
        # Load Zarr and resolve the variable name based on index
        ds = _load_zarr(index)
        try:
            var_name = settings.VAR_NAMES[index]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Unknown index '{index}': {e}")
        if var_name not in ds.data_vars:
            raise HTTPException(status_code=400, detail=f"Variable '{var_name}' not found in dataset for '{index}'.")

        da = ds[var_name]

        # Normalize run list (base_time is a coordinate)
        if "base_time" not in ds.coords:
            raise HTTPException(status_code=400, detail="Dataset is missing 'base_time' coordinate.")
        base_vals = pd.to_datetime(ds["base_time"].values)
        # Make naive + second precision
        base_vals = [pd.Timestamp(bv).tz_localize(None).replace(microsecond=0) for bv in base_vals]

        # Optional base_time filtering
        if start_base:
            sb = _iso_drop_tz(start_base)
            base_vals = [bt for bt in base_vals if bt >= sb]
        if end_base:
            eb = _iso_drop_tz(end_base)
            base_vals = [bt for bt in base_vals if bt <= eb]
        if not base_vals:
            raise HTTPException(status_code=404, detail="No base_time runs found for the given filters.")

        # Select only those runs (works whether base_time is a dim or coord)
        da_sel = da.sel(base_time=base_vals)

        # Optional spatial subsetting (EPSG:3857 bbox -> EPSG:4326 inside utility)
        if bbox:
            try:
                da_sel = _extract_spatial_subset(da_sel, bbox=bbox)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid bbox or extraction error: {e}")

        # ---- Compute run-to-run stats without Dask; use _agg_mean_median ----
        mean_vals: list[float | None] = []
        median_vals: list[float | None] = []

        # Preserve the order of the selected runs as in da_sel
        bt_coord = pd.to_datetime(da_sel["base_time"].values)
        for bt in bt_coord:
            da_bt = da_sel.sel(base_time=bt)
            # _agg_mean_median reduces over lat/lon; if forecast_index remains,
            # it will also be included in the aggregation (intended behavior).
            m, md = _agg_mean_median(da_bt)
            mean_vals.append(m)
            median_vals.append(md)

        # Base-time timestamps → ISO strings (UTC with trailing 'Z')
        timestamps_iso = [_iso_utc(pd.Timestamp(t)) for t in bt_coord]

        response = {
            "index": index.lower(),
            "mode": "by_base_time",
            "stat": ["mean", "median"],
            "bbox_epsg3857": unquote(bbox) if bbox else None,
            "timestamps": timestamps_iso,   # x-axis is base_time runs
            "mean": mean_vals,
            "median": median_vals,
        }

        logger.info(f"Time series response: {response}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to build run-to-run series")
        return JSONResponse(status_code=400, content={"error": str(e)})
