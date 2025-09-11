
from fastapi import APIRouter, Query, Path, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from urllib.parse import unquote
import pandas as pd

from utils.zarr_handler import _load_zarr
from utils.time_utils import (
    _iso_drop_tz,
    _iso_utc_str,
)
from utils.stats import _agg_mean_median
from utils.bounds_utils import _extract_spatial_subset, _bbox_to_latlon

from config.config import settings
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/time_series")
async def time_series(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    bbox: str = Query(None, description="EPSG:3857 bbox as 'x_min,y_min,x_max,y_max' (e.g., '1033428.6224155831%2C4259682.712276304%2C2100489.537276644%2C4770282.061221281')"),
    start_base: Optional[str] = Query(None, description="Filter runs from this base_time (inclusive). Base time ISO8601 (e.g., '2025-09-01T00:00:00Z')."),
    end_base: Optional[str]   = Query(None, description="Filter runs up to this base_time (inclusive). Base time ISO8601 (e.g., '2025-09-04T00:00:00Z')."),
):
    """
    Build a run-to-run time series (mean & median) over the dataset variable for `index`.
    Optionally filters base_time to [start_base, end_base] (inclusive) and subsets
    spatially by `bbox` (EPSG:3857). Returns ISO8601 UTC timestamps (Z) for each run
    and the corresponding mean/median values aggregated over the spatial slice.
    """
    try:
        ds = _load_zarr(index)
        var_name = settings.VAR_NAMES[index]
        da = ds[var_name]

        base_vals = pd.to_datetime(ds["base_time"].values)
        base_vals = [pd.Timestamp(bv).tz_localize(None).replace(microsecond=0) for bv in base_vals]
        base_vals_sorted = sorted(base_vals)

        if start_base:
            sb = _iso_drop_tz(start_base)
            base_vals_sorted = [bt for bt in base_vals_sorted if bt >= sb]
        if end_base:
            eb = _iso_drop_tz(end_base)
            base_vals_sorted = [bt for bt in base_vals_sorted if bt <= eb]

        # Select only those runs
        da_sel = da.sel(base_time=base_vals_sorted)

        # Optional spatial subsetting (EPSG:3857 bbox -> EPSG:4326 inside utility)
        da_sel = _extract_spatial_subset(da_sel, bbox=bbox)

        ### **************************************************** ###
        # ---- diagnostics: shape, ranges, data presence ----

        def _safe_coord_range(da, name):
            if name not in da.coords:
                return None
            n = int(da.sizes.get(name, 0))
            if n == 0:
                return None
            c = da.coords[name]
            try:
                mn = float(c.min().item())
                mx = float(c.max().item())
            except Exception:
                # dask-backed or other edge cases
                try:
                    mn = float(c.min().compute().item())
                    mx = float(c.max().compute().item())
                except Exception:
                    return None
            return (mn, mx)

        sizes = {k: int(v) for k, v in da_sel.sizes.items()}

        lon_rng = _safe_coord_range(da_sel, "lon")
        lat_rng = _safe_coord_range(da_sel, "lat")

        # total non-null values across all dims
        if da_sel.size == 0:
            non_null_total = 0
        else:
            try:
                non_null_total = int(da_sel.count().item())
            except Exception:
                non_null_total = int(da_sel.count().compute().item())

        # boolean: do we have any finite data anywhere?
        if da_sel.size == 0:
            has_any_data = False
        else:
            try:
                has_any_data = bool(da_sel.notnull().any().item())
            except Exception:
                has_any_data = bool(da_sel.notnull().any().compute().item())

        # per-run preview: non-null counts per base_time (first 5)
        per_run_head = None
        if "base_time" in da_sel.dims and da_sel.sizes.get("base_time", 0) > 0:
            reduce_dims = [d for d in ("lat", "lon") if d in da_sel.dims]
            per_run = da_sel.count(dim=reduce_dims)
            try:
                per_run_head = per_run.isel(base_time=slice(0, 5)).astype("int64").values.tolist()
            except Exception:
                per_run_head = (
                    per_run.isel(base_time=slice(0, 5)).compute().astype("int64").values.tolist()
                )

        logger.info(
            "🔥🔥🔥🔥 da_sel: sizes=%s lon_rng=%s lat_rng=%s non_null_total=%d has_any_data=%s per_run_head=%s",
            sizes, lon_rng, lat_rng, non_null_total, has_any_data, per_run_head,
        )

        # ---- END diagnostics ----
        ### **************************************************** ###


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
        timestamps_iso = [_iso_utc_str(pd.Timestamp(t)) for t in bt_coord]

        if bbox:
            lon_min, lat_min, lon_max, lat_max = _bbox_to_latlon(bbox)
            bbox_latlon_flat = (lat_min, lon_min, lat_max, lon_max)
        else:
            bbox_latlon_flat = None

        response = {
            "index": index.lower(),
            "mode": "by_base_time",
            "stat": ["mean", "median"],
            "bbox_epsg3857": unquote(bbox) if bbox else None,
            "bbox_epsg4326": bbox_latlon_flat,  # lon_min, lat_min, lon_max, lat_max
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
