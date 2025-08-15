from fastapi import APIRouter, Query, Path, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from urllib.parse import unquote

import pandas as pd

from app.utils.zarr_loader import _load_zarr
from app.utils.time_utils import (
    _parse_iso_naive,
    _iso_utc,
)
from app.utils.stats import _agg_mean_median
from app.utils.bounds_utils import _extract_spatial_subset  # ← NEW

from config.config import settings
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/time_series")
async def time_series(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    bbox: Optional[str] = Query(
        None,
        description=(
            "EPSG:3857 bbox (e.g., "
            "'1033428.6224155831%2C4259682.712276304%2C2100489.537276644%2C4770282.061221281'). "
            "Double-encoded commas are supported."
        ),
    ),
    start_base: Optional[str] = Query(
        None,
        description=(
            "Filter runs from this base_time (inclusive). Base time ISO8601 "
            "(e.g., '2025-07-03T00:00:00' or '2025-07-03T00:00:00Z')."
        ),
    ),
    end_base: Optional[str] = Query(
        None,
        description=(
            "Filter runs up to this base_time (inclusive). Base time ISO8601 "
            "(e.g., '2025-07-07T00:00:00' or '2025-07-07T00:00:00Z')."
        ),
    ),
):
    """
    Returns run-to-run statistics with x-axis = base_time.

    Response:
    {
      "index": "pof" | "fopi",
      "mode": "by_base_time",
      "stat": ["mean", "median"],
      "timestamps": ["2025-07-01T00:00:00Z", ...],   # base_time runs
      "mean": [<float|null>, ...],                   # one per base_time
      "median": [<float|null>, ...],
      "bbox_epsg3857": "<original bbox or null>"
    }
    """
    try:
        ds = _load_zarr(index)
        var_name = settings.VAR_NAMES[index]
        da = ds[var_name]

        # Normalize and optionally filter the list of runs
        base_vals = pd.to_datetime(ds["base_time"].values)
        base_vals = [pd.Timestamp(bv).tz_localize(None).replace(microsecond=0) for bv in base_vals]

        if start_base:
            sb = _parse_iso_naive(start_base)
            base_vals = [bt for bt in base_vals if bt >= sb]
        if end_base:
            eb = _parse_iso_naive(end_base)
            base_vals = [bt for bt in base_vals if bt <= eb]
        if not base_vals:
            raise HTTPException(status_code=404, detail="No base_time runs found for the given filters.")

        # Select only those runs (works whether base_time is a dim or coord)
        da_sel = da.sel(base_time=base_vals)

        # --- spatial subset using EPSG:3857 bbox (antimeridian-aware) ---
        # This keeps all non-spatial dims (e.g., base_time) intact.
        try:
            da_sel = _extract_spatial_subset(da_sel, bbox=bbox)
        except Exception as e:
            logger.exception("Failed to apply bbox filter")
            raise HTTPException(status_code=400, detail=f"Invalid bbox or spatial coords: {e}")

        # If bbox excludes all pixels, bail out early
        if ("lat" in da_sel.dims and da_sel.sizes.get("lat", 0) == 0) or (
            "lon" in da_sel.dims and da_sel.sizes.get("lon", 0) == 0
        ):
            raise HTTPException(status_code=404, detail="No data within the provided bbox.")

        # ---- Compute run-to-run stats without Dask; use _agg_mean_median ----
        mean_vals: list[float | None] = []
        median_vals: list[float | None] = []

        # Preserve the order of the selected runs as in da_sel
        bt_coord = pd.to_datetime(da_sel["base_time"].values)
        for bt in bt_coord:
            da_bt = da_sel.sel(base_time=bt)
            m, md = _agg_mean_median(da_bt)
            mean_vals.append(m)
            median_vals.append(md)

        # Base-time timestamps → ISO strings
        timestamps_iso = [_iso_utc(pd.Timestamp(t)) for t in bt_coord]

        return {
            "index": index.lower(),
            "mode": "by_base_time",
            "stat": ["mean", "median"],
            "bbox_epsg3857": bbox or None,
            "timestamps": timestamps_iso,   # x-axis is base_time runs
            "mean": mean_vals,
            "median": median_vals,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to build run-to-run series")
        return JSONResponse(status_code=400, content={"error": str(e)})
