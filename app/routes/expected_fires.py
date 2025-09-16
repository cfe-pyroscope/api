from fastapi import APIRouter, Query, Path, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, Literal, Dict, List, Tuple
from urllib.parse import unquote
import pandas as pd

from utils.zarr_handler import _load_zarr
from utils.time_utils import _iso_drop_tz, _iso_utc_str
from utils.bounds_utils import _extract_spatial_subset, _bbox_to_latlon
from config.config import settings
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/expected_fires")
async def expected_fires(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    bbox: Optional[str] = Query(
        None,
        description=(
            "EPSG:3857 bbox as 'x_min,y_min,x_max,y_max' (URL-encoded commas).\n"
            "Example: '1033428.6224155831%2C4259682.712276304%2C2100489.537276644%2C4770282.061221281'"
        ),
    ),
    start_base: Optional[str] = Query(
        None,
        description="Filter runs from this base_time (inclusive). ISO8601 (e.g., '2025-09-01T00:00:00Z').",
    ),
    end_base: Optional[str] = Query(
        None,
        description="Filter runs up to this base_time (inclusive). ISO8601 (e.g., '2025-09-04T00:00:00Z').",
    )
):
    """
    Compute the **expected number of fires** for a region/time window by **summing per-cell probabilities**.

    Assumptions & interpretation
    ----------------------------
    - The dataset variable for `index` is a *probability* in [0, 1] that at least one fire occurs
      in the grid cell on that run/day. The **sum over cells** inside the region is the expected
      number of fire-positive cells (no-correlation assumption affects variance, **not** the mean).
    - If a `bbox` is given (in EPSG:3857), we subset spatially before aggregation. Partial-cell
      overlap is approximated by the grid subset (no sub-cell weighting).
    """
    try:
        # ---- Load dataset & variable ----
        ds = _load_zarr(index)
        var_name = settings.VAR_NAMES[index]
        da = ds[var_name]

        # ---- Select base_time window (inclusive) ----
        base_vals = pd.to_datetime(ds["base_time"].values)
        base_vals = [pd.Timestamp(bv).tz_localize(None).replace(microsecond=0) for bv in base_vals]
        base_vals_sorted = sorted(base_vals)

        if start_base:
            sb = _iso_drop_tz(start_base)
            base_vals_sorted = [bt for bt in base_vals_sorted if bt >= sb]
        if end_base:
            eb = _iso_drop_tz(end_base)
            base_vals_sorted = [bt for bt in base_vals_sorted if bt <= eb]

        if not base_vals_sorted:
            return JSONResponse(
                status_code=200,
                content={
                    "index": index.lower(),
                    "stat": ["sum"],
                    "bbox_epsg3857": unquote(bbox) if bbox else None,
                    "bbox_epsg4326": _bbox_to_latlon(bbox) if bbox else None,
                    "dates": [],
                    "expected_sum": [],
                    "cumulative_expected": [],
                },
            )

        # ---- Narrow to selected runs ----
        da_sel = da.sel(base_time=base_vals_sorted)

        # ---- spatial subset (EPSG:3857 -> 4326 inside utility) ----
        da_sel = _extract_spatial_subset(da_sel, bbox=bbox)

        # ---- Compute per-run sums over spatial dims ----
        # ---- Compute per-run sums over *all* non-time dims ----
        # This guarantees a single number per base_time (run).
        non_time_dims = [d for d in da_sel.dims if d != "base_time"]

        # Reduce across all non-time dims in one go
        summed_per_run = da_sel.sum(dim=non_time_dims, skipna=True)

        # Bring to NumPy if dask-backed
        try:
            summed_per_run = summed_per_run.compute()
        except Exception:
            pass  # already NumPy-backed

        # Extract arrays
        bt_coord = pd.to_datetime(summed_per_run["base_time"].values)
        values_per_run = summed_per_run.values  # 1D array, same length as bt_coord

        # Safety: squeeze in case of stray size-1 dims (shouldn't happen, but harmless)
        import numpy as np
        values_per_run = np.asarray(values_per_run).squeeze()
        if values_per_run.ndim != 1:
            raise ValueError(f"Expected a 1D array per base_time; got shape {values_per_run.shape}")

        # Now build the (date, value) table for grouping by UTC day
        df = pd.DataFrame({
            "date": [pd.Timestamp(ts).tz_localize("UTC").date().isoformat() for ts in bt_coord],
            "val": [float(v) for v in values_per_run],
        })
        grouped = df.groupby("date", sort=True)["val"].sum()
        dates = list(grouped.index)
        values = [float(v) for v in grouped.values]
        cumulative = list(pd.Series(values).cumsum())

        if bbox:
            lon_min, lat_min, lon_max, lat_max = _bbox_to_latlon(bbox)
            bbox_latlon_flat = (lat_min, lon_min, lat_max, lon_max)
        else:
            bbox_latlon_flat = None

        response = {
            "index": index.lower(),
            "mode": "by_date",
            "stat": ["sum"],
            "bbox_epsg3857": unquote(bbox) if bbox else None,
            "bbox_epsg4326": bbox_latlon_flat,
            "dates": dates,  # YYYY-MM-DD
            "expected_sum": values,
            "cumulative_expected": [float(x) for x in cumulative],
            "notes": (
                "Values represent expected number of fire-positive grid cells per UTC day "
                "(sum of probabilities over the spatial subset)."
            ),
        }

        logger.info(f"Expected fires response: {response}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to compute expected fires series")
        return JSONResponse(status_code=400, content={"error": str(e)})
