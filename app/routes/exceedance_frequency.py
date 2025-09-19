from fastapi import APIRouter, Query, Path, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional, List
from urllib.parse import unquote
import numpy as np
import pandas as pd

from utils.zarr_handler import _load_zarr
from utils.time_utils import _iso_drop_tz
from utils.bounds_utils import _extract_spatial_subset, _bbox_to_latlon
from config.config import settings
from config.logging_config import logger

router = APIRouter()


@router.get("/{index}/exceedance_frequency")
async def exceedance_frequency(
    index: str = Path(..., description="Dataset identifier, e.g. 'fopi' or 'pof'."),
    bbox: str = Query(
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
    ),
    thresholds: Optional[str] = Query(
        None,
        description=(
            "Optional comma-separated list of thresholds in [0,1] at which to compute exceedance "
            "fractions (e.g., '0.2,0.4,0.6,0.8,1.0'). If omitted, uses the official THRESHOLDS_ based on index "
            "from 0.00 to 1.00."
        ),
    ),
):
    """
    Exceedance frequency / CCDF for a probability index.

    What it computes
    ----------------
    For each selected `base_time` (run/day) and for each threshold τ in [0,1], we compute:

        fraction(τ) = (# of grid cells with value >= τ) / (# of valid cells)

    Returned both:
      - **overall**: pooling all selected runs together (useful for a smooth exceedance curve),
      - **by_date**: per UTC day (so it can be plotted as small multiples or compared across days).

    Notes
    -----
    - Works entirely on the given index (no external datasets).
    - If `bbox` provided, subsets spatially before computing frequencies.
    - Missing values (NaNs) are ignored in the denominator.
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

        # Early return if no runs
        if not base_vals_sorted:
            return JSONResponse(
                status_code=200,
                content={
                    "index": index.lower(),
                    "mode": "exceedance_frequency",
                    "bbox_epsg3857": unquote(bbox) if bbox else None,
                    "bbox_epsg4326": _bbox_to_latlon(bbox) if bbox else None,
                    "thresholds": [],
                    "overall": {"thresholds": [], "fraction": [], "count": [], "total": 0},
                    "by_date": {"dates": [], "fraction": [], "count": [], "total": []},
                    "notes": "No base_time runs match the requested window.",
                },
            )

        # ---- Narrow to selected runs ----
        da_sel = da.sel(base_time=base_vals_sorted)

        # ---- Spatial subset (EPSG:3857 -> 4326 inside utility) ----
        da_sel = _extract_spatial_subset(da_sel, bbox=bbox)

        # ---- Prepare thresholds ----
        if thresholds:
            try:
                thr = sorted(
                    {
                        float(x)
                        for x in thresholds.split(",")
                        if x.strip() != ""
                    }
                )
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid 'thresholds' list. Use comma-separated floats.")
            if any((t < 0 or t > 1) for t in thr):
                raise HTTPException(status_code=400, detail="All thresholds must be within [0, 1].")
        else:
            thr = [float(x) for x in np.linspace(0.0, 1.0, 101)]

        # ---- Stack spatial dims into a single dimension for vectorized math ----
        non_time_dims = [d for d in da_sel.dims if d != "base_time"]
        if not non_time_dims:
            raise HTTPException(status_code=400, detail="Dataset is missing spatial dimensions.")

        # stack -> shape (base_time, space)
        try:
            da_stacked = da_sel.stack(space=non_time_dims).transpose("base_time", "space")
            # bring into memory if dask-backed
            try:
                arr = da_stacked.values  # may trigger dask compute
            except Exception:
                arr = np.asarray(da_stacked)
        except Exception as e:
            logger.exception("Failed to stack spatial dimensions.")
            raise HTTPException(status_code=400, detail=f"Failed to prepare data: {e}")

        # arr: (n_time, n_space)
        if arr.ndim != 2:
            raise HTTPException(status_code=400, detail=f"Unexpected data shape {arr.shape}; expected 2D after stacking.")

        n_time, n_space = arr.shape
        bt_coord = pd.to_datetime(da_stacked["base_time"].values)

        # Mask NaNs
        valid_mask = ~np.isnan(arr)

        # ---- Overall CCDF (pool across all runs) ----
        # Flatten time+space
        flat_vals = arr[valid_mask]
        overall_total = int(flat_vals.size)
        overall_frac, overall_cnt = [], []
        for t in thr:
            cnt = int(np.count_nonzero(flat_vals >= t))
            overall_cnt.append(cnt)
            overall_frac.append(float(cnt / overall_total) if overall_total > 0 else float("nan"))

        # ---- Per-run exceedances -> group by UTC day ----
        # For each run, compute counts per threshold over valid cells
        per_run_total = valid_mask.sum(axis=1).astype(int)  # length n_time
        # Vectorized comparison for each threshold: loop over thresholds to save memory
        per_run_cnt = np.empty((n_time, len(thr)), dtype=np.int64)
        for j, t in enumerate(thr):
            per_run_cnt[:, j] = np.count_nonzero((arr >= t) & valid_mask, axis=1)

        # Build (date, counts, totals) then aggregate runs that share the same UTC date
        dates_utc = [pd.Timestamp(ts).tz_localize("UTC").date().isoformat() for ts in bt_coord]
        df_cnt = pd.DataFrame(per_run_cnt, index=dates_utc)  # columns correspond to thr indices
        df_tot = pd.Series(per_run_total, index=dates_utc, name="total")

        # Sum across runs per UTC day
        by_date_cnt = df_cnt.groupby(level=0).sum()
        by_date_tot = df_tot.groupby(level=0).sum()

        dates_sorted = list(by_date_cnt.index)
        # Fractions per date: 2D list [date_index][threshold_index]
        by_date_fraction = []
        by_date_count = []
        by_date_total = []
        for d in dates_sorted:
            tot = int(by_date_tot.loc[d])
            cnt_vec = by_date_cnt.loc[d].astype(int).tolist()
            frac_vec = [float(c / tot) if tot > 0 else float("nan") for c in cnt_vec]
            by_date_total.append(tot)
            by_date_count.append(cnt_vec)
            by_date_fraction.append(frac_vec)

        # BBox in EPSG:4326 for convenience
        if bbox:
            lon_min, lat_min, lon_max, lat_max = _bbox_to_latlon(bbox)
            bbox_latlon_flat = (lat_min, lon_min, lat_max, lon_max)
        else:
            bbox_latlon_flat = None

        response = {
            "index": index.lower(),
            "mode": "exceedance_frequency",
            "bbox_epsg3857": unquote(bbox) if bbox else None,
            "bbox_epsg4326": bbox_latlon_flat,
            "thresholds": [float(t) for t in thr],
            "overall": {
                "thresholds": [float(t) for t in thr],
                "fraction": [float(x) for x in overall_frac],
                "count": [int(c) for c in overall_cnt],
                "total": overall_total,
            },
            "by_date": {
                "dates": dates_sorted,                           # YYYY-MM-DD
                "fraction": by_date_fraction,                    # shape: (n_dates, n_thresholds)
                "count": by_date_count,                          # shape: (n_dates, n_thresholds)
                "total": by_date_total,                          # length: n_dates
            },
            "notes": (
                "Fractions represent the share of valid grid cells with index >= threshold. "
                "'overall' pools all selected runs; 'by_date' aggregates runs by UTC date. "
                "Provide 'thresholds' query (comma-separated) to use specific levels; "
                "otherwise 0.00..1.00 in 0.01 steps are used."
            ),
        }

        logger.info(f"Exceedance frequency response: index={index} thresholds={len(thr)} "
                    f"runs={n_time} space={n_space}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to compute exceedance frequency")
        return JSONResponse(status_code=400, content={"error": str(e)})
