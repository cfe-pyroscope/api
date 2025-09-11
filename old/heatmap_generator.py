import io
import logging
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib
from utils.zarr_handler import _load_zarr, _slice_field, _select_first_param
from utils.bounds_utils import _extract_spatial_subset, _reproject_and_prepare
from utils.time_utils import _normalize_times, _match_base_time, _match_forecast_time
from config.config import COLORS, RANGE

matplotlib.use("Agg")
logger = logging.getLogger("uvicorn")


def _log_subset_stats(subset: xr.DataArray) -> None:
    """
    Compute and log basic summary statistics for an xarray DataArray subset. Just for internal purpose

    Notes
    -----
    - Uses xarray reductions (`.min()`, `.max()`, `.mean()`, `.count()`) and calls
      `.compute()` to realize values, which will trigger Dask execution if the
      data is lazy.
    - `valid_count` counts non-NaN elements only.
    - If `subset` is empty or all-NaN, the reductions may return `nan` and
      `valid_count` will be 0.
    """
    subset_min = float(subset.min().compute())
    subset_max = float(subset.max().compute())
    subset_mean = float(subset.mean().compute())
    valid_count = int(subset.count().compute().values)
    logger.info(f"📊 Subset stats — min: {subset_min}, max: {subset_max}, mean: {subset_mean}")
    logger.info(f"🚩 D - subset shape: {subset.shape}, valid count: {valid_count}")


def _render_from_subset(index: str, subset: xr.DataArray) -> tuple[io.BytesIO, list[float], float, float]:
    """
    Reproject a DataArray subset and render it as a heatmap image.

    This helper first calls `_reproject_and_prepare(subset)` to produce a 2D array
    and its spatial extent, then delegates to `render_heatmap(...)` to create the
    final image.
    """
    data, extent = _reproject_and_prepare(subset)
    logger.info(f"🚩 E - data.shape: {data.shape}, extent: {extent}")
    return render_heatmap(index, data, extent)


def generate_heatmap_image(index: str, base_time: str, forecast_time: str, bbox: str = None):
    """
    Generate a heatmap image for a single forecast slice, optionally clipped to a
    bounding box.

    This orchestrates the full pipeline:
    load → time-match → slice → spatial subset → stats → reproject → render.

    Workflow Details
    ----------------
    1) `_load_zarr(index)` loads the dataset.
    2) `_select_first_param(ds)` chooses the variable to render.
    3) `_normalize_times(...)` and `_match_*_time(...)` align requested times to
       dataset coordinates.
    4) `_slice_field(...)` extracts the 2D (lat, lon) field for those times.
    5) `_extract_spatial_subset(..., bbox)` crops the field and loads it into memory.
    6) `_reproject_and_prepare(...)` reprojects data and returns the extent.
    7) `render_heatmap(...)` produces the final image and value range.
    """
    try:
        logger.info("🚩 A - loading zarr")
        ds = _load_zarr(index)

        param = _select_first_param(ds)
        logger.info(f"🚩 B - param selected: {param}")

        req_base, req_fcst = _normalize_times(base_time, forecast_time)
        matched_base = _match_base_time(ds, req_base)
        matched_fcst = _match_forecast_time(ds, matched_base, req_fcst)
        logger.info(f"🚩 C - indices: base={matched_base.isoformat()}, forecast={matched_fcst.isoformat()}")

        da_at_time = _slice_field(ds, param, matched_base, matched_fcst)

        subset = _extract_spatial_subset(da_at_time, bbox=bbox).load()
        _log_subset_stats(subset)

        image_stream, extent, vmin, vmax = _render_from_subset(index, subset)
        return image_stream, extent, vmin, vmax

    except Exception:
        logger.exception("🛑 generate_heatmap_image failed")
        raise


def render_heatmap(index: str, data: np.ndarray, extent: list[float]) -> tuple[io.BytesIO, list[float], float, float]:
    """
    Render a transparent-background heatmap PNG from a 2D raster array.

    Scaling rules:
      - POF  : fixed [0.0, 0.05], values >= 0.05 are max category.
      - FOPI : fixed [0.0, 1.0], linear from no risk to max risk.
      - other: robust [p2, p98] from valid (non-masked, finite) data.

    Notes
    -----
    - 'extent' must be [xmin, xmax, ymin, ymax] for imshow.
    - Zeros, NaN, and Inf are masked (transparent).
    """
    # Figure sizing based on extent
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    aspect_ratio = (x_range / y_range) if y_range else 1.0
    height = 5
    width = max(height * aspect_ratio, 1e-3)

    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    ax.axis("off")
    ax.set_aspect("auto")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Mask invalid + zeros (e.g., ocean/nodata)
    masked_data = np.ma.masked_invalid(data)
    masked_data = np.ma.masked_where((masked_data == 0), masked_data)

    # Build colormap (masked values are fully transparent)

    cmap = ListedColormap(COLORS)
    cmap.set_bad(color=(0, 0, 0, 0))  # transparent for masked

    # Determine vmin/vmax according to the index
    idx = (index or "").lower()
    if idx == "pof":
        vmin, vmax = RANGE["pof"]
        logger.info("📏 POF fixed scaling: [vmin=0.00, vmax=0.05] — values ≥ 0.05 will be max color.")
    elif idx == "fopi":
        vmin, vmax = RANGE["fopi"]
        logger.info("📏 FOPI fixed scaling: [vmin=0.0, vmax=1.0].")
    else:
        valid_data = masked_data.compressed()  # 1D ndarray of finite, unmasked values
        if valid_data.size:
            p2 = float(np.percentile(valid_data, 2))
            p98 = float(np.percentile(valid_data, 98))
            if not np.isfinite(p2) or not np.isfinite(p98) or p2 == p98:
                vmin = float(np.min(valid_data))
                vmax = float(np.max(valid_data))
            else:
                vmin, vmax = p2, p98
            if vmin == vmax:
                vmax = vmin + 1e-6
            logger.info(f"📊 Default scaling - Display range: [{vmin:.6g}, {vmax:.6g}] (robust 2–98th).")
        else:
            vmin, vmax = RANGE["default"]
            logger.warning("⚠️ No valid data found for default scaling; using fallback [0, 1].")

    logger.info(f"🖼️ Final extent used in imshow (EPSG:3857): {extent}")
    im = ax.imshow(masked_data, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)

    if idx == "pof":
        pos = (0.05 - vmin) / (vmax - vmin) if vmax > vmin else 1.0
        logger.info(f"POF severe threshold 0.05 maps to the top of the scale (position {pos:.2f}).")

    buf = io.BytesIO()
    plt.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        pad_inches=0,
        transparent=True,
        facecolor="none",
    )
    plt.close(fig)
    buf.seek(0)
    return buf, extent, float(vmin), float(vmax)