"""
This is the original version in which the color scale change dynamically according to the min and max value of the selected area
"""

import io
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
from app.utils.zarr_handler import _load_zarr, _slice_field, _select_first_param
from app.utils.bounds_utils import _extract_spatial_subset, _reproject_and_prepare
from app.utils.time_utils import _normalize_times, _match_base_time, _match_forecast_time

matplotlib.use("Agg")
logger = logging.getLogger("uvicorn")


def _log_subset_stats(subset: xr.DataArray) -> None:
    """
    Compute and log basic summary statistics for an xarray DataArray subset. Just for internal purpose

    Parameters
    ----------
    subset : xr.DataArray
        The array to summarize. May be backed by Dask; NaNs are allowed.

    Returns
    -------
    None

    Side Effects
    ------------
    Logs two INFO-level messages using the module-level `logger`:
    1) "📊 Subset stats — min: <min>, max: <max>, mean: <mean>"
    2) "🚩 D - subset shape: <shape>, valid count: <count>"

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

    Parameters
    ----------
    index : str
        Identifier/label for the rendered layer (passed through to `render_heatmap`).
    subset : xr.DataArray
        Input data to visualize. Expected to represent a single 2D field that can
        be reprojected to a regular grid.

    Returns
    -------
    tuple[io.BytesIO, list[float], float, float]
        - In-memory image buffer (e.g., PNG) of the heatmap.
        - Spatial extent as `[xmin, ymin, xmax, ymax]`.
        - Minimum data value used for rendering (vmin).
        - Maximum data value used for rendering (vmax).

    Side Effects
    ------------
    Logs an INFO-level message with the prepared array shape and extent.

    Notes
    -----
    - The exact output format and color mapping are determined by `render_heatmap`.
    - Any errors arising from reprojection or rendering propagate to the caller.
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

    Parameters
    ----------
    index : str
        Dataset identifier used by `_load_zarr` (and passed through to rendering).
    base_time : str
        Requested model initialization time. Expected to be an ISO 8601—like string
        (e.g., "2025-08-11T00:00Z"); it is normalized by `_normalize_times`.
    forecast_time : str
        Requested forecast valid time, same formatting expectations as `base_time`;
        normalized and matched to dataset coordinates.
    bbox : str, optional
        Spatial bounding box used by `_extract_spatial_subset`. Expected format
        "minx,miny,maxx,maxy" in the dataset's CRS (commonly lon/lat in WGS84).
        If omitted, the full 2D field is used.

    Returns
    -------
    tuple[io.BytesIO, list[float], float, float]
        - In-memory image buffer (e.g., PNG) containing the rendered heatmap.
        - Spatial extent as [xmin, ymin, xmax, ymax].
        - vmin: minimum data value used for the color scale.
        - vmax: maximum data value used for the color scale.

    Raises
    ------
    Exception
        Any errors from I/O, time matching, selection, reprojection, or rendering
        are logged with stack traces and re-raised. Examples include:
        - File/connection errors while loading Zarr
        - Value/Key errors during time/variable selection
        - Projection/resampling failures

    Side Effects
    ------------
    - Logs progress and debug information at INFO level (steps A—E) via the
      module-level `logger`.
    - Computes and logs subset statistics via `_log_subset_stats`.
    - Forces data materialization for the spatial subset (`.load()`), which may
      trigger Dask execution.

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


def render_heatmap(index, data, extent):
    """
    Render a transparent-background heatmap PNG from a 2D raster array.

    The image uses a predefined sequential colormap with transparency for masked
    pixels (NaN/Inf/zero), so ocean or nodata areas are invisible.

    Parameters
    ----------
    index : str
        Dataset identifier that selects the color scaling strategy:
        - "pof": percentile-based scaling (5th—95th) with floors/ceilings.
        - "fopi": fixed range [0, 1].
        - other: robust scaling using 2nd—98th percentiles.
    data : np.ndarray
        2D array of raster values in EPSG:3857 (Web Mercator).
    extent : list[float]
        Spatial bounds in EPSG:3857 as [xmin, xmax, ymin, ymax]
        (aka [left, right, bottom, top]). Passed to `imshow`.

    Returns
    -------
    tuple[io.BytesIO, list[float], float, float]
        - In-memory PNG image stream (transparent background).
        - The extent used for rendering, as [xmin, xmax, ymin, ymax].
        - vmin: minimum value mapped to the colormap.
        - vmax: maximum value mapped to the colormap.

    Notes
    -----
    - Pixels where `data == 0`, `NaN`, or `Inf` are masked and rendered fully
      transparent. The colormap's "bad" color is also transparent.
    - Figure size is derived from `extent` to preserve aspect ratio; axes and
      background are hidden and fully transparent.
    - Scaling:
      - **POF**: vmin = max(p5, 0.001), vmax = max(p95, 0.05).
      - **FOPI**: vmin = 0, vmax = 1.
      - **Default**: vmin = 2nd percentile, vmax = 98th percentile.
      If no finite values are present, falls back to vmin=0, vmax=1.
    - Uses `origin="upper"` in `imshow`.

    Side Effects
    ------------
    Logs INFO/WARNING messages (extent used, chosen display range, and, for "pof",
    the relative position of the 0.05 threshold) via the module-level `logger`.

    Raises
    ------
    Any exceptions from NumPy/Matplotlib (e.g., invalid shapes or extents) are not
    caught and will propagate to the caller.
    """
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    aspect_ratio = x_range / y_range if y_range else 1
    height = 5
    width = height * aspect_ratio
    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    ax.axis("off")
    ax.set_aspect("auto")
    # Make figure background transparent
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)

    # Mask both invalid values AND zeros (ocean areas)
    masked_data = np.ma.masked_where((data == 0) | np.isnan(data) | np.isinf(data), data)

    # Color map - first color is transparent for masked values
    colors = [(0, 0, 0, 0), "#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84",
              "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#7f0000"]
    cmap = ListedColormap(colors)
    cmap.set_bad(color=(0, 0, 0, 0))  # Transparent for masked values

    logger.info(f"🖼️ Final extent used in imshow (EPSG:3857): {extent}")

    # Calculate vmin/vmax excluding invalid values
    valid_data = data[np.isfinite(data)]
    if len(valid_data) > 0:
        """ ORIGINAL CALCULATION
        data_min = np.min(valid_data)
        data_max = np.max(valid_data)"""
        data_min = 0
        data_max = 1

        # Different scaling strategies based on index type
        if index.lower() == 'pof':
            # For POF: Use percentile-based scaling to enhance contrast
            # This ensures we use the full color range even for small values
            p5 = np.percentile(valid_data, 5)  # 5th percentile as minimum
            p95 = np.percentile(valid_data, 95)  # 95th percentile as maximum

            # Ensure minimum threshold for severe conditions (0.05)
            vmin = max(p5, 0.001)  # Don't go below 0.001 to avoid over-stretching
            vmax = max(p95, 0.05)  # Ensure we can show severe conditions (≥0.05)

            logger.info(f"📊 POF scaling - Data range: [{data_min:.4f}, {data_max:.4f}], "
                        f"Display range: [{vmin:.4f}, {vmax:.4f}]")

        elif index.lower() == 'fopi':
            # For FOPI: Use the full data range (original behavior)
            vmin = data_min
            vmax = data_max

            logger.info(f"📊 FOPI scaling - Data range: [{data_min:.4f}, {data_max:.4f}]")

        else:
            # Default: Use full range but with some outlier protection
            p2 = np.percentile(valid_data, 2)
            p98 = np.percentile(valid_data, 98)
            vmin = p2
            vmax = p98

            logger.info(f"📊 Default scaling - Data range: [{data_min:.4f}, {data_max:.4f}], "
                        f"Display range: [{vmin:.4f}, {vmax:.4f}]")
    else:
        vmin, vmax = 0, 1  # Fallback if no valid data
        logger.warning("⚠️ No valid data found, using fallback range [0, 1]")

    im = ax.imshow(masked_data, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)

    if index.lower() == 'pof':
        logger.info(f"🔥 POF severe conditions threshold (0.05) maps to color position: "
                    f"{(0.05 - vmin) / (vmax - vmin):.2f}")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0,
                transparent=True, facecolor='none')  # Ensure transparency is preserved
    plt.close(fig)
    buf.seek(0)
    return buf, extent, float(vmin), float(vmax)