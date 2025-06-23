import io
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
from app.services.zarr_loader import load_zarr
from app.services.bounds_utils import extract_spatial_subset, reproject_and_prepare
from app.services.time_utils import calculate_time_index

matplotlib.use("Agg")
logger = logging.getLogger("uvicorn")


def generate_heatmap_image(index: str, base_time: str, lead_hours: int, bbox: str = None):
    """
    Generate a heatmap image for a specific forecast time and spatial region.

    Parameters:
        index (str): Dataset identifier (e.g., 'fopi', 'pof').
        base_time (str): ISO 8601 base time string.
        lead_hours (int): Forecast lead time in hours from base time.
        bbox (str, optional): Optional bounding box in EPSG:3857, formatted as 'x_min,y_min,x_max,y_max'.

    Returns:
        tuple[io.BytesIO, list[float]]: A tuple containing:
            - The rendered PNG image as an in-memory BytesIO buffer.
            - The extent of the image in EPSG:3857 as [left, right, bottom, top].
    """
    ds = load_zarr(index)
    param = list(ds.data_vars.keys())[0]
    logger.info(f"📥 Loaded index: {index}, param: {param}")

    time_index = calculate_time_index(ds, index, base_time, lead_hours)
    subset = extract_spatial_subset(ds, param, time_index, bbox)
    data, extent = reproject_and_prepare(subset)
    return render_heatmap(data, extent)


def render_heatmap(data, extent):
    """
    Render a heatmap image from raster data using a predefined color map.

    Parameters:
        data (np.ndarray): 2D array of raster values in EPSG:3857.
        extent (list[float]): Bounding box in EPSG:3857 [left, right, bottom, top].

    Returns:
        tuple[io.BytesIO, list[float]]: A PNG image stream and the corresponding extent.
    """
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    aspect_ratio = x_range / y_range if y_range else 1
    height = 5
    width = height * aspect_ratio

    fig, ax = plt.subplots(figsize=(width, height), dpi=150)
    ax.axis("off")
    ax.set_aspect("auto")

    masked_data = np.ma.masked_invalid(data)
    colors = [(0, 0, 0, 0), "#fff7ec", "#fee8c8", "#fdd49e", "#fdbb84",
              "#fc8d59", "#ef6548", "#d7301f", "#b30000", "#7f0000"]
    cmap = ListedColormap(colors)
    cmap.set_bad(color=(0, 0, 0, 0))

    logger.info(f"🖼️ Final extent used in imshow (EPSG:3857): {extent}")
    ax.imshow(masked_data, extent=extent, origin="upper", cmap=cmap, vmin=0, vmax=1)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf, extent
