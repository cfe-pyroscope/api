import io
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib
import rasterio
from pyproj import Transformer
import logging

from app.services.zarr_loader import load_zarr
from app.services.bounds_utils import normalize_bbox

matplotlib.use("Agg")
logger = logging.getLogger("uvicorn")


def generate_heatmap_image(index: str, base_time: str, lead_hours: int, bbox: str = None):
    ds = load_zarr(index)
    param = list(ds.data_vars.keys())[0]
    logger.info(f"📥 Loaded index: {index}, param: {param}")

    # Parse base file time from encoding
    encoding_source = str(ds.encoding.get("source", ""))
    match = re.search(rf"{index}_(\d{{10}})", encoding_source)
    base_time_str = match.group(1) if match else "2024120100"
    file_base_time = datetime.strptime(base_time_str, "%Y%m%d%H")
    logger.info(f"📆 Base file time: {file_base_time.isoformat()}")

    base_dt = datetime.fromisoformat(base_time.replace("Z", ""))
    valid_hours = (base_dt - file_base_time).total_seconds() / 3600 + lead_hours
    logger.info(f"🕐 Requested valid hours from base: {valid_hours}")

    time_in_hours = np.array([
        (pd.to_datetime(t).to_pydatetime() - file_base_time).total_seconds() / 3600
        for t in ds.time.values
    ])
    time_index = int(np.argmin(np.abs(time_in_hours - valid_hours)))
    logger.info(f"🧭 Closest time index: {time_index}, Available times: {time_in_hours.tolist()}")

    if bbox:
        x_min, y_min, x_max, y_max = map(float, bbox.split(','))
        logger.info(f"📥 bbox 3857 received: {bbox}")

        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon_min, lat_min = transformer.transform(x_min, y_min)
        lon_max, lat_max = transformer.transform(x_max, y_max)

        lat_min, lon_min, lat_max, lon_max = normalize_bbox(lat_min, lon_min, lat_max, lon_max)
    else:
        lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())
        lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
        lat_min, lon_min, lat_max, lon_max = normalize_bbox(lat_min, lon_min, lat_max, lon_max)
        logger.info("🛑 No bbox provided – using full domain")

    lat_mask = (ds.lat >= lat_min) & (ds.lat <= lat_max)
    lon_mask = (ds.lon >= lon_min) & (ds.lon <= lon_max)

    subset = ds[param].isel(time=time_index).where(lat_mask & lon_mask, drop=True)
    logger.info(f"🗺️ Subset lat range: {float(subset.lat.min())} → {float(subset.lat.max())}")
    logger.info(f"🗺️ Subset lon range: {float(subset.lon.min())} → {float(subset.lon.max())}")

    subset_rio = subset.rio.write_crs(4326).rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    subset_3857 = subset_rio.rio.reproject("EPSG:3857", resampling=rasterio.enums.Resampling.bilinear)

    data = np.nan_to_num(subset_3857.values, nan=0.0)
    x = subset_3857.x.values
    y = subset_3857.y.values

    if y[0] < y[-1]:
        data = np.flipud(data)

    x_res = abs(x[1] - x[0])
    y_res = abs(y[1] - y[0])
    extent = [
        x.min() - x_res / 2,
        x.max() + x_res / 2,
        y.min() - y_res / 2,
        y.max() + y_res / 2,
    ]

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
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