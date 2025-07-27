import rasterio
import numpy as np
from pyproj import Transformer
from app.logging_config import logger


def wrap_longitude(lon: float) -> float:
    """
    Wrap a longitude value from the range [-180, 180] to [0, 360].

    Parameters:
        lon (float): Longitude in degrees, expected in the range [-180, 180].

    Returns:
        float: Longitude wrapped to the range [0, 360].
    """
    return lon if lon >= 0 else lon + 360


def normalize_bbox(lat_min, lon_min, lat_max, lon_max, wrap=False):
    """
    Normalize a bounding box by ordering its latitude and longitude bounds and optionally wrapping longitudes.

    Parameters:
        lat_min (float): Minimum latitude in degrees.
        lon_min (float): Minimum longitude in degrees.
        lat_max (float): Maximum latitude in degrees.
        lon_max (float): Maximum longitude in degrees.
        wrap (bool): If True, wrap longitude values from [-180, 180] to [0, 360] before normalization.

    Returns:
        tuple[float, float, float, float]: A tuple containing (lat_min, lon_min, lat_max, lon_max) with
        values reordered so that lat_min <= lat_max and lon_min <= lon_max, and longitudes wrapped if requested.
    """
    if wrap:
        lon_min = wrap_longitude(lon_min)
        lon_max = wrap_longitude(lon_max)

    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min

    return lat_min, lon_min, lat_max, lon_max


def extract_spatial_subset(ds, param: str, time_index: int, bbox: str):
    """
    Extract a spatial subset from a dataset based on a bounding box or the full domain.

    Parameters:
        ds (xr.Dataset): Input dataset containing lat/lon dimensions.
        param (str): Name of the variable to extract.
        time_index (int): Index of the time dimension to slice.
        bbox (str): Optional bounding box in EPSG:3857, formatted as "x_min,y_min,x_max,y_max".

    Returns:
        xr.DataArray: A subset of the input variable at the given time index and spatial bounds.
    """
    if bbox:
        x_min, y_min, x_max, y_max = map(float, bbox.split(','))
        logger.info(f"📥 bbox 3857 received: {bbox}")
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon_min, lat_min = transformer.transform(x_min, y_min)
        lon_max, lat_max = transformer.transform(x_max, y_max)
        lat_min, lon_min, lat_max, lon_max = normalize_bbox(lat_min, lon_min, lat_max, lon_max)
        logger.info(f"🗺️ Reprojected bbox to lat/lon: ({lat_min}, {lon_min}) → ({lat_max}, {lon_max})")
    else:
        lat_min, lat_max = float(ds.lat.min()), float(ds.lat.max())
        lon_min, lon_max = float(ds.lon.min()), float(ds.lon.max())
        lat_min, lon_min, lat_max, lon_max = normalize_bbox(lat_min, lon_min, lat_max, lon_max)
        logger.info(f"🗺️ Reprojected bbox to lat/lon: ({lat_min}, {lon_min}) → ({lat_max}, {lon_max})")
        logger.info("🛑 No bbox provided – using full domain")

    lat_mask = (ds.lat >= lat_min) & (ds.lat <= lat_max)
    lon_mask = (ds.lon >= lon_min) & (ds.lon <= lon_max)
    logger.info(f"🌍 Dataset lat range: {ds.lat.min().item()} → {ds.lat.max().item()}")
    logger.info(f"🌍 Dataset lon range: {ds.lon.min().item()} → {ds.lon.max().item()}")

    subset = ds[param].isel(time=time_index).where(lat_mask & lon_mask, drop=True)
    logger.info(f"🗺️ Subset lat range: {float(subset.lat.min())} → {float(subset.lat.max())}")
    logger.info(f"🗺️ Subset lon range: {float(subset.lon.min())} → {float(subset.lon.max())}")
    return subset


def reproject_and_prepare(subset):
    """
    Reproject a spatial subset from EPSG:4326 to EPSG:3857 and compute its extent.

    Parameters:
        subset (xr.DataArray): Input data array with lat/lon coordinates in EPSG:4326.

    Returns:
        tuple[np.ndarray, list[float]]: Tuple of:
            - 2D numpy array with NaNs filled and optionally flipped vertically.
            - List representing the spatial extent in EPSG:3857 [left, right, bottom, top].
    """
    rasterio.show_versions()
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
    return data, extent
