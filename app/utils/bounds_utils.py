import rasterio
import numpy as np
from pyproj import Transformer
from urllib.parse import unquote
import xarray as xr


def _decode_coords(b: str) -> str:
    """
    Decode a URL-encoded coords string (2) or bounding box string (4).

    Performs one or two rounds of URL decoding on the input string. A second decoding
    pass is applied if the first decoded result still contains a "%2C" sequence
    (encoded comma), which can occur if the string was encoded twice.

    Args:
        b (str): Coords/bbox string, potentially URL-encoded one or more times.
            Example: "-8237642%2C4970351%2C-8235642%2C4972351"

    Returns:
        str: Decoded coords/bbox string in plain text form.
            Example: "-8237642,4970351,-8235642,4972351"

    Example:
       >>> _decode_coords("-8237642%2C4970351%2C-8235642%2C4972351")
        '-8237642,4970351,-8235642,4972351'
    """
    if not b:
        return b
    s1 = unquote(b)
    return unquote(s1) if "%2C" in s1 else s1


def _parse_coords(coords: str) -> tuple[float, float]:
    """
    Parse 'x,y' in EPSG:3857 into floats.
    """
    if not coords:
        raise ValueError("Missing 'coords' query parameter.")
    decoded = _decode_coords(coords).strip()
    parts = decoded.split(",")
    if len(parts) != 2:
        raise ValueError("Invalid 'coords' format. Expected 'x,y' in EPSG:3857.")
    try:
        x, y = float(parts[0]), float(parts[1])
    except ValueError:
        raise ValueError("Invalid numeric values in 'coords'.")
    return x, y


def _bbox_to_latlon(bbox_str: str):
    """
    Convert 'x_min,y_min,x_max,y_max' from EPSG:3857 → EPSG:4326.
    Returns tuple (lon_min, lat_min, lon_max, lat_max).
    """
    bbox = unquote(bbox_str).strip()
    x_min, y_min, x_max, y_max = map(float, bbox.split(","))

    transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer.transform(x_min, y_min)
    lon_max, lat_max = transformer.transform(x_max, y_max)

    return lon_min, lat_min, lon_max, lat_max


def _extract_spatial_subset(ds_or_da, param: str = None, bbox: str = None):
    """
    Return a lat/lon spatial subset of a DataArray (or Dataset variable).

    - Accepts an xarray.DataArray, or an xarray.Dataset plus `param` to select a variable.
    - If `bbox` is given, it must be an EPSG:3857 bbox string (possibly URL-encoded)
      in "x_min,y_min,x_max,y_max". It is transformed to EPSG:4326 (lon/lat).
    - Bounds are made antimeridian-aware and padded by ~½ a grid step so very small
      boxes still capture at least one pixel. If the result is still empty, it snaps
      to the nearest grid point at the bbox center.
    - Expects `lat` and `lon` coordinates; raises if missing.

    Returns
    -------
    xarray.DataArray
        The subset with `lat`/`lon` reduced to the requested region.
    """
    da = ds_or_da[param] if isinstance(ds_or_da, xr.Dataset) else ds_or_da
    if not all(c in da.coords for c in ("lat", "lon")):
        raise ValueError(f"Expected 'lat' and 'lon' coordinates; got dims={da.dims}")

    # ---- 3857 → 4326
    if bbox:
        lon_min, lat_min, lon_max, lat_max = _bbox_to_latlon(bbox)  # (lon, lat) order
    else:
        lat_min, lat_max = float(da.lat.min()), float(da.lat.max())
        lon_min, lon_max = float(da.lon.min()), float(da.lon.max())

    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min

    # ---- estimate grid step and pad ~½ a cell so tiny boxes still hit pixels
    def _step(coord):
        if coord.size <= 1:
            return 0.0
        try:
            d = np.diff(coord.values)
        except Exception:
            d = coord.diff(coord.dims[0]).values
        return float(np.nanmedian(np.abs(d)))

    lon_step = _step(da.lon)
    lat_step = _step(da.lat)
    pad_lon = 0.51 * lon_step if lon_step > 0 else 0.0
    pad_lat = 0.51 * lat_step if lat_step > 0 else 0.0

    lon_min_e = lon_min - pad_lon
    lon_max_e = lon_max + pad_lon
    lat_min_e = lat_min - pad_lat
    lat_max_e = lat_max + pad_lat

    # ---- antimeridian-aware mask
    crosses = lon_min_e > lon_max_e
    if crosses:
        lon_mask = (da.lon >= lon_min_e) | (da.lon <= lon_max_e)
    else:
        lon_mask = (da.lon >= lon_min_e) & (da.lon <= lon_max_e)
    lat_mask = (da.lat >= lat_min_e) & (da.lat <= lat_max_e)

    out = da.where(lat_mask & lon_mask, drop=True)

    # ---- fallback: still empty? snap to nearest grid point at bbox center
    if out.sizes.get("lat", 0) == 0 or out.sizes.get("lon", 0) == 0:
        lon_c = (lon_min + lon_max) / 2.0
        lat_c = (lat_min + lat_max) / 2.0
        # find nearest coordinates
        nearest = da.sel(lon=lon_c, lat=lat_c, method="nearest")
        # re-select with those coords to keep 2D dims
        out = da.sel(lon=[float(nearest.lon)], lat=[float(nearest.lat)])

    return out


def _reproject_and_prepare(subset):
    """
    Reproject a spatial subset from EPSG:4326 to EPSG:3857 and prepare it for visualization.

    This function:
    1. Assigns the CRS (EPSG:4326) to the input subset.
    2. Sets the spatial dimensions to `lon` (x) and `lat` (y).
    3. Reprojects the subset to Web Mercator (EPSG:3857) using bilinear resampling.
    4. Fills NaN values with zeros and flips the data vertically if needed so the
       origin is at the top-left (common for raster images).
    5. Computes the bounding box extent in projected coordinates.

    Notes:
        - Uses bilinear resampling via `rasterio.enums.Resampling.bilinear`.
        - Vertical flip is applied if y-coordinates are ascending.
        - The extent is padded by half a pixel on all sides to align with pixel centers.
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
