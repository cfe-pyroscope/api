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


def _extract_spatial_subset(ds_or_da, param: str = None, bbox: str = None):
    """
    Extract a spatial subset from a DataArray or a Dataset variable using an optional bounding box.

    This function accepts either:
    - A 2D xarray.DataArray with latitude (`lat`) and longitude (`lon`) coordinates, or
    - An xarray.Dataset with a specified `param` key selecting the variable of interest.

    If a `bbox` is provided, it should be in EPSG:3857 coordinates and will be transformed
    to EPSG:4326 (longitude/latitude). The subset operation is aware of antimeridian
    crossings in the [-180, 180) longitude convention.

    Args:
        ds_or_da (xarray.DataArray | xarray.Dataset):
            Input spatial dataset. Must have `lat` and `lon` coordinates.
        param (str, optional):
            Name of the variable to extract when passing a Dataset. Required if `ds_or_da` is a Dataset.
        bbox (str, optional):
            Bounding box in EPSG:3857 coordinates, formatted as:
            "x_min,y_min,x_max,y_max".
            If omitted, the entire spatial extent of the dataset is used.

    Returns:
        xarray.DataArray:
            Subset of the input data containing only points within the bounding box.
            If no `bbox` is provided, returns the full input extent.

    Notes:
        - Antimeridian handling: If `lon_min > lon_max` after transformation,
          the function assumes the bounding box crosses the antimeridian and
          applies a logical OR mask to select the correct range.
        - Latitude bounds are automatically reordered if inverted (min > max)
          after coordinate transformation.
        - Minor floating-point precision drift is clamped during coordinate checks.
    """
    # Resolve to DataArray
    if isinstance(ds_or_da, xr.Dataset):
        if not param:
            raise ValueError("param is required when passing a Dataset.")
        da = ds_or_da[param]
    else:
        da = ds_or_da

    if not all(c in da.coords for c in ("lat", "lon")):
        raise ValueError(f"Expected 'lat' and 'lon' coordinates; got dims={da.dims}")

    # If bbox present, transform EPSG:3857 -> EPSG:4326
    if bbox:
        bbox = _decode_coords(bbox).strip()
        x_min, y_min, x_max, y_max = map(float, bbox.split(","))
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        lon_min, lat_min = transformer.transform(x_min, y_min)
        lon_max, lat_max = transformer.transform(x_max, y_max)
    else:
        lat_min, lat_max = float(da.lat.min()), float(da.lat.max())
        lon_min, lon_max = float(da.lon.min()), float(da.lon.max())

    # Clamp tiny fp drift and ensure latitude ordering
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min

    # Antimeridian-aware longitude mask for [-180,180)
    # crossing happens when lon_min > lon_max (e.g., 170 -> -170)
    crosses = lon_min > lon_max
    if crosses:
        lon_mask = (da.lon >= lon_min) | (da.lon <= lon_max)
    else:
        lon_mask = (da.lon >= lon_min) & (da.lon <= lon_max)

    lat_mask = (da.lat >= lat_min) & (da.lat <= lat_max)
    return da.where(lat_mask & lon_mask, drop=True)


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

    Args:
        subset (xarray.DataArray):
            Input data array with `lat` and `lon` coordinates in EPSG:4326.

    Returns:
        tuple:
            - **numpy.ndarray**:
                2D array of reprojected data, with NaNs replaced by `0.0` and
                potentially flipped vertically for correct raster orientation.
            - **list[float]**:
                Bounding box extent in EPSG:3857 as
                `[x_min, x_max, y_min, y_max]`, with edges expanded by half a pixel
                to match raster conventions.

    Notes:
        - Uses bilinear resampling via `rasterio.enums.Resampling.bilinear`.
        - Vertical flip is applied if y-coordinates are ascending.
        - The extent is padded by half a pixel on all sides to align with pixel centers.

    Example:
        >>> data, extent = _reproject_and_prepare(subset)
        >>> data.shape
        (256, 256)
        >>> extent
        [-8237642.0, -8235642.0, 4970351.0, 4972351.0]
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
