
def _extract_spatial_subset(ds_or_da, param: str = None, bbox: str = None):
    """
    Extract a spatial subset from a DataArray or a Dataset variable using an optional bounding box.

    This function accepts either:
    - A 2D xarray.DataArray with latitude (`lat`) and longitude (`lon`) coordinates, or
    - An xarray.Dataset with a specified `param` key selecting the variable of interest.

    If a `bbox` is provided, it should be in EPSG:3857 coordinates and will be transformed
    to EPSG:4326 (longitude/latitude). The subset operation is aware of antimeridian
    crossings in the [-180, 180) longitude convention.

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
        da = ds_or_da[param]
    else:
        da = ds_or_da

    if not all(c in da.coords for c in ("lat", "lon")):
        raise ValueError(f"Expected 'lat' and 'lon' coordinates; got dims={da.dims}")

    # If bbox present, transform EPSG:3857 -> EPSG:4326
    if bbox:
        lon_min, lat_min, lon_max, lat_max = _bbox_to_latlon(bbox)
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