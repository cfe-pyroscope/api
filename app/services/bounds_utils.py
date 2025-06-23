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