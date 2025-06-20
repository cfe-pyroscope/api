def wrap_longitude(lon: float) -> float:
    """Convert -180:180 → 0:360"""
    return lon if lon >= 0 else lon + 360

def normalize_bbox(lat_min, lon_min, lat_max, lon_max, wrap=False):
    if wrap:
        lon_min = wrap_longitude(lon_min)
        lon_max = wrap_longitude(lon_max)

    if lon_min > lon_max:
        lon_min, lon_max = lon_max, lon_min
    if lat_min > lat_max:
        lat_min, lat_max = lat_max, lat_min

    return lat_min, lon_min, lat_max, lon_max
