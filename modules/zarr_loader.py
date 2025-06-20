import xarray as xr
import numpy as np
from pathlib import Path
import re

BASE_NC_PATH = Path("data/nc")
BASE_ZARR_PATH = Path("data/zarr")

FILENAME_PATTERNS = {
    "fopi": re.compile(r"fopi_(\d{10})\.nc"),
    "pof": re.compile(r"POF_V2_(\d{4})_(\d{2})_(\d{2})_FC\.nc"),
}


def get_latest_nc_file(index: str) -> Path:
    folder = BASE_NC_PATH / index
    pattern = FILENAME_PATTERNS.get(index)

    if not pattern:
        raise ValueError(f"No filename pattern defined for index '{index}'")

    files = list(folder.glob("*.nc"))
    matched = []

    for f in files:
        m = pattern.search(f.name)
        if m:
            if index == "fopi":
                timestamp = m.group(1)
            elif index == "pof":
                y, m_, d = m.groups()
                timestamp = f"{y}{m_}{d}00"
            else:
                continue
            matched.append((timestamp, f))

    if not matched:
        raise FileNotFoundError(f"No valid NetCDF files found for index '{index}'")

    matched.sort(key=lambda x: x[0], reverse=True)
    return matched[0][1]


def clean_time(ds, time_min=0, time_max=1e5):
    time = ds['time']
    valid_time_mask = time.where(
        time.notnull() & np.isfinite(time) & (time >= time_min) & (time <= time_max),
        drop=True
    )
    return ds.sel(time=valid_time_mask['time'])


def convert_nc_to_zarr(index: str, force=False) -> Path:
    nc_path = get_latest_nc_file(index)

    if index == "fopi":
        timestamp = re.search(r"fopi_(\d{10})", nc_path.name).group(1)
    elif index == "pof":
        y, m, d = re.search(r"POF_V2_(\d{4})_(\d{2})_(\d{2})", nc_path.name).groups()
        timestamp = f"{y}{m}{d}00"
    else:
        raise ValueError(f"Zarr timestamp logic not implemented for index '{index}'")

    zarr_store = BASE_ZARR_PATH / index / f"{index}_{timestamp}.zarr"

    if not zarr_store.exists() or force:
        ds = xr.open_dataset(nc_path, chunks={"time": 1, "lat": 256, "lon": 256}, decode_times=False)
        ds = clean_time(ds)
        zarr_store.parent.mkdir(parents=True, exist_ok=True)
        ds.to_zarr(zarr_store, mode="w", consolidated=False)

    return zarr_store


def load_zarr(index: str) -> xr.Dataset:
    path = convert_nc_to_zarr(index)
    ds = xr.open_zarr(path)

    # ✅ Solo per FOPI: converti longitudes 0–360 → -180–180
    if index == "fopi" and 'lon' in ds.coords and ds.lon.max() > 180:
        lons = ds.lon.values
        shifted_lons = (lons + 180) % 360 - 180
        sort_idx = np.argsort(shifted_lons)
        ds = ds.isel(lon=sort_idx)
        ds['lon'].values[:] = shifted_lons[sort_idx]

    return ds


