import xarray as xr
import zarr
from pathlib import Path

DATA_PATH = Path("data/raw/fopi_2024120100.nc")
ZARR_STORE = Path("data/zarr/fopi_2024120100.zarr")


def convert_nc_to_zarr(force=False):
    if not ZARR_STORE.exists() or force:
        ds = xr.open_dataset(DATA_PATH, chunks={"time": 1, "lat": 256, "lon": 256})
        ds.to_zarr(ZARR_STORE, mode="w")
    return ZARR_STORE


def load_zarr():
    return xr.open_zarr(ZARR_STORE)
