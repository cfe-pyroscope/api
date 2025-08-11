import math
import xarray as xr
from pathlib import Path
import pandas as pd
import zarr
from config.config import settings


def _list_nc_files(data_path: Path):
    """Return a sorted list of NetCDF (.nc) files under data_path."""
    return sorted([f for f in Path(data_path).iterdir() if f.is_file() and f.suffix == '.nc'])


def _safe_chunks_for_fopi(probe_ds: xr.Dataset) -> dict:
    """Compute small, memory-safe chunks for very large 'fopi' datasets (~4MB target per chunk)."""
    # Inspect sizes and dtype
    lat_size = probe_ds.sizes["lat"]
    lon_size = probe_ds.sizes["lon"]
    # Use the first variable's dtype as a proxy (all usually same in these products)
    first_var = next(iter(probe_ds.data_vars.values()))
    dtype_size = first_var.dtype.itemsize  # bytes per element

    # Target small chunks (~4 MB) to stay far from 2 GB codec limit even with many vars
    target_bytes = 4 * 1024**2
    max_elements = max(1, target_bytes // dtype_size)
    aspect_ratio = lon_size / lat_size if lat_size else 1.0

    chunk_lat = max(1, int((max_elements / aspect_ratio) ** 0.5))
    chunk_lon = max(1, int(chunk_lat * aspect_ratio))

    # Ensure we actually split (don’t allow full-grid single chunk)
    if chunk_lat >= lat_size:
        chunk_lat = max(1, lat_size // 4 or 1)
    if chunk_lon >= lon_size:
        chunk_lon = max(1, lon_size // 4 or 1)

    return {"time": 1, "lat": chunk_lat, "lon": chunk_lon}


def _choose_chunks(index: str, probe_ds: xr.Dataset) -> dict:
    """Decide chunk sizes based on the dataset index ('fopi' uses small chunks; otherwise coarse)."""
    if index == "fopi":
        return _safe_chunks_for_fopi(probe_ds)
    # Faster path: coarser chunks (full lat/lon), time chunk=1
    # This keeps writes/reads snappy on moderate-sized files.
    return {"time": 1, "lat": probe_ds.sizes["lat"], "lon": probe_ds.sizes["lon"]}


def _wrap_longitudes_if_needed(ds: xr.Dataset, index: str) -> xr.Dataset:
    """For 'fopi', wrap longitude >180° to [-180, 180) and sort the longitude coordinate, if needed."""
    if index == "fopi" and ('lon' in ds.coords) and (float(ds['lon'].max()) > 180):
        import numpy as np  # local import to avoid touching global imports
        lons = ds['lon'].values
        shifted_lons = (lons + 180) % 360 - 180
        sort_idx = np.argsort(shifted_lons)
        ds = ds.isel(lon=sort_idx)
        ds = ds.assign_coords(lon=shifted_lons[sort_idx])
        print("Applied longitude wrap to [-180, 180) and sorted longitude coordinate.")
    return ds


def _check_consistency(ds: xr.Dataset, expected_vars, expected_dims, file_name: str):
    """Validate that variable names and dimension order match the first dataset."""
    if set(ds.data_vars.keys()) != expected_vars:
        raise ValueError(f"Variable mismatch in {file_name}. Expected {expected_vars}, got {set(ds.data_vars.keys())}")
    if tuple(ds.sizes.keys()) != expected_dims:
        raise ValueError(f"Dimension order mismatch in {file_name}. Expected {expected_dims}, got {tuple(ds.sizes.keys())}")


def _extract_base_time(ds: xr.Dataset, index: str) -> pd.Timestamp:
    """Extract base_time from first 'time' value; floor to day for 'pof'."""
    return (
        pd.Timestamp(ds.time.values[0]).floor('D')
        if index == "pof"
        else pd.Timestamp(ds.time.values[0])
    )


def _prepare_for_write(ds: xr.Dataset, base_time: pd.Timestamp, index: str) -> xr.Dataset:
    """Rename dims/coords and enforce target ordering before writing."""
    # Rename 'time' dimension to 'forecast_time'
    ds = ds.rename({"time": "forecast_time"})  # forecast_time contains forecast dates

    # Adjust forecast_time coordinate values if needed
    ds = ds.assign_coords(
        forecast_time=ds.forecast_time.dt.floor('D') if index == "pof" else ds.forecast_time
    )

    # Add base_time as a coordinate and expand to make it its own dimension
    ds = ds.assign_coords(base_time=base_time)
    ds = ds.expand_dims('base_time')

    # Ensure consistent dimension ordering
    ds = ds.transpose('base_time', 'forecast_time', 'lat', 'lon')
    return ds


def _write_zarr(ds: xr.Dataset, output_zarr_path: Path, first: bool):
    """Write or append a dataset to the Zarr store (Zarr v2, consolidate later)."""
    if first:
        ds.to_zarr(
            output_zarr_path,
            mode='w',
            consolidated=False,  # consolidate once at the end
            zarr_format=2  # force Zarr v2 for compatibility + consolidation
        )
    else:
        ds.to_zarr(
            output_zarr_path,
            mode='a',
            append_dim='base_time',
            consolidated=False,
            zarr_format=2
        )


def _consolidate_metadata(output_zarr_path: Path):
    """Consolidate Zarr metadata at the end to create '.zmetadata'."""
    print("Consolidating Zarr metadata...")
    zarr.consolidate_metadata(str(output_zarr_path))
    print("Zarr file successfully written and consolidated.")


def merge_netcdf_to_zarr(index: str):
    """
    Convert a sequence of large NetCDF forecast files into a single, chunked Zarr store (Zarr v2 consolidated).

    For index == "fopi" (very large >4GB files on laptop):
    - Apply memory-safe, small lat/lon chunks and time chunk=1 to avoid >2GB codec limits and big boolean masks.
    - Automatically wrap longitudes >180° to the [-180, 180) range and sort the longitude coordinate.
    - Write incrementally (first file mode='w', then append_dim='base_time').

    For index != "fopi" (e.g., "pof"):
    - Use a faster path with coarser chunking (full lat/lon, time chunk=1), still writing incrementally.
    - This is faster for moderate-sized files while remaining safe.

    Common steps:
    1) Add 'base_time' from first timestamp of 'time'.
    2) Rename 'time' → 'forecast_time'; enforce order ('base_time','forecast_time','lat','lon').
    3) Consolidate metadata at the end to create '.zmetadata'.

    Reasons of this approach:
    - Handles very large files on a laptop without blowing memory.
    - Corrects longitude coordinates for fopi datasets with values >180°, improving spatial consistency.
    - Produces a standard Zarr v2 consolidated store.
    - Allows lazy reads (e.g., select one base_time).

    Requirements before running:
    - Ensure that all NetCDF files have matching variable names and dimension orders.
    - For "fopi", lat/lon chunk sizes are auto-computed to keep chunks small.
    - For "fopi", if longitude coordinates exceed 180°, they will be shifted and sorted.
    """
    print(f"Starting Zarr conversion for index='{index}'")

    # Paths for input NetCDF files and output Zarr store
    data_path = settings.NC_PATH / index  # Directory containing NetCDF files
    output_zarr_path = settings.ZARR_PATH / index / f"{index}.zarr"  # Output Zarr store

    # List all NetCDF files
    files = _list_nc_files(data_path)
    print(f"Found {len(files)} NetCDF files to process.")

    expected_vars = None
    expected_dims = None

    # Process and write each file one at a time
    for i, file in enumerate(files, start=1):
        print(f"[{i}/{len(files)}] Opening file: {file.name}")

        # Probe once without chunking to decide chunk strategy and check metadata
        probe = xr.open_dataset(file, engine="netcdf4")
        chunks = _choose_chunks(index, probe)
        print(f"Using chunk sizes: forecast_time=1, lat={chunks['lat']}, lon={chunks['lon']}")
        probe.close()

        # Open with chosen chunking
        ds = xr.open_dataset(file, chunks=chunks, engine="netcdf4")

        # wrap/shift longitudes to [-180, 180) for fopi if needed ---
        ds = _wrap_longitudes_if_needed(ds, index)

        # Consistency checks
        if i == 1:
            expected_vars = set(ds.data_vars.keys())
            expected_dims = tuple(ds.sizes.keys())
        else:
            _check_consistency(ds, expected_vars, expected_dims, file.name)

        # Extract base_time from the first value in the time dimension
        base_time = _extract_base_time(ds, index)
        print(f"Initial time extracted: {base_time}")

        # Prepare dataset for write (rename, coords, ordering)
        ds = _prepare_for_write(ds, base_time, index)

        # Write to Zarr
        _write_zarr(ds, output_zarr_path, first=(i == 1))

        print(f"[{i}/{len(files)}] File {file.name} written to Zarr store.")

    # Consolidate metadata once after all files are written
    _consolidate_metadata(output_zarr_path)

merge_netcdf_to_zarr("fopi")
