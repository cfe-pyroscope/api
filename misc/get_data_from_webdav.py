import os
from dotenv import load_dotenv

import fsspec
import xarray as xr
import zarr


def get_zarr_size_info(zarr_group):
    """Get size information for a Zarr group"""
    total_size = 0
    array_info = {}

    for name, array in zarr_group.arrays():
        # Calculate uncompressed size in bytes
        array_size = array.nbytes
        total_size += array_size

        array_info[name] = {
            'shape': array.shape,
            'dtype': array.dtype,
            'size_bytes': array_size,
            'size_mb': array_size / (1024 * 1024),
            'chunks': getattr(array, 'chunks', None)
        }

    return {
        'total_size_bytes': total_size,
        'total_size_mb': total_size / (1024 * 1024),
        'total_size_gb': total_size / (1024 * 1024 * 1024),
        'arrays': array_info
    }


def get_webdav_storage_size(fs, zarr_path):
    """Get actual storage size of Zarr on WebDAV"""
    total_size = 0
    file_count = 0

    try:
        # Walk through all files in the Zarr directory
        for root, dirs, files in fs.walk(zarr_path):
            for file in files:
                file_path = f"{root.rstrip('/')}/{file}"
                try:
                    info = fs.info(file_path)
                    if 'size' in info:
                        total_size += info['size']
                        file_count += 1
                except Exception as e:
                    print(f"Could not get size for {file_path}: {e}")

    except Exception as e:
        print(f"Error walking directory: {e}")
        return None

    return {
        'storage_size_bytes': total_size,
        'storage_size_mb': total_size / (1024 * 1024),
        'storage_size_gb': total_size / (1024 * 1024 * 1024),
        'file_count': file_count
    }


def get_dataset_size_info(ds):
    """Get size information from xarray Dataset"""
    total_size = 0
    var_info = {}

    # Data variables
    for name, var in ds.data_vars.items():
        size_bytes = var.nbytes
        total_size += size_bytes
        var_info[name] = {
            'shape': var.shape,
            'dtype': var.dtype,
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024 * 1024)
        }

    # Coordinates
    for name, var in ds.coords.items():
        size_bytes = var.nbytes
        total_size += size_bytes
        var_info[f"coord_{name}"] = {
            'shape': var.shape,
            'dtype': var.dtype,
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024 * 1024)
        }

    return {
        'total_size_bytes': total_size,
        'total_size_mb': total_size / (1024 * 1024),
        'total_size_gb': total_size / (1024 * 1024 * 1024),
        'variables': var_info
    }


load_dotenv()
url = os.getenv("WEBDAV_URL")
user = os.getenv("WEBDAV_USER")
pw = os.getenv("WEBDAV_PASS")
if not url or not user or not pw:
    raise RuntimeError("Missing WEBDAV_URL/WEBDAV_USER/WEBDAV_PASS in environment")

fs = fsspec.filesystem(
    "webdav",
    base_url=url,
    auth=(user, pw),
    follow_redirects=True,  # avoids 301 on directory URLs
    # verify=False,         # only for testing self-signed certs
)

# Inner Zarr directory
zarr_rel = "fopi/fopi.zarr/fopi.zarr/"

# Create fsspec mapper
store = fs.get_mapper(zarr_rel)

# Open with Zarr - auto-detects format
try:
    zarr_group = zarr.open_group(store, mode='r')
    print("Successfully opened Zarr group")

    # Get Zarr size information
    print("\n=== ZARR SIZE ANALYSIS ===")
    size_info = get_zarr_size_info(zarr_group)
    print(f"Total uncompressed size: {size_info['total_size_gb']:.2f} GB")
    print(f"Individual arrays:")
    for name, info in size_info['arrays'].items():
        print(f"  {name}: {info['size_mb']:.2f} MB, shape: {info['shape']}, dtype: {info['dtype']}")
        if info['chunks']:
            print(f"    chunks: {info['chunks']}")

except Exception as e:
    print(f"Failed to open Zarr group: {e}")
    raise

# Get actual storage size on WebDAV
print("\n=== WEBDAV STORAGE SIZE ===")
storage_info = get_webdav_storage_size(fs, zarr_rel)
if storage_info:
    print(f"Actual storage size: {storage_info['storage_size_gb']:.2f} GB ({storage_info['file_count']} files)")
    if size_info:
        compression_ratio = size_info['total_size_bytes'] / storage_info['storage_size_bytes']
        print(f"Compression ratio: {compression_ratio:.2f}x")
else:
    print("Could not determine storage size")


# Convert to xarray Dataset
def zarr_to_xarray(zarr_group):
    """Convert Zarr group to xarray Dataset"""
    data_vars = {}
    coords = {}

    for name, array in zarr_group.arrays():
        # Get array data
        data = array[:]

        # Get attributes
        attrs = dict(array.attrs) if hasattr(array, 'attrs') else {}

        if array.ndim == 1:
            dims = [name]
        elif array.ndim == 2:
            dims = [f'{name}_dim_0', f'{name}_dim_1']
        elif array.ndim == 3:
            dims = [f'{name}_dim_0', f'{name}_dim_1', f'{name}_dim_2']
        else:
            dims = [f'{name}_dim_{i}' for i in range(array.ndim)]

        # Check if this might be a coordinate variable
        if name in ['time', 'lat', 'lon', 'latitude', 'longitude', 'x', 'y', 'z']:
            coords[name] = (dims, data, attrs)
        else:
            data_vars[name] = (dims, data, attrs)

    # Get group attributes
    group_attrs = dict(zarr_group.attrs) if hasattr(zarr_group, 'attrs') else {}

    # Create Dataset
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs=group_attrs
    )

    return ds


try:
    # Check for consolidated metadata
    consolidated = fs.exists(zarr_rel + ".zmetadata")
    ds = xr.open_zarr(store, consolidated=consolidated)
    print("\nOpened directly with xarray")
except Exception as e:
    print(f"\nDirect xarray opening failed: {e}")
    print("Converting manually from Zarr group...")
    ds = zarr_to_xarray(zarr_group)

print("\n=== DATASET INFORMATION ===")
print(ds)

# Get dataset size information
ds_size_info = get_dataset_size_info(ds)
print(f"\nDataset memory usage: {ds_size_info['total_size_gb']:.2f} GB")
print("Variables breakdown:")
for name, info in ds_size_info['variables'].items():
    if info['size_mb'] > 0.1:  # Only show variables > 0.1 MB
        print(f"  {name}: {info['size_mb']:.2f} MB, shape: {info['shape']}")