import os
from dotenv import load_dotenv

import fsspec
import zarr
import posixpath


def download_zarr_dir(fs, remote_zarr_dir, local_dir):
    """
    Recursively download a Zarr directory from a remote filesystem to a local folder.

    Parameters
    ----------
    fs : fsspec.AbstractFileSystem
        The (remote) filesystem, e.g. your WebDAV fs.
    remote_zarr_dir : str
        Remote path to the Zarr directory (folder that ends with `.zarr/`).
    local_dir : str
        Local destination directory.
    """
    os.makedirs(local_dir, exist_ok=True)

    total_bytes = 0
    file_count = 0

    for root, dirs, files in fs.walk(remote_zarr_dir):
        # Compute the relative subpath under the remote zarr directory
        rel = root[len(remote_zarr_dir):].lstrip("/")
        local_root = os.path.join(local_dir, rel) if rel else local_dir
        os.makedirs(local_root, exist_ok=True)

        for fname in files:
            rpath = posixpath.join(root.rstrip("/"), fname)
            lpath = os.path.join(local_root, fname)

            # Download the single file
            fs.get(rpath, lpath)

            # Stats
            try:
                info = fs.info(rpath)
                if "size" in info:
                    total_bytes += info["size"]
            except Exception:
                pass
            file_count += 1

    print(f"Downloaded {file_count} files "
          f"({total_bytes / (1024*1024):.2f} MB) to {os.path.abspath(local_dir)}")


def get_zarr_size_info(zarr_group):
    """Get size information for a Zarr group"""
    total_size = 0
    array_info = {}

    for name, array in zarr_group.arrays():
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

    for name, var in ds.data_vars.items():
        size_bytes = var.nbytes
        total_size += size_bytes
        var_info[name] = {
            'shape': var.shape,
            'dtype': var.dtype,
            'size_bytes': size_bytes,
            'size_mb': size_bytes / (1024 * 1024)
        }

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


# ====== MAIN SCRIPT ======

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
    follow_redirects=True,
)


index = "fopi"

# Build paths dynamically
zarr_rel = f"{index}/{index}.zarr/{index}.zarr/"
local_target = f"../data/zarr/{index}/{index}.zarr"

download_zarr_dir(fs, zarr_rel, local_target)

# Create fsspec mapper
store = fs.get_mapper(zarr_rel)

try:
    zarr_group = zarr.open_group(store, mode='r')
    print("Successfully opened Zarr group")

    print("\n=== ZARR SIZE ANALYSIS ===")
    size_info = get_zarr_size_info(zarr_group)
    print(f"Total uncompressed size: {size_info['total_size_gb']:.2f} GB")
    for name, info in size_info['arrays'].items():
        print(f"  {name}: {info['size_mb']:.2f} MB, shape: {info['shape']}, dtype: {info['dtype']}")
        if info['chunks']:
            print(f"    chunks: {info['chunks']}")

except Exception as e:
    print(f"Failed to open Zarr group: {e}")
    raise

print("\n=== WEBDAV STORAGE SIZE ===")
storage_info = get_webdav_storage_size(fs, zarr_rel)
if storage_info:
    print(f"Actual storage size: {storage_info['storage_size_gb']:.2f} GB ({storage_info['file_count']} files)")
    if size_info:
        compression_ratio = size_info['total_size_bytes'] / storage_info['storage_size_bytes']
        print(f"Compression ratio: {compression_ratio:.2f}x")
else:
    print("Could not determine storage size")


