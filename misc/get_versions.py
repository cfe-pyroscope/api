import zarr, numcodecs
print("zarr:", zarr.__version__)
print("numcodecs:", numcodecs.__version__)


import os
from pyproj import datadir, show_versions
print("PROJ_DATA:", os.environ.get("PROJ_DATA"))
print("pyproj data dir:", datadir.get_data_dir())
show_versions()