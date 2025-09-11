
import numpy as np
import xarray as xr


def _agg_mean_median(da: xr.DataArray) -> tuple[float | None, float | None]:
    """
    Compute the mean and median of a DataArray over its spatial dimensions.

    This function eagerly loads the input `xarray.DataArray` into memory via `.to_numpy()`
    and computes statistics over the `lat` and `lon` dimensions if they are present.
    If neither `lat` nor `lon` is present, it assumes the array contains a single
    scalar value and returns that value for both statistics.
    """

    dims = [d for d in ("lat", "lon") if d in da.dims]

    def _to_opt_float(v) -> float | None:
        if v is None:
            return None
        try:
            v = float(v)
        except Exception:
            return None
        return None if (isinstance(v, float) and np.isnan(v)) else v

    # Scalar (no lat/lon): just return the single value for both stats
    if not dims:
        v = da.to_numpy().item()
        f = _to_opt_float(v)
        return f, f

    # Reduce over lat/lon by flattening them into a single axis
    try:
        arr = da.stack(z=tuple(dims)).to_numpy()  # eager NumPy array
        # If everything is masked/NaN, nanmean/nanmedian -> NaN; we'll map to None below.
        mean_v = np.nanmean(arr) if arr.size else np.nan
        med_v  = np.nanmedian(arr) if arr.size else np.nan
    except Exception:
        # Extremely defensive fallback
        arr = da.to_numpy()
        mean_v = np.nanmean(arr) if arr.size else np.nan
        med_v  = np.nanmedian(arr) if arr.size else np.nan

    return _to_opt_float(mean_v), _to_opt_float(med_v)
