import pandas as pd
import numpy as np
import numpy.typing as npt
import xarray as xr
import logging


def _iso_utc_str(dt_like) -> str:
    """
    Convert a datetime-like object to an ISO 8601 UTC string at midnight (00:00:00Z).
    """
    ts = pd.to_datetime(dt_like, utc=True).normalize()
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _iso_utc_ndarray(values) -> pd.DatetimeIndex:
    """
    Convert datetime-like values (NumPy ndarray or scalar) to a normalized UTC DatetimeIndex.
    Each value is truncated to midnight (00:00:00) in UTC.
    """
    return pd.to_datetime(values, utc=True).normalize().sort_values()


def _iso_drop_tz(s: str) -> pd.Timestamp:
    """
    Parse an ISO 8601 string into a naive Timestamp, dropping any timezone info.

    If the string has a timezone, it is removed without conversion, so the
    local clock time is preserved. The result is truncated to whole seconds.
    """
    ts = pd.Timestamp(s)
    return (ts.tz_localize(None) if ts.tz is not None else ts).replace(microsecond=0)


def _iso_naive_utc(dt_str: str) -> pd.Timestamp:
    """
    Parse an ISO 8601 string into a timezone-naive UTC Timestamp.

    - Reuses _iso_utc_str for normalization & UTC handling.
    - Returns a naive pd.Timestamp.
    """
    return pd.to_datetime(_iso_utc_str(dt_str))


def _naive_utc_ts(dt_like) -> pd.Timestamp:
    """
    Convert a datetime-like to a tz-naive UTC midnight timestamp.
    """
    # tz-aware UTC → midnight → tz-naive
    ts = pd.to_datetime(dt_like, utc=True).normalize()
    return ts.tz_localize(None)


def _naive_utc_ndarray(values) -> pd.DatetimeIndex:
    """
    Convert an array of datetime-like values to tz-naive UTC midnights.
    """
    ts = pd.to_datetime(values, utc=True).normalize()
    return ts.tz_localize(None)


def _normalize_times(base_time: str, forecast_time: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Normalize base and forecast times to tz-naive UTC midnights.

    Args:
        base_time (str): Base time in ISO8601 format.
        forecast_time (str): Forecast time in ISO8601 format.

    Returns:
        tuple[pd.Timestamp, pd.Timestamp]: Normalized base and forecast times.
    """
    return _naive_utc_ts(base_time), _naive_utc_ts(forecast_time)


def _match_base_time(ds: xr.Dataset, req_base: pd.Timestamp) -> pd.Timestamp:
    """
    Match a requested base date to the Dataset's 'base_time' coordinate.
    Ignore hours; select any timestamp whose UTC date equals the requested date.
    Prefer 00Z if available for that date; otherwise choose the earliest hour.
    Return tz-naive, second-precision pandas Timestamp taken from the dataset.
    """

    # Normalize request to UTC date (00:00 on that date in UTC, used only for comparison)
    req_date_utc = pd.to_datetime(req_base, utc=True).normalize().date()

    # utc-aware copies for robust date-only comparisons
    # originals for returning EXACT labels present in the dataset
    base_vals = pd.to_datetime(ds["base_time"].values)  # original labels (likely tz-naive numpy datetime64[ns])
    base_vals_utc = pd.to_datetime(ds["base_time"].values, utc=True)  # for comparison only

    # Filter all candidates on the same UTC date
    same_date_idx = [i for i, ts in enumerate(base_vals_utc) if ts.date() == req_date_utc]

    # Sort by hour while keeping track of original indices
    same_date_pairs = sorted(((i, base_vals_utc[i]) for i in same_date_idx), key=lambda p: p[1])
    # Prefer 00Z if present, else earliest
    chosen_idx, chosen_utc = next(
        ((i, ts) for i, ts in same_date_pairs if ts.hour == 0),
        same_date_pairs[0]
    )
    # return the original label (exactly as stored), coerced to tz-naive seconds
    chosen_orig = pd.Timestamp(base_vals[chosen_idx])
    return chosen_orig.tz_localize(None).replace(microsecond=0)


def _match_forecast_time(ds: xr.Dataset, matched_base: pd.Timestamp, req_fcst: pd.Timestamp) -> pd.Timestamp:
    """
    Match a requested forecast DATE to the Dataset's 'forecast_time' values
    for a specific base_time. Ignore hours and match only by date.

    Returns
    -------
    pd.Timestamp
        The matched forecast time (tz-naive, second precision).
    """
    # Select the forecast_time values for this base_time
    ds_bt = ds.sel(base_time=matched_base)
    # Keep original labels + utc-aware copies for date-only compare
    fcst_vals = pd.to_datetime(ds_bt["forecast_time"].values)  # original labels
    fcst_vals_utc = pd.to_datetime(ds_bt["forecast_time"].values, utc=True)  # compare only

    if fcst_vals_utc.size == 0:
        raise ValueError(f"No forecast times available for base_time '{matched_base.isoformat()}'.")

    # Normalize the request to its UTC date
    req_date = pd.to_datetime(req_fcst, utc=True).normalize().date()

    # Find all candidates with that same date
    same_date_idx = [i for i, ts in enumerate(fcst_vals_utc) if ts.date() == req_date]
    if not same_date_idx:
        # No match → build a helpful error with available forecast dates
        unique_dates = sorted({ts.date() for ts in fcst_vals_utc})
        examples = ", ".join(d.isoformat() for d in unique_dates[:5])
        raise ValueError(
            f"forecast_date '{req_date.isoformat()}' not found for base_time '{matched_base.isoformat()}'. "
            f"Available dates: {examples}"
        )

    # Sort the candidates by hour (UTC view), prefer 00Z if present
    same_date_pairs = sorted(((i, fcst_vals_utc[i]) for i in same_date_idx), key=lambda p: p[1])
    chosen_idx, chosen_utc = next(
        ((i, ts) for i, ts in same_date_pairs if ts.hour == 0),
        same_date_pairs[0]
    )
    # Return the exact original label from the dataset, coerced to tz-naive seconds
    chosen_orig = pd.Timestamp(fcst_vals[chosen_idx])
    return chosen_orig.tz_localize(None).replace(microsecond=0)
