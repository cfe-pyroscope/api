import pandas as pd
import numpy as np
import xarray as xr
from datetime import datetime, timezone
from config.logging_config import logger


def _iso_utc(dt_like) -> str:
    """
    Convert datetime-like values (np.datetime64, pandas.Timestamp, datetime)
    to ISO-8601 in UTC with a trailing 'Z'.
    """
    ts = pd.Timestamp(dt_like)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    # Use 'Z' instead of +00:00 for compactness
    return ts.isoformat().replace("+00:00", "Z")


def _parse_naive(dt_str: str) -> pd.Timestamp:
    """
    Parse an ISO 8601-like datetime string into a timezone-naive pandas Timestamp in UTC.

    This function parses the input string into a pandas Timestamp and ensures the
    result is timezone-unaware (naive) while preserving the actual moment in time.
    If the parsed datetime is timezone-aware (e.g., contains 'Z' for UTC or an offset
    like '+02:00'), it is first converted to UTC, then the timezone information is
    removed. If the input is already naive, it is left as-is except for microsecond
    truncation.

    Parameters
    ----------
    dt_str : str
        ISO 8601-like datetime string, with or without timezone information.

    Returns
    -------
    pd.Timestamp
        A naive pandas Timestamp in UTC-equivalent time, with microseconds set to zero.

    Notes
    -----
    - This function differs from `_parse_iso_naive` in that it normalizes all
      timezone-aware datetimes to UTC before removing the timezone, ensuring
      consistent absolute time representation.
    - Examples:
        >>> _parse_naive("2025-07-11T12:34:56.789Z")
        Timestamp('2025-07-11 12:34:56')
        >>> _parse_naive("2025-07-11T12:34:56+02:00")
        Timestamp('2025-07-11 10:34:56')  # converted to UTC first
    """
    ts = pd.to_datetime(dt_str, errors="raise")
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    else:
        ts = ts.tz_localize(None)
    return pd.Timestamp(ts.replace(microsecond=0))


def _parse_iso_naive(s: str) -> pd.Timestamp:
    """
    Parse an ISO 8601-like datetime string into a naive (timezone-unaware) pandas Timestamp.

    This function drops any timezone information from the parsed datetime without
    converting it to UTC. If the input string is timezone-aware (e.g., contains an
    offset like '+02:00' or 'Z'), the offset is simply removed and the resulting
    naive timestamp retains the same local clock time. The timestamp is truncated
    to whole-second precision by setting microseconds to zero.

    Parameters
    ----------
    s : str
        An ISO 8601-like datetime string (with or without timezone information).

    Returns
    -------
    pd.Timestamp
        A timezone-naive pandas Timestamp with microseconds removed.

    Notes
    -----
    - This function does not normalize to UTC. Use a conversion function if you
      require a consistent UTC-based naive timestamp.
    - Examples:
        >>> _parse_iso_naive("2025-07-11T12:34:56.789Z")
        Timestamp('2025-07-11 12:34:56')
        >>> _parse_iso_naive("2025-07-11T12:34:56+02:00")
        Timestamp('2025-07-11 12:34:56')
    """
    ts = pd.Timestamp(s)
    return (ts.tz_localize(None) if ts.tz is not None else ts).replace(microsecond=0)

def _normalize_times(base_time: str, forecast_time: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Parse and normalize base and forecast time strings into naive pandas Timestamps.
    Accepts flexible ISO-like formats (with/without 'Z') and normalizes to seconds.

    Returns
    -------
    tuple[pd.Timestamp, pd.Timestamp]
        A pair `(base_ts, fcst_ts)` of naive `pd.Timestamp` objects normalized
        to whole seconds.
    """
    return _parse_naive(base_time), _parse_naive(forecast_time)


def _match_base_time(ds: xr.Dataset, req_base: pd.Timestamp) -> pd.Timestamp:
    """
    Match a requested base time to the Dataset's 'base_time' coordinate.

    Steps:
    1) Normalize all dataset base times to timezone-naive, second precision.
    2) Try an exact match with `req_base`.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset that must contain a 'base_time' coordinate.
    req_base : pd.Timestamp
        Requested base (initialization) time; expected to be naive and at
        second precision.

    Returns
    -------
    pd.Timestamp
        The matched base time (naive, second precision).
        Note: the current implementation ends with `.to_pydatetime()`, which
        actually returns a `datetime.datetime`. Replace that with
        `pd.Timestamp(req_base)` to keep the annotated return type.

    Raises
    ------
    ValueError
        If 'base_time' is missing or no suitable match is found.

    Notes
    -----
    All dataset base times are made naive (`tz_localize(None)`) and trimmed to
    second precision (microseconds set to 0) before comparison.
    """
    if "base_time" not in ds.coords:
        raise ValueError("Dataset is missing 'base_time' coordinate.")

    # Enforce 00Z if that's the only valid cycle in your data
    if req_base.hour != 0:
        raise ValueError(
            f"Expected base_time at 00Z; got {req_base.isoformat(timespec='seconds')}."
        )

    # Normalize dataset coordinate values to naive, second precision
    base_vals = pd.to_datetime(ds["base_time"].values)
    base_vals = pd.to_datetime(
        [pd.Timestamp(bv).tz_localize(None).replace(microsecond=0) for bv in base_vals]
    )

    match_idx = np.where(base_vals == req_base)[0]
    if match_idx.size == 0:
        same_day = [bv for bv in base_vals if bv.date() == req_base.date()]
        hint = ", ".join(pd.Timestamp(x).isoformat(timespec="seconds") for x in same_day[:5])
        raise ValueError(
            f"base_time '{req_base.isoformat(timespec='seconds')}' not found (00Z only). "
            + (f"Available that date: {hint}" if same_day else "No base_time on that date.")
        )

    # Return as pandas Timestamp (naive, second precision)
    return pd.Timestamp(base_vals[match_idx[0]])


def _match_forecast_time(ds: xr.Dataset, matched_base: pd.Timestamp, req_fcst: pd.Timestamp) -> pd.Timestamp:
    """
    Match a requested forecast time to the Dataset's 'forecast_time' coordinate for
    a specific base time.

    This performs an exact label match at second precision against the list of
    forecast times available for `matched_base`. If no match is found, a helpful
    error is raised showing a few available examples.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing 'base_time' and 'forecast_time' coordinates.
    matched_base : pd.Timestamp
        The already-matched base (initialization) time, expected to be timezone-naive
        and trimmed to second precision.
    req_fcst : pd.Timestamp
        Requested forecast valid time, expected to be timezone-naive and at
        second precision.

    Returns
    -------
    pd.Timestamp
        The matched forecast time (naive, second precision). On success this is the
        same value as `req_fcst`.

    Raises
    ------
    ValueError
        If 'forecast_time' is missing in `ds`, or if `req_fcst` is not present for
        the given `matched_base`.

    Notes
    -----
    - Dataset forecast times are normalized to timezone-naive, second precision
      before comparison.
    - Only exact matches are supported
    """
    if "forecast_time" not in ds.coords:
        raise ValueError("Dataset is missing 'forecast_time' coordinate.")

    ds_bt = ds.sel(base_time=matched_base)
    fcst_vals = pd.to_datetime(ds_bt["forecast_time"].values)
    fcst_vals = [pd.Timestamp(x).tz_localize(None).replace(microsecond=0) for x in fcst_vals]

    try:
        next(i for i, v in enumerate(fcst_vals) if v == req_fcst)
    except StopIteration:
        hint = ", ".join(pd.Timestamp(x).isoformat(timespec="seconds") for x in fcst_vals[:5])
        raise ValueError(
            f"forecast_time '{req_fcst.isoformat()}' not found for base_time '{matched_base.isoformat()}'. "
            f"Available examples: {hint}"
        )
    return req_fcst
