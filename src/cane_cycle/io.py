"""Reading Sentinel-2 field time series from CSV."""

import warnings
from pathlib import Path

import pandas as pd

# Non-band columns that must exist in the input CSV.
REQUIRED_COLUMNS = ["date", "field_id", "pixelTotal"]

# Default Sentinel-2 bands expected in the input CSV. Passed as a function
# parameter (not hardcoded) because different extracts carry different bands
# (e.g. some include B8A and B9, others don't).
DEFAULT_BAND_COLUMNS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B11", "B12"]

# Every band that may appear in a Sentinel-2 L2A extract. All of these are
# converted to numbers and checked, not only the required ones.
KNOWN_BAND_COLUMNS = [
    "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"
]

# Date format of the `date` column in the raw file (e.g. 20200404).
DATE_FORMAT = "%Y%m%d"


def load_s2_csv(path, fields=None, drop_duplicates=True, band_columns=DEFAULT_BAND_COLUMNS):
    """Load a Sentinel-2 field time series CSV.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file (one row per date and field).
    fields : list of int, optional
        Field ids to keep. If None, all fields are kept.
    drop_duplicates : bool, default True
        If True, keep a single row per (field_id, date): the one with the
        largest `pixelTotal` (most complete field coverage).
    band_columns : list of str, default DEFAULT_BAND_COLUMNS
        Band columns that must be present in the CSV. Pass a different list
        if the source file carries other bands (e.g. adding "B8A", "B9").
        Rows with no valid pixels (`pixelTotal == 0`, all bands NaN) are
        always dropped, since they carry no usable information.

    Returns
    -------
    pandas.DataFrame
        One row per (field_id, date), sorted by field_id and date, with
        `date` as datetime64 and an extra `coverage_pct` column: the
        percentage of valid pixels in the image, relative to the largest
        `pixelTotal` observed for that field (100 = the most complete image).
    """
    df = pd.read_csv(Path(path))

    missing = [c for c in REQUIRED_COLUMNS + list(band_columns) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["date"] = pd.to_datetime(df["date"].astype(str), format=DATE_FORMAT)

    # Band values that cannot be read as numbers (e.g. exported with thousand
    # separators by a spreadsheet) become NaN and the observation is dropped,
    # rather than silently guessing what the value was meant to be.
    present_bands = [b for b in KNOWN_BAND_COLUMNS if b in df.columns]
    for band in present_bands:
        df[band] = pd.to_numeric(df[band], errors="coerce")

    unreadable = df[present_bands].isna().any(axis=1)
    if unreadable.any():
        dates = sorted(df.loc[unreadable, "date"].dt.strftime("%Y-%m-%d").unique())
        warnings.warn(
            f"Dropped {int(unreadable.sum())} rows with unreadable band values "
            f"on {len(dates)} dates: {', '.join(dates)}",
            stacklevel=2,
        )
    df = df[~unreadable]

    # Drop rows with no valid pixels (empty observations, e.g. fully clouded scenes).
    df = df[df["pixelTotal"] > 0].reset_index(drop=True)

    if fields is not None:
        df = df[df["field_id"].isin(fields)]

    if drop_duplicates:
        # Best row first, then keep the first row of each (field_id, date).
        df = df.sort_values(
            ["field_id", "date", "pixelTotal"],
            ascending=[True, True, False],
        )
        df = df.drop_duplicates(subset=["field_id", "date"], keep="first")

    # Percentage of valid pixels, relative to the field's best coverage.
    max_pixels = df.groupby("field_id")["pixelTotal"].transform("max")
    df["coverage_pct"] = 100 * df["pixelTotal"] / max_pixels

    df = df.sort_values(["field_id", "date"]).reset_index(drop=True)
    return df
