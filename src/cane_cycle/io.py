"""Reading Sentinel-2 field time series from CSV."""

from pathlib import Path

import pandas as pd

# Columns that must exist in the input CSV.
REQUIRED_COLUMNS = ["date", "field_id", "pixelTotal", "cloudyPerc"]

# Sentinel-2 bands expected in the input CSV (B10 is not delivered in L2A).
BAND_COLUMNS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]

# Date format of the `date` column in the raw file (e.g. 20200404).
DATE_FORMAT = "%Y%m%d"


def load_s2_csv(path, fields=None, drop_duplicates=True):
    """Load a Sentinel-2 field time series CSV.

    Parameters
    ----------
    path : str or Path
        Path to the CSV file (one row per date and field).
    fields : list of int, optional
        Field ids to keep. If None, all fields are kept.
    drop_duplicates : bool, default True
        If True, keep a single row per (field_id, date): the one with the
        largest `pixelTotal` (most complete field coverage); ties are broken
        by the lowest `cloudyPerc`.

    Returns
    -------
    pandas.DataFrame
        One row per (field_id, date), sorted by field_id and date, with
        `date` as datetime64 and an extra `coverage` column
        (pixelTotal / maximum pixelTotal of that field, from 0 to 1).
    """
    df = pd.read_csv(Path(path))

    missing = [c for c in REQUIRED_COLUMNS + BAND_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {path}: {missing}")

    df["date"] = pd.to_datetime(df["date"].astype(str), format=DATE_FORMAT)

    if fields is not None:
        df = df[df["field_id"].isin(fields)]

    if drop_duplicates:
        # Best row first, then keep the first row of each (field_id, date).
        df = df.sort_values(
            ["field_id", "date", "pixelTotal", "cloudyPerc"],
            ascending=[True, True, False, True],
        )
        df = df.drop_duplicates(subset=["field_id", "date"], keep="first")

    # Fraction of the field covered by the image, relative to the best coverage.
    max_pixels = df.groupby("field_id")["pixelTotal"].transform("max")
    df["coverage"] = df["pixelTotal"] / max_pixels

    df = df.sort_values(["field_id", "date"]).reset_index(drop=True)
    return df
