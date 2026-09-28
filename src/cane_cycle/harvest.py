"""Harvest date detection and season (safra) boundaries.

A harvest date must meet two conditions at the same time:

1. `swir_above_nir`: B11 above B8A. A live canopy reflects more in the NIR
   than in the SWIR; straw and bare soil do the opposite. In the field
   spectral library every sugarcane sample has B8A above B11 and every straw
   and bare soil sample has the reverse, so the comparison marks the loss of
   the canopy without caring how bright the surface is. Cloud also has B8A
   above B11 in all library samples, so this condition rejects cloud too.
2. `low_ndvi`: NDVI at or below `ndvi_max`. Sugarcane never goes below 0.536
   in the library, straw and bare soil never exceed 0.212, so the default
   threshold of 0.30 sits inside that gap.

The harvest of a cycle is the *first* date of the year meeting both
conditions: the following dates are the same bare field, not new cuts.

A season (safra) then runs from one harvest to the next. The first and last
cycles of the record are open - they are bounded by the start and the end of
the series rather than by a harvest - and they are kept, flagged as open.
"""

import pandas as pd

from cane_cycle.vegetation_indices import add_indices

# Observations with B1 (coastal aerosol) above this are treated as cloud.
CLOUD_B1_MAX = 0.10

# NDVI at or below this means no live canopy (harvest candidate).
HARVEST_NDVI_MAX = 0.30


def filter_clouds(df, cloud_b1_max=CLOUD_B1_MAX):
    """Drop cloud contaminated observations. Always the first filter.

    In the field spectral library cloud starts at B1 = 0.113 and no land
    class exceeds 0.087.
    """
    return df[df["B1"] <= cloud_b1_max].copy()


def flag_harvest_conditions(df, ndvi_max=HARVEST_NDVI_MAX):
    """Return a copy of df with the two harvest conditions as 0/1 columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with band columns and NDVI.
    ndvi_max : float, default HARVEST_NDVI_MAX
        NDVI at or below this marks a date with no live canopy.

    Returns
    -------
    pandas.DataFrame
        Copy of df with `swir_above_nir` (B11 above the NIR band) and
        `low_ndvi` (NDVI at or below the threshold).
    """
    df = df.copy()
    nir = df["B8A"] if "B8A" in df.columns else df["B8"]

    df["swir_above_nir"] = (df["B11"] > nir).astype(int)
    df["low_ndvi"] = (df["NDVI"] <= ndvi_max).astype(int)
    return df


def detect_harvests(df, ndvi_max=HARVEST_NDVI_MAX, cloud_b1_max=CLOUD_B1_MAX):
    """Detect one harvest date per field and year.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with `field_id`, `date` and Sentinel-2 band columns. NDVI is
        computed if it is not already a column.
    ndvi_max : float, default HARVEST_NDVI_MAX
        NDVI at or below this marks a date with no live canopy.
    cloud_b1_max : float, default CLOUD_B1_MAX
        B1 threshold above which an observation is discarded as cloud.

    Returns
    -------
    pandas.DataFrame
        One row per field and year: `field_id`, `year`, `harvest_date` (the
        first date meeting both conditions) and `ndvi_at_harvest`.
    """
    if "NDVI" not in df.columns:
        df = add_indices(df, indices=["NDVI"])

    # Cloud always goes first.
    df = filter_clouds(df, cloud_b1_max=cloud_b1_max)
    df = flag_harvest_conditions(df, ndvi_max=ndvi_max)

    candidates = df[(df["swir_above_nir"] == 1) & (df["low_ndvi"] == 1)].copy()
    candidates["year"] = candidates["date"].dt.year

    harvests = (
        candidates.sort_values(["field_id", "date"])
        .groupby(["field_id", "year"], as_index=False)
        .agg(harvest_date=("date", "first"), ndvi_at_harvest=("NDVI", "first"))
    )
    return harvests


def build_seasons(df, harvests):
    """Build the season (safra) intervals between consecutive harvests.

    The first and last cycles of the record are bounded by the start and the
    end of the series instead of by a harvest, and are flagged as open.

    Parameters
    ----------
    df : pandas.DataFrame
        The observation table, used for the first and last date of each field.
    harvests : pandas.DataFrame
        Output of `detect_harvests`.

    Returns
    -------
    pandas.DataFrame
        One row per season: `field_id`, `season` (e.g. "2021/22"),
        `start_date` (previous harvest), `end_date` (this harvest),
        `duration_days` and `open_season` (True when either end of the
        interval is the limit of the record, not a detected harvest).
    """
    # The limits of the record close the first and the last cycle.
    limits = (
        df.groupby("field_id")["date"]
        .agg(["min", "max"])
        .reset_index()
        .melt(id_vars="field_id", value_name="harvest_date")
        .drop(columns="variable")
    )
    limits["is_limit"] = True

    events = pd.concat([harvests.assign(is_limit=False), limits], ignore_index=True)
    # A limit falling on a detected harvest is one event, and it stays a limit:
    # the record simply starts or ends on a bare field, so the cut date of that
    # cycle is unknown.
    events = events.groupby(["field_id", "harvest_date"], as_index=False).agg(
        is_limit=("is_limit", "max")
    )
    events = events.sort_values(["field_id", "harvest_date"]).reset_index(drop=True)

    events["start_date"] = events.groupby("field_id")["harvest_date"].shift(1)
    events["start_is_limit"] = events.groupby("field_id")["is_limit"].shift(1)

    seasons = events[events["start_date"].notna()].copy()
    seasons["end_date"] = seasons["harvest_date"]
    seasons["duration_days"] = (seasons["end_date"] - seasons["start_date"]).dt.days
    seasons["open_season"] = seasons["is_limit"] | seasons["start_is_limit"]

    # Season label of the Centre-South crop year, e.g. 2021 -> "2021/22".
    end_year = seasons["end_date"].dt.year
    seasons["season"] = end_year.astype(str) + "/" + (end_year + 1).astype(str).str[-2:]

    return seasons[
        [
            "field_id",
            "season",
            "start_date",
            "end_date",
            "duration_days",
            "open_season",
        ]
    ].reset_index(drop=True)
