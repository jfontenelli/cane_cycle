"""Harvest date detection and season (safra) boundaries.

A crop cycle is defined by the crop itself, never by the calendar:

    canopy closes (NDVI >= canopy_ndvi_min)
        -> canopy is lost (B11 > B8A and NDVI <= harvest_ndvi_max)
        = one cycle, and its harvest is that first bare date.

Grouping by calendar year (or by crop year) cannot do this. A field in
renewal stays bare across any year boundary, so the first image of the new
year looks like a fresh harvest; and a soybean plus maize rotation has two
complete cycles inside one year, of which a yearly grouping would only ever
report the first. Requiring a closed canopy since the previous harvest solves
both: there is no harvest without a crop to harvest.

A harvest date must meet two spectral conditions at the same time:

1. `swir_above_nir`: B11 above B8A. A live canopy reflects more in the NIR
   than in the SWIR; straw and bare soil do the opposite. In the field
   spectral library every sugarcane sample has B8A above B11 and every straw
   and bare soil sample has the reverse, so the comparison marks the loss of
   the canopy without caring how bright the surface is. Cloud also has B8A
   above B11 in all library samples, so this condition rejects cloud too.
2. `low_ndvi`: NDVI at or below `harvest_ndvi_max`. Sugarcane never goes below
   0.536 in the library, straw and bare soil never exceed 0.212, so the
   default threshold of 0.30 sits inside that gap.

A single image is enough. Clouds often leave only one clear view of a freshly
cut field, and the two conditions above already exclude cloud, so demanding a
second bare image would only push the date later.

A season (safra) then runs from one harvest to the next. The first and last
cycles of the record are open - they are bounded by the start and the end of
the series rather than by a harvest - and they are kept, flagged as open.
The crop year is only a label on the result, never a grouping criterion.
"""

import pandas as pd

from cane_cycle.vegetation_indices import add_indices

# Observations with B1 (coastal aerosol) above this are treated as cloud.
CLOUD_B1_MAX = 0.10

# NDVI at or below this means no live canopy (harvest candidate).
HARVEST_NDVI_MAX = 0.30

# NDVI a crop must reach for the field to count as having a canopy to harvest.
# Weak ratoons in these fields peak at 0.546, while the bare renewal period
# peaked at 0.452, so the threshold sits inside that gap.
CANOPY_NDVI_MIN = 0.50

# First month of the Centre-South crop year (April to March).
SEASON_START_MONTH = 4


def season_year(dates, season_start_month=SEASON_START_MONTH):
    """Return the crop year of each date: the year its season started.

    This only labels the result; cycles are separated by the canopy, not here.
    """
    return dates.dt.year.where(dates.dt.month >= season_start_month, dates.dt.year - 1)


def season_label(years):
    """Format crop years as the usual season label, e.g. 2021 -> "2021/22"."""
    return years.astype(str) + "/" + (years + 1).astype(str).str[-2:]


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


def _field_harvests(field_data, canopy_ndvi_min):
    """Walk one field in time order, closing a cycle at each canopy loss.

    The canopy flag arms when the crop reaches `canopy_ndvi_min` and is spent
    on the next bare date, which becomes that cycle's harvest. A field that
    never regrew therefore cannot produce a second harvest, however many bare
    images it shows.
    """
    harvests = []
    canopy_seen = False
    peak_ndvi = 0.0

    for row in field_data.itertuples():
        peak_ndvi = max(peak_ndvi, row.NDVI)
        if row.NDVI >= canopy_ndvi_min:
            canopy_seen = True
        elif canopy_seen and row.swir_above_nir and row.low_ndvi:
            harvests.append(
                {
                    "field_id": row.field_id,
                    "harvest_date": row.date,
                    "ndvi_at_harvest": row.NDVI,
                    "cycle_peak_ndvi": peak_ndvi,
                }
            )
            canopy_seen = False
            peak_ndvi = 0.0

    return harvests


def detect_harvests(
    df,
    ndvi_max=HARVEST_NDVI_MAX,
    canopy_ndvi_min=CANOPY_NDVI_MIN,
    cloud_b1_max=CLOUD_B1_MAX,
):
    """Detect the harvest date of every crop cycle.

    A cycle is closed by the first date with no canopy that follows a date
    with a closed canopy, so a field can have two harvests in one year
    (soybean and maize) or none at all (renewal), and a single clear image is
    enough to date the cut.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with `field_id`, `date` and Sentinel-2 band columns. NDVI is
        computed if it is not already a column.
    ndvi_max : float, default HARVEST_NDVI_MAX
        NDVI at or below this marks a date with no live canopy.
    canopy_ndvi_min : float, default CANOPY_NDVI_MIN
        NDVI the crop must reach before a harvest can be recorded.
    cloud_b1_max : float, default CLOUD_B1_MAX
        B1 threshold above which an observation is discarded as cloud.

    Returns
    -------
    pandas.DataFrame
        One row per harvest: `field_id`, `season` (label of the crop year it
        falls in), `harvest_date`, `ndvi_at_harvest` and `cycle_peak_ndvi`,
        the highest NDVI the crop reached during that cycle.
    """
    if "NDVI" not in df.columns:
        df = add_indices(df, indices=["NDVI"])

    # Cloud always goes first.
    df = filter_clouds(df, cloud_b1_max=cloud_b1_max)
    df = flag_harvest_conditions(df, ndvi_max=ndvi_max)

    harvests = []
    for _, field_data in df.groupby("field_id"):
        field_data = field_data.sort_values("date")
        harvests += _field_harvests(field_data, canopy_ndvi_min)

    harvests = pd.DataFrame(harvests)
    harvests.insert(1, "season", season_label(season_year(harvests["harvest_date"])))
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

    # The season is named after the crop year in which it was harvested.
    seasons["season"] = season_label(season_year(seasons["end_date"]))

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
