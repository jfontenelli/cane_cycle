"""Crop residue (straw) cover on the ground, and the surface state of a field.

Follows the method Embrapa adopted for the Zarc Niveis de Manejo (DOCUMENTOS
454, Embrapa Soja, 2023), which in turn follows Beeson et al. (2020) and
Hively et al. (2018): NDTI is stretched linearly onto a 0-100% residue cover
scale, with the two ends taken from the data itself rather than from field
measurements.

    cover% = slope * NDTI + intercept
    slope     = (100 - 0) / (NDTI_max - NDTI_min)
    intercept = -slope * NDTI_min

NDTI_min and NDTI_max are the mean minus and plus `sigma` standard
deviations, as the Embrapa document specifies, which keeps a single noisy
observation from setting the whole scale. The bare end is assumed to be 0%
cover and the covered end 100%, "o que nao requer dados de campo".

Two deliberate departures from the published protocol:

* Embrapa takes the two ends from the spatial spread of one image, over
  thousands of pixels. This data is already averaged per field, so the ends
  come from the field's own time series instead. The assumption is the same -
  that the extremes really are bare soil and full residue - and here it can be
  checked against the spectral library, which it passes.
* The stretch is fitted only over the dates with no canopy. Standing cane has
  the highest NDTI of all (0.305 to 0.389 in the library, against 0.158 to
  0.193 for straw), so including it would put the 100% anchor on green cane.
"""

import numpy as np
import pandas as pd

from cane_cycle.harvest import (
    CLOUD_B1_MAX,
    CLOUD_NDMI_MIN,
    flag_harvest_conditions,
)

# Standard deviations that set the ends of the stretch (Embrapa DOC-454).
RESIDUE_SIGMA = 3

# Residue cover assumed at each end of the stretch (Embrapa DOC-454).
COVER_MIN = 0.0
COVER_MAX = 100.0

# Cover classes of the Zarc Niveis de Manejo, indicator 2 (Embrapa DOC-454).
ZARC_COVER_CLASSES = (45, 60, 75)

# Cover at or above which the surface counts as straw rather than bare soil.
# This is the lowest Zarc class boundary, and on these two fields it falls in
# the gap between the normal harvests (54.6% to 69.9%) and the two that open a
# renewal (27.2% and 35.7%).
STRAW_COVER_MIN = 45.0

# Days after a harvest over which the residue cover is summarised. Not from
# the Embrapa protocol, which reads a single image at sowing time; 30 days
# keeps most of the contrast while leaving at least four images per harvest.
RESIDUE_WINDOW_DAYS = 30

# Labels of the `surface` column.
SURFACE_CANOPY = "canopy"
SURFACE_STRAW = "straw"
SURFACE_SOIL = "bare soil"
SURFACE_CLOUD = "cloud"


def residue_stretch(ndti, sigma=RESIDUE_SIGMA):
    """Return the (slope, intercept) that map NDTI onto residue cover percent.

    Parameters
    ----------
    ndti : pandas.Series
        NDTI of the observations with no canopy.
    sigma : float, default RESIDUE_SIGMA
        Standard deviations from the mean that define the two ends.

    Returns
    -------
    tuple of float
        Slope and intercept of `cover% = slope * NDTI + intercept`.
    """
    low = ndti.mean() - sigma * ndti.std()
    high = ndti.mean() + sigma * ndti.std()
    slope = (COVER_MAX - COVER_MIN) / (high - low)
    return slope, -slope * low


def add_residue_cover(df, sigma=RESIDUE_SIGMA):
    """Return a copy of df with the residue cover of every bare date.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with band columns; NDVI and NDTI are computed if missing.
    sigma : float, default RESIDUE_SIGMA
        Standard deviations that set the ends of the stretch.

    Returns
    -------
    pandas.DataFrame
        Copy of df with `residue_cover_pct`, filled only on the dates with no
        canopy and left as NaN on the others, where the number would be
        meaningless.
    """
    df = flag_harvest_conditions(df)
    bare = (df["swir_above_nir"] == 1) & (df["low_ndvi"] == 1)

    slope, intercept = residue_stretch(df.loc[bare, "NDTI"], sigma=sigma)
    df["residue_cover_pct"] = np.nan
    df.loc[bare, "residue_cover_pct"] = (
        slope * df.loc[bare, "NDTI"] + intercept
    ).clip(COVER_MIN, COVER_MAX)
    return df


def classify_surface(
    df,
    straw_cover_min=STRAW_COVER_MIN,
    sigma=RESIDUE_SIGMA,
    cloud_b1_max=CLOUD_B1_MAX,
    cloud_ndmi_min=CLOUD_NDMI_MIN,
):
    """Return a copy of df with the surface state of every date.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with band columns; residue cover is computed if missing.
    straw_cover_min : float, default STRAW_COVER_MIN
        Cover at or above which a bare date counts as straw.
    sigma : float, default RESIDUE_SIGMA
        Standard deviations that set the ends of the stretch.
    cloud_b1_max, cloud_ndmi_min : float
        Thresholds of the cloud test, bright and wet.

    Returns
    -------
    pandas.DataFrame
        Copy of df with `surface`: "cloud" where the surface cannot be seen,
        "canopy" while the crop covers the ground, then "straw" or
        "bare soil" once it does not.

    Notes
    -----
    The cloud label is tested last but wins, because "canopy" is otherwise
    defined by exclusion and would collect every date the two harvest
    conditions reject, cloud included. Cloud never reaches the straw or bare
    soil labels: the harvest conditions already exclude it, since cloud
    reflects more in the NIR than in the SWIR, exactly like a live canopy.

    The straw / bare soil split cuts through a continuum, not a natural
    boundary: about half of the bare dates sit between the pure endmembers of
    the spectral library, so `residue_cover_pct` carries more information than
    this label does.
    """
    if "residue_cover_pct" not in df.columns:
        df = add_residue_cover(df, sigma=sigma)
    else:
        df = df.copy()

    covered = df["residue_cover_pct"] >= straw_cover_min
    surface = np.where(
        df["residue_cover_pct"].isna(),
        SURFACE_CANOPY,
        np.where(covered, SURFACE_STRAW, SURFACE_SOIL),
    )
    is_cloud = (df["B1"] > cloud_b1_max) & (df["NDMI"] > cloud_ndmi_min)
    df["surface"] = np.where(is_cloud, SURFACE_CLOUD, surface)
    return df


def summarize_residue(df, harvests, window_days=RESIDUE_WINDOW_DAYS):
    """Summarise the residue left on the ground by each harvest.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with `residue_cover_pct`, from `add_residue_cover`.
    harvests : pandas.DataFrame
        Output of `cane_cycle.harvest.detect_harvests`.
    window_days : int, default RESIDUE_WINDOW_DAYS
        Days after the harvest over which the cover is summarised.

    Returns
    -------
    pandas.DataFrame
        One row per harvest with `n_images`, the median `residue_cover_pct`
        over the window, and the `zarc_class` that median falls in.
    """
    if "residue_cover_pct" not in df.columns:
        df = add_residue_cover(df)

    rows = []
    for harvest in harvests.itertuples():
        window = df[
            (df["field_id"] == harvest.field_id)
            & df["date"].between(
                harvest.harvest_date,
                harvest.harvest_date + pd.Timedelta(days=window_days),
            )
            & df["residue_cover_pct"].notna()
        ]
        cover = window["residue_cover_pct"].median()
        rows.append(
            {
                "field_id": harvest.field_id,
                "season": harvest.season,
                "harvest_date": harvest.harvest_date,
                "n_images": len(window),
                "residue_cover_pct": cover,
                "zarc_class": _zarc_class(cover),
            }
        )
    return pd.DataFrame(rows)


def _zarc_class(cover):
    """Name the Zarc cover class a percentage falls in."""
    if pd.isna(cover):
        return None
    low, mid, high = ZARC_COVER_CLASSES
    if cover < low:
        return f"<{low}%"
    if cover < mid:
        return f"{low}-{mid}%"
    if cover < high:
        return f"{mid}-{high}%"
    return f">{high}%"
