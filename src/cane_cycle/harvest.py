"""Harvest date detection from Sentinel-2 time series.

The detection runs in three steps, in this order:

1. Cloud filter. Cloud brightens every band and can mimic a harvest in any
   index, so contaminated dates are removed before anything else is measured.
   B1 (coastal aerosol) is the separator: in the field spectral library cloud
   starts at 0.113 and no land class exceeds 0.087.
2. Low canopy. Dates with NDVI at or below `harvest_ndvi_max` have no live
   canopy. In the spectral library sugarcane never goes below 0.536 and
   straw and bare soil never exceed 0.212, so the threshold sits in that gap.
   Consecutive low dates form one harvest episode.
3. Peak SWIR. Inside each episode the harvest date is the date of maximum
   B11. Right after the cut the dry matter (straw, or exposed soil) is fully
   exposed and SWIR1 reflectance is at its highest; it then declines as the
   straw breaks down and the ratoon greens up.

Because the satellite only revisits the field every few days, the cut is also
reported as an interval: the last image still showing a canopy and the first
image already showing none.
"""

import pandas as pd

from cane_cycle.vegetation_indices import add_indices

# Step 1: observations with B1 (coastal aerosol) above this are cloud.
CLOUD_B1_MAX = 0.10

# Step 2: NDVI at or below this means no live canopy (harvest candidate).
HARVEST_NDVI_MAX = 0.30

# Step 3: band whose maximum inside the episode marks the harvest date.
HARVEST_PEAK_BAND = "B11"

# A change of state must be confirmed by this many consecutive observations.
MIN_RUN_OBS = 2

# A green run shorter than this is too short for sugarcane and is flagged as a
# rotation crop (soybean, peanut, sunn hemp grown before field renewal).
MIN_CANE_GREEN_DAYS = 180

# Two sugarcane harvests cannot be closer than this (one harvest per cycle).
MIN_CYCLE_DAYS = 240

# Labels used in the `kind` column of the output.
KIND_CANE = "cane"
KIND_ROTATION = "rotation?"
KIND_REPEAT = "repeat?"


def filter_clouds(df, cloud_b1_max=CLOUD_B1_MAX):
    """Drop cloud contaminated observations. Always the first filter.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with a B1 column.
    cloud_b1_max : float, default CLOUD_B1_MAX
        Observations with B1 above this are discarded.

    Returns
    -------
    pandas.DataFrame
        Only the cloud free observations.
    """
    return df[df["B1"] <= cloud_b1_max].copy()


def _absorb_short_runs(states, min_run_obs):
    """Merge runs shorter than `min_run_obs` into the neighbouring run.

    A single observation crossing the threshold is noise (thin cloud, cloud
    shadow, partial canopy), not a real change of surface. Flipping a short
    run into its neighbour only makes runs longer, so this always terminates.
    """
    states = list(states)
    while True:
        runs, start = [], 0
        for i in range(1, len(states) + 1):
            if i == len(states) or states[i] != states[start]:
                runs.append((start, i))
                start = i
        if len(runs) <= 1:
            return states

        short = next(((s, e) for s, e in runs if e - s < min_run_obs), None)
        if short is None:
            return states

        position = runs.index(short)
        neighbour = runs[position - 1] if position > 0 else runs[position + 1]
        for i in range(short[0], short[1]):
            states[i] = states[neighbour[0]]


def detect_harvests(
    df,
    cloud_b1_max=CLOUD_B1_MAX,
    harvest_ndvi_max=HARVEST_NDVI_MAX,
    peak_band=HARVEST_PEAK_BAND,
    min_run_obs=MIN_RUN_OBS,
    min_cane_green_days=MIN_CANE_GREEN_DAYS,
    min_cycle_days=MIN_CYCLE_DAYS,
):
    """Detect candidate harvest dates.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with `field_id`, `date` and Sentinel-2 band columns. NDVI is
        computed if it is not already a column.
    cloud_b1_max : float, default CLOUD_B1_MAX
        B1 threshold above which an observation is discarded as cloud.
    harvest_ndvi_max : float, default HARVEST_NDVI_MAX
        NDVI at or below this marks a date with no live canopy.
    peak_band : str, default HARVEST_PEAK_BAND
        Band whose maximum inside the episode gives the harvest date.
    min_run_obs : int, default MIN_RUN_OBS
        Consecutive observations needed to confirm a change of state.
    min_cane_green_days : int, default MIN_CANE_GREEN_DAYS
        Minimum length of the green run before an episode for it to be
        considered sugarcane; shorter ones are flagged as a rotation crop.
    min_cycle_days : int, default MIN_CYCLE_DAYS
        Minimum interval between two sugarcane harvests of the same field.

    Returns
    -------
    pandas.DataFrame
        One row per candidate harvest: `harvest_date` (date of maximum
        `peak_band` inside the episode, the point estimate), the bracketing
        interval `last_green_date` and `first_low_date` with its width in
        `uncertainty_days`, the peak reflectance `peak_value`, the length of
        the preceding green run `green_run_days`, and the label `kind`:
        "cane", "rotation?" when the green run is too short for cane, or
        "repeat?" when it follows another harvest too closely.
    """
    if "NDVI" not in df.columns:
        df = add_indices(df, indices=["NDVI"])

    # Step 1: cloud always goes first.
    df = filter_clouds(df, cloud_b1_max=cloud_b1_max)

    harvests = []
    for field_id, field_data in df.groupby("field_id"):
        field_data = field_data.sort_values("date").reset_index(drop=True)

        # Step 2: dates with no live canopy, grouped into episodes.
        state = field_data["NDVI"].le(harvest_ndvi_max).map({True: "low", False: "high"})
        field_data["state"] = _absorb_short_runs(state, min_run_obs)
        run_id = (field_data["state"] != field_data["state"].shift()).cumsum()
        runs = [group for _, group in field_data.groupby(run_id, sort=True)]

        last_cane_date = None
        for i in range(1, len(runs)):
            green, episode = runs[i - 1], runs[i]
            if green["state"].iloc[0] != "high" or episode["state"].iloc[0] != "low":
                continue

            # Step 3: inside the episode, the driest date is the harvest.
            peak_row = episode.loc[episode[peak_band].idxmax()]

            last_green_date = green["date"].max()
            first_low_date = episode["date"].min()
            green_run_days = (green["date"].max() - green["date"].min()).days
            harvest_date = peak_row["date"]

            if green_run_days < min_cane_green_days:
                kind = KIND_ROTATION
            elif (
                last_cane_date is not None
                and (harvest_date - last_cane_date).days < min_cycle_days
            ):
                kind = KIND_REPEAT
            else:
                kind = KIND_CANE
                last_cane_date = harvest_date

            harvests.append(
                {
                    "field_id": field_id,
                    "harvest_date": harvest_date,
                    "last_green_date": last_green_date,
                    "first_low_date": first_low_date,
                    "uncertainty_days": (first_low_date - last_green_date).days,
                    "peak_value": peak_row[peak_band],
                    "green_run_days": green_run_days,
                    "kind": kind,
                }
            )

    return (
        pd.DataFrame(harvests)
        .sort_values(["field_id", "harvest_date"])
        .reset_index(drop=True)
    )
