"""Harvest date detection from Sentinel-2 spectral state transitions.

The method follows the signatures measured in the field spectral library:
a live sugarcane canopy peaks in the narrow NIR (B8A), while straw and bare
soil both peak in SWIR1 (B11). A harvest is therefore the first date on which
the spectral peak moves from NIR to SWIR, i.e. the canopy is gone.

Straw and bare soil are deliberately treated as a single "dry" state here.
Telling a harvest apart from field renewal (reforma) is a later step, and it
needs the dry state to be detected first.

Because the satellite only sees the field every few days, a harvest is
recorded as an interval - the last image still showing a canopy and the first
image already showing a dry surface - plus its midpoint as the point estimate.
"""

import pandas as pd

from cane_cycle.vegetation_indices import add_peak_band

# Observations with B1 (coastal aerosol) above this are treated as cloud.
# In the spectral library cloud starts at 0.113 and no land class exceeds 0.087.
CLOUD_B1_MAX = 0.10

# Peak bands that mean a live canopy and a dry surface (straw or bare soil).
GREEN_PEAK_BANDS = ("B8", "B8A")
DRY_PEAK_BANDS = ("B11", "B12")

# A state change must be confirmed by at least this many consecutive
# observations; shorter runs are treated as noise and absorbed.
MIN_RUN_OBS = 2

# A green run shorter than this is too short for sugarcane and is flagged as a
# rotation crop (soybean, peanut, sunn hemp grown before field renewal).
# Observed sugarcane green runs last 220-300 days; rotation crops 50-120 days.
MIN_CANE_GREEN_DAYS = 180

# Two sugarcane harvests cannot be closer than this (one harvest per cycle).
MIN_CYCLE_DAYS = 240

# Labels used in the `kind` column of the output.
KIND_CANE = "cane"
KIND_ROTATION = "rotation?"
KIND_REPEAT = "repeat?"


def classify_state(df, cloud_b1_max=CLOUD_B1_MAX):
    """Return the surface state of each observation.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with `peak_band` and a B1 column.
    cloud_b1_max : float, default CLOUD_B1_MAX
        Observations with B1 above this are labelled "cloud".

    Returns
    -------
    pandas.Series
        One of "cloud", "green" (live canopy) or "dry" (straw or bare soil).
    """
    state = pd.Series("other", index=df.index, dtype="object")
    state[df["peak_band"].isin(GREEN_PEAK_BANDS)] = "green"
    state[df["peak_band"].isin(DRY_PEAK_BANDS)] = "dry"
    state[df["B1"] > cloud_b1_max] = "cloud"
    return state


def _absorb_short_runs(states, min_run_obs):
    """Merge runs shorter than `min_run_obs` into the neighbouring run.

    A single observation flipping state is noise (thin cloud, cloud shadow,
    partial canopy), not a real change of surface. Flipping a short run into
    its neighbour can only make runs longer, so this always terminates.
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


def _field_runs(field_data):
    """Group consecutive observations of the same state into runs."""
    run_id = (field_data["state"] != field_data["state"].shift()).cumsum()
    runs = field_data.groupby(run_id).agg(
        state=("state", "first"),
        start=("date", "min"),
        end=("date", "max"),
        n_obs=("date", "size"),
    )
    runs["days"] = (runs["end"] - runs["start"]).dt.days
    return runs.reset_index(drop=True)


def detect_harvests(
    df,
    cloud_b1_max=CLOUD_B1_MAX,
    min_run_obs=MIN_RUN_OBS,
    min_cane_green_days=MIN_CANE_GREEN_DAYS,
    min_cycle_days=MIN_CYCLE_DAYS,
):
    """Detect candidate harvest dates from green-to-dry transitions.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with `field_id`, `date` and Sentinel-2 band columns.
    cloud_b1_max : float, default CLOUD_B1_MAX
        B1 threshold above which an observation is discarded as cloud.
    min_run_obs : int, default MIN_RUN_OBS
        Consecutive observations needed to confirm a change of state.
    min_cane_green_days : int, default MIN_CANE_GREEN_DAYS
        Minimum length of the green run before a transition for it to be
        considered sugarcane; shorter ones are flagged as a rotation crop.
    min_cycle_days : int, default MIN_CYCLE_DAYS
        Minimum interval between two sugarcane harvests of the same field.

    Returns
    -------
    pandas.DataFrame
        One row per candidate harvest, with the bracketing interval
        (`last_green_date`, `first_dry_date`), the point estimate
        (`mid_date`), the width of the interval in days
        (`uncertainty_days`), the length of the preceding green run
        (`green_run_days`) and the label `kind`: "cane" for a sugarcane
        harvest, "rotation?" when the green run is too short for cane, and
        "repeat?" when it follows another harvest too closely.
    """
    if "peak_band" not in df.columns:
        df = add_peak_band(df)

    df = df.copy()
    df["state"] = classify_state(df, cloud_b1_max=cloud_b1_max)
    # Cloudy dates carry no surface information: they must not break a run.
    df = df[df["state"].isin(["green", "dry"])]

    harvests = []
    for field_id, field_data in df.groupby("field_id"):
        field_data = field_data.sort_values("date").reset_index(drop=True)
        field_data["state"] = _absorb_short_runs(field_data["state"], min_run_obs)
        runs = _field_runs(field_data)

        last_cane_date = None
        for i in range(1, len(runs)):
            green, dry = runs.loc[i - 1], runs.loc[i]
            if green["state"] != "green" or dry["state"] != "dry":
                continue

            # The cut happened between the last canopy image and the first dry one.
            last_green_date, first_dry_date = green["end"], dry["start"]
            mid_date = last_green_date + (first_dry_date - last_green_date) / 2

            if green["days"] < min_cane_green_days:
                kind = KIND_ROTATION
            elif (
                last_cane_date is not None
                and (mid_date - last_cane_date).days < min_cycle_days
            ):
                kind = KIND_REPEAT
            else:
                kind = KIND_CANE
                last_cane_date = mid_date

            harvests.append(
                {
                    "field_id": field_id,
                    "last_green_date": last_green_date,
                    "first_dry_date": first_dry_date,
                    "mid_date": mid_date,
                    "uncertainty_days": (first_dry_date - last_green_date).days,
                    "green_run_days": green["days"],
                    "kind": kind,
                }
            )

    return pd.DataFrame(harvests).sort_values(["field_id", "mid_date"]).reset_index(
        drop=True
    )
