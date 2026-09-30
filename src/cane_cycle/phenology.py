"""Phenological phases of each crop cycle, read off the smoothed curves.

Three phases run from one harvest to the next:

===============  ===========================  =============================
Phase            Starts at                    What the crop is doing
===============  ===========================  =============================
`post_harvest`   the harvest                  cut; shoots coming up under
                                              the straw, not yet visible
`growth`         the ground stops being bare  canopy establishing, then
                                              stalks elongating
`maturation`     growth stops                 sugar accumulating, then
                                              waiting for the mill
===============  ===========================  =============================

`post_harvest` is the stretch the surface classification calls straw or bare
soil, the two merged into one. Sprouting happens inside it - the ratoon starts
within days of the cut - but under the trash blanket, where the satellite
cannot see it, so it gets no phase of its own.

The end of that stretch is read *backwards*, as the last bare image before
growth stops, not forwards as the first green one. In a normal cycle the two
agree, since the bare period is one unbroken run. In a renovation they do not:
the surface flickers for months while the field is worked, so reading forwards
ends the phase at the first flicker and leaves a year of fallow inside
`growth`. Read backwards, the fallow lands in `post_harvest`, which is what it
is.

`growth` and `maturation` are split at the peak of the smoothed index. The
peak is the one landmark inside a cycle that can be read from the curve alone,
with no threshold to calibrate and no derivative to estimate: before it the
canopy is still building, after it the index stabilises or falls and the crop
is filling with sugar.

The peak is taken on **EVI**, not NDVI. NDVI saturates at the top: measured on
these fields, it responds a third as much in its top decile as in its bottom,
so its own peak sits on a plateau rather than on a turn, and the NDVI curve
still climbs after the EVI peak in 9 of the 14 cycles - by 0.001 to 0.028,
which is at or below the fitting error of the smoother. EVI does not saturate
and turns first in 11 of the 14 cycles, by a median of 27 days and up to 92 on
the widest plateau; two cycles tie and one turns 19 days the other way, so the
lead is the rule and not a law. NDVI stays the curve the phases are drawn on,
since it carries the shape of the cycle.

These three phases are a satellite reading of the classical phases of the
crop, not a mapping of the eleven stages of the Brazilian scale (Barbosa,
2023), which need internodes counted by hand (CV1 to CV3) or Brix measured in
a lab (M1 to M3).

A cycle is left unclassified when it has no crop to have phenology: no bare
image before the peak, or landmarks out of order. A cycle that opens with the
canopy already closed peaks on its first day and falls here.
"""

import numpy as np
import pandas as pd

from cane_cycle.residue import SURFACE_SOIL, SURFACE_STRAW
from cane_cycle.smoothing import smooth_cycles_v2

# Index the phases are drawn on: it carries the shape of the cycle.
SHAPE_INDEX = "NDVI"

# Index read at the top of the curve, where the first one saturates.
GROWTH_INDEX = "EVI"

# Phase names, in the order they occur.
PHASES = ["post_harvest", "growth", "maturation"]


def phase_landmarks(df, seasons, growth_index=GROWTH_INDEX):
    """Find the two dates that split each cycle into three phases.

    Parameters
    ----------
    df : pandas.DataFrame
        Observations with `surface`, from `cane_cycle.residue.classify_surface`.
    seasons : pandas.DataFrame
        Cycles from `cane_cycle.harvest.build_seasons`.
    growth_index : str, default GROWTH_INDEX
        Index whose peak marks the end of growth.

    Returns
    -------
    pandas.DataFrame
        One row per cycle with `cut`, `bare_end`, `cessation`, `next_cut`, and
        `classified`, which is False for a cycle with no bare image before the
        peak, or whose landmarks came out of order.
    """
    growth = smooth_cycles_v2(df, seasons, index=growth_index)

    rows = []
    for cycle in seasons.itertuples():
        peak = growth[
            (growth["field_id"] == cycle.field_id) & (growth["season"] == cycle.season)
        ].sort_values("date")
        if peak.empty:
            continue
        cessation = peak["date"].to_numpy()[
            int(np.argmax(peak[f"{growth_index}_smooth"]))
        ]

        # The bare period ends at the last bare image before growth stops.
        bare = df[
            (df["field_id"] == cycle.field_id)
            & df["date"].between(cycle.start_date, cycle.end_date)
            & df["surface"].isin([SURFACE_STRAW, SURFACE_SOIL])
            & (df["date"] < cessation)
        ]
        bare_end = bare["date"].max() if len(bare) else None

        marks = [cycle.start_date, bare_end, cessation, cycle.end_date]
        classified = all(m is not None for m in marks) and all(
            pd.Timestamp(a) <= pd.Timestamp(b) for a, b in zip(marks, marks[1:])
        )
        rows.append(
            {
                "field_id": cycle.field_id,
                "season": cycle.season,
                "cut": cycle.start_date,
                "bare_end": bare_end,
                "cessation": cessation,
                "next_cut": cycle.end_date,
                "classified": classified,
            }
        )
    return pd.DataFrame(rows)


def add_phenology(df, seasons, shape_index=SHAPE_INDEX, growth_index=GROWTH_INDEX):
    """Return the daily smoothed curve with the phase of every day.

    Returns
    -------
    tuple of pandas.DataFrame
        The daily curve with a `phase` column (None on an unclassified
        cycle), and the landmark table from `phase_landmarks`.
    """
    curve = smooth_cycles_v2(df, seasons, index=shape_index)
    landmarks = phase_landmarks(df, seasons, growth_index)

    curve = curve.merge(landmarks, on=["field_id", "season"], how="left")
    boundaries = ["bare_end", "cessation"]
    # Each day takes the last phase whose start it has reached.
    phase = pd.Series(PHASES[0], index=curve.index, dtype="object")
    for name, boundary in zip(PHASES[1:], boundaries):
        phase = phase.mask(curve["date"] >= curve[boundary], name)
    curve["phase"] = phase.where(curve["classified"].fillna(False))

    return curve.drop(columns=["cut", "next_cut"]), landmarks


def phase_durations(landmarks):
    """Return the length of each phase, in days, for every classified cycle."""
    marks = ["cut", "bare_end", "cessation", "next_cut"]
    out = landmarks[landmarks["classified"]][["field_id", "season"]].copy()
    for name, start, end in zip(PHASES, marks, marks[1:]):
        out[name] = (
            landmarks.loc[landmarks["classified"], end]
            - landmarks.loc[landmarks["classified"], start]
        ).dt.days
    return out
