"""Phenological phases of each crop cycle, read off the smoothed curves.

Five phases run from one harvest to the next:

===============  ===========================  =============================
Phase            Starts at                    What the crop is doing
===============  ===========================  =============================
`post_harvest`   the harvest                  cut; shoots coming up under
                                              the straw, not yet visible
`tillering`      the ground stops being bare  shoots multiplying, covering
`grand_growth`   the canopy closes            stalks elongating
`maturation`     growth stops                 sugar accumulating, then
                                              waiting for the mill
===============  ===========================  =============================

`post_harvest` is the stretch the surface classification calls straw or bare
soil, the two merged into one. Sprouting happens inside it - the ratoon starts
within days of the cut - but under the trash blanket, where the satellite
cannot see it, so it gets no phase of its own.

The end of that stretch is read *backwards*, as the last bare image before the
canopy closes, not forwards as the first green one. In a normal cycle the two
agree, since the bare period is one unbroken run. In a renovation they do not: the
surface flickers for months while the field is worked, so reading forwards ends
the phase at the first flicker and leaves a year of fallow inside `tillering`.
Read backwards, the fallow lands in `post_harvest`, which is what it is.

Only one of the eleven stages of the Brazilian scale (Barbosa, 2023) can be
seen from orbit, and it is the one this uses: P3, "final do perfilhamento",
defined as *fechamento da entrelinha* - the canopy closing over the inter-row.
The others need internodes counted by hand (CV1 to CV3) or Brix measured in a
lab (M1 to M3). So these five phases are a satellite reading of the four
classical phases, not a mapping of the eleven.

Two indices are used, each where it is trustworthy:

* **NDVI** for the trough and for canopy closure. Both are read on the steep
  part of the curve, where NDVI does not saturate, and its closure threshold
  is the one anchored to the spectral library.
* **EVI** for the end of growth. NDVI saturates at the top: measured on these
  fields, it responds a third as much in its top decile as in its bottom, so
  its peak drifts to the middle of a plateau that can be 105 days wide. EVI
  does not saturate and turns earlier, and by an amount proportional to the
  plateau's width - 0 days when there is no plateau, 99 days on the widest.
  That proportionality is what saturation predicts and atmospheric noise
  would not.

A cycle is left unclassified when it has no crop to have phenology. A field in
renovation stays bare for a year, and forcing phases on it would report 447 days
of "sprouting".
"""

import numpy as np
import pandas as pd

from cane_cycle.residue import SURFACE_SOIL, SURFACE_STRAW
from cane_cycle.smoothing import smooth_cycles_v2

# Index read on the steep parts of the curve, and where its canopy closes.
SHAPE_INDEX = "NDVI"
CLOSURE_THRESHOLD = 0.50

# Index read at the top of the curve, where the first one saturates.
GROWTH_INDEX = "EVI"

# Phase names, in the order they occur.
PHASES = ["post_harvest", "tillering", "grand_growth", "maturation"]


def phase_landmarks(
    df,
    seasons,
    shape_index=SHAPE_INDEX,
    growth_index=GROWTH_INDEX,
    closure_threshold=CLOSURE_THRESHOLD,
):
    """Find the three dates that split each cycle into four phases.

    Parameters
    ----------
    df : pandas.DataFrame
        Observations with `surface`, from `cane_cycle.residue.classify_surface`.
    seasons : pandas.DataFrame
        Cycles from `cane_cycle.harvest.build_seasons`.
    shape_index : str, default SHAPE_INDEX
        Index used for the trough and for canopy closure.
    growth_index : str, default GROWTH_INDEX
        Index whose peak marks the end of growth.
    closure_threshold : float, default CLOSURE_THRESHOLD
        Value of `shape_index` at which the canopy counts as closed.

    Returns
    -------
    pandas.DataFrame
        One row per cycle with `cut`, `bare_end`, `closure`, `cessation`,
        `next_cut`, and `classified`, which is False for a cycle whose canopy
        never closed or whose landmarks came out of order.
    """
    shape = smooth_cycles_v2(df, seasons, index=shape_index)
    growth = smooth_cycles_v2(df, seasons, index=growth_index)

    rows = []
    for cycle in seasons.itertuples():
        curve = shape[
            (shape["field_id"] == cycle.field_id) & (shape["season"] == cycle.season)
        ].sort_values("date")
        if curve.empty:
            continue

        days = curve["date"].to_numpy()
        values = curve[f"{shape_index}_smooth"].to_numpy()

        reached = np.flatnonzero(values >= closure_threshold)
        closure = days[reached[0]] if len(reached) else None

        # The bare period ends at the last bare image before the canopy closes.
        bare = df[
            (df["field_id"] == cycle.field_id)
            & df["date"].between(cycle.start_date, cycle.end_date)
            & df["surface"].isin([SURFACE_STRAW, SURFACE_SOIL])
        ]
        if closure is not None:
            bare = bare[bare["date"] < closure]
        bare_end = bare["date"].max() if len(bare) else None

        peak = growth[
            (growth["field_id"] == cycle.field_id) & (growth["season"] == cycle.season)
        ].sort_values("date")
        cessation = (
            peak["date"].to_numpy()[int(np.argmax(peak[f"{growth_index}_smooth"]))]
            if not peak.empty
            else None
        )

        marks = [cycle.start_date, bare_end, closure, cessation, cycle.end_date]
        classified = all(m is not None for m in marks) and all(
            pd.Timestamp(a) <= pd.Timestamp(b) for a, b in zip(marks, marks[1:])
        )
        rows.append(
            {
                "field_id": cycle.field_id,
                "season": cycle.season,
                "cut": cycle.start_date,
                "bare_end": bare_end,
                "closure": closure,
                "cessation": cessation,
                "next_cut": cycle.end_date,
                "classified": classified,
            }
        )
    return pd.DataFrame(rows)


def add_phenology(
    df,
    seasons,
    shape_index=SHAPE_INDEX,
    growth_index=GROWTH_INDEX,
    closure_threshold=CLOSURE_THRESHOLD,
):
    """Return the daily smoothed curve with the phase of every day.

    Returns
    -------
    tuple of pandas.DataFrame
        The daily curve with a `phase` column (None on an unclassified
        cycle), and the landmark table from `phase_landmarks`.
    """
    curve = smooth_cycles_v2(df, seasons, index=shape_index)
    landmarks = phase_landmarks(
        df, seasons, shape_index, growth_index, closure_threshold
    )

    curve = curve.merge(landmarks, on=["field_id", "season"], how="left")
    boundaries = ["bare_end", "closure", "cessation"]
    # Each day takes the last phase whose start it has reached.
    phase = pd.Series(PHASES[0], index=curve.index, dtype="object")
    for name, boundary in zip(PHASES[1:], boundaries):
        phase = phase.mask(curve["date"] >= curve[boundary], name)
    curve["phase"] = phase.where(curve["classified"].fillna(False))

    return curve.drop(columns=["cut", "next_cut"]), landmarks


def phase_durations(landmarks):
    """Return the length of each phase, in days, for every classified cycle."""
    marks = ["cut", "bare_end", "closure", "cessation", "next_cut"]
    out = landmarks[landmarks["classified"]][["field_id", "season"]].copy()
    for name, start, end in zip(PHASES, marks, marks[1:]):
        out[name] = (
            landmarks.loc[landmarks["classified"], end]
            - landmarks.loc[landmarks["classified"], start]
        ).dt.days
    return out
