"""Whittaker smoothing of an index series, on a daily grid, cycle by cycle.

The phenological phases are defined by where the curve turns, so they need a
derivative, and the derivative of a raw satellite series is noise. This module
produces the smooth daily curve the phases are read from.

The smoother is the Whittaker penalised least squares (Eilers, 2003): it finds
the curve `z` that minimises

    sum(w * (y - z)^2)  +  lambda * sum(second difference of z, squared)

The first term keeps the curve near the observations, weighted by `w`; the
second penalises roughness. `lambda` sets the balance. Measured on these two
fields, lambda = 10 leaves the first derivative changing sign up to 38 times
per cycle, which is unusable, while lambda = 10000 doubles the fitting error;
1000 sits between, with a fitting error of 0.02 to 0.04 (the order of the
sensor noise) and 3 to 12 inflections per cycle.

Two choices matter as much as lambda:

* **One cycle at a time.** A harvest is a step, and a smoother asked to cross
  it would round it off. Because the harvest dates are already known, each
  cycle is smoothed on its own and the cut is a boundary of the domain rather
  than a feature to be fitted.
* **A daily grid, with weights.** Observations are 5 to 50 days apart, so the
  curve is solved on every day of the cycle and the days without a usable
  image simply carry weight zero. Nothing is interpolated beforehand: the
  smoother fills those days itself, as part of the same fit.

Whittaker was chosen over Savitzky-Golay because it takes the irregular
sampling natively, through the weights, and because it penalises exactly the
second difference, which is what makes the derivative usable.
"""

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Roughness penalty. See the module docstring for how it was chosen.
WHITTAKER_LAMBDA = 1000.0

# Weight of a cloud contaminated observation. Zero drops it; a small positive
# value keeps it as a weak constraint, which can help across long summer gaps.
CLOUD_WEIGHT = 0.0

# Weight of a clear observation.
CLEAR_WEIGHT = 1.0

# A cycle needs at least this many usable observations to be smoothed.
MIN_OBSERVATIONS = 10

# Extra weight on the first and last observation of a cycle. Those two are the
# harvests that bound it, and a curve as stiff as lambda = 1000 cannot turn
# fast enough to reach them on its own: without the anchor the fit ends 0.047
# above the observed value at the cut and consecutive cycles disagree by 0.065
# on the date they share. At 10 both fall to about 0.01, and the overall
# fitting error moves from 0.0334 to 0.0338.
BOUNDARY_ANCHOR = 10.0


def whittaker(values, weights, lam=WHITTAKER_LAMBDA):
    """Smooth `values` by penalised least squares.

    Parameters
    ----------
    values : numpy.ndarray
        The series to smooth, one entry per day. Days without an observation
        may hold any finite number, since their weight is zero.
    weights : numpy.ndarray
        Weight of each day: 0 for a day with no usable image.
    lam : float, default WHITTAKER_LAMBDA
        Roughness penalty. Larger is smoother and less faithful.

    Returns
    -------
    numpy.ndarray
        The smoothed series, same length as `values`.
    """
    n = len(values)
    if n < 3:
        return np.asarray(values, dtype=float)

    # Second difference operator: rows of [1, -2, 1].
    second_difference = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    left = sparse.diags(weights) + lam * second_difference.T @ second_difference
    return spsolve(left.tocsc(), weights * np.nan_to_num(values))


def smooth_cycles(
    df,
    seasons,
    index="NDVI",
    lam=WHITTAKER_LAMBDA,
    cloud_weight=CLOUD_WEIGHT,
    min_observations=MIN_OBSERVATIONS,
):
    """Smooth one index over each crop cycle, on a daily grid.

    Parameters
    ----------
    df : pandas.DataFrame
        Observations with `field_id`, `date`, the index column and `surface`
        from `cane_cycle.residue.classify_surface`.
    seasons : pandas.DataFrame
        Cycles from `cane_cycle.harvest.build_seasons`.
    index : str, default "NDVI"
        Index to smooth. The same call with "EVI" gives the other reading:
        NDVI saturates near the peak and EVI does not, so the two disagree on
        where growth stops, by as much as 99 days on a wide plateau.
    lam : float, default WHITTAKER_LAMBDA
        Roughness penalty.
    cloud_weight : float, default CLOUD_WEIGHT
        Weight given to cloud contaminated observations.
    min_observations : int, default MIN_OBSERVATIONS
        Cycles with fewer usable observations are skipped.

    Returns
    -------
    pandas.DataFrame
        One row per day of each cycle, with `field_id`, `season`, `date`,
        `<index>_smooth`, `<index>_slope` (first derivative, per day),
        `<index>_curvature` (second derivative) and `observed`, which marks
        the days that had a real image behind them.
    """
    return _smooth(
        df, seasons, index, lam, cloud_weight, min_observations,
        fit=lambda v, w: whittaker(v, w, lam=lam),
    )


def _smooth(df, seasons, index, lam, cloud_weight, min_observations, fit):
    """Run `fit` over every cycle and return the daily curve and derivatives."""
    smooth_col = f"{index}_smooth"
    slope_col = f"{index}_slope"
    curvature_col = f"{index}_curvature"

    frames = []
    for cycle in seasons.itertuples():
        observations = df[
            (df["field_id"] == cycle.field_id)
            & df["date"].between(cycle.start_date, cycle.end_date)
            & df[index].notna()
        ]
        if (observations["surface"] != "cloud").sum() < min_observations:
            continue

        days = pd.date_range(cycle.start_date, cycle.end_date, freq="D")
        values = pd.Series(np.nan, index=days)
        weights = pd.Series(0.0, index=days)
        values.loc[observations["date"].values] = observations[index].values
        weights.loc[observations["date"].values] = np.where(
            observations["surface"].values == "cloud", cloud_weight, CLEAR_WEIGHT
        )
        # Pin the curve to the harvests that bound the cycle.
        observed_days = np.flatnonzero(weights.to_numpy() > 0)
        if len(observed_days):
            weights.iloc[observed_days[0]] *= BOUNDARY_ANCHOR
            weights.iloc[observed_days[-1]] *= BOUNDARY_ANCHOR

        smooth = fit(np.nan_to_num(values.to_numpy()), weights.to_numpy())
        slope = np.gradient(smooth)
        frames.append(
            pd.DataFrame(
                {
                    "field_id": cycle.field_id,
                    "season": cycle.season,
                    "date": days,
                    smooth_col: smooth,
                    slope_col: slope,
                    curvature_col: np.gradient(slope),
                    "observed": weights.to_numpy() > 0,
                }
            )
        )

    if not frames:
        return pd.DataFrame(
            columns=[
                "field_id", "season", "date",
                smooth_col, slope_col, curvature_col, "observed",
            ]
        )
    return pd.concat(frames, ignore_index=True)


# --- Version 2: the upper envelope --------------------------------------
# Residual cloud and haze push a vegetation index *down*, never up, so a
# symmetric fit sits below the true canopy signal. The usual answer is to pull
# the curve toward the observations above it (Chen et al., 2004, with a
# rolling maximum). The same effect is obtained here by refitting with the
# observations below the curve down-weighted, which keeps the daily grid and
# the fixed time window that a rolling maximum over observations loses.

# Weight given to an observation that falls below the current fit.
ENVELOPE_WEIGHT = 0.25

# Refits after the first one. Two is enough; more barely moves the curve.
ENVELOPE_ITERATIONS = 2

# The envelope is only applied where the fit is above this level. Below it the
# field is bare, contamination is no longer one-sided, and pulling the curve up
# would inflate the straw signal.
ENVELOPE_MIN_LEVEL = 0.40


def whittaker_envelope(
    values,
    weights,
    lam=WHITTAKER_LAMBDA,
    envelope_weight=ENVELOPE_WEIGHT,
    iterations=ENVELOPE_ITERATIONS,
    min_level=ENVELOPE_MIN_LEVEL,
):
    """Smooth `values`, biased toward the observations above the fit.

    Parameters
    ----------
    values, weights, lam
        As in `whittaker`.
    envelope_weight : float, default ENVELOPE_WEIGHT
        Weight an observation keeps when it falls below the fit.
    iterations : int, default ENVELOPE_ITERATIONS
        Number of refits after the first.
    min_level : float, default ENVELOPE_MIN_LEVEL
        Only where the fit is above this is the envelope applied.

    Returns
    -------
    numpy.ndarray
        The smoothed series.
    """
    values = np.asarray(values, dtype=float)
    smooth = whittaker(values, weights, lam=lam)

    for _ in range(iterations):
        below = (values < smooth) & (weights > 0) & (smooth >= min_level)
        adjusted = np.where(below, weights * envelope_weight, weights)
        smooth = whittaker(values, adjusted, lam=lam)
    return smooth


def smooth_cycles_v2(
    df,
    seasons,
    index="NDVI",
    lam=WHITTAKER_LAMBDA,
    cloud_weight=CLOUD_WEIGHT,
    min_observations=MIN_OBSERVATIONS,
    envelope_weight=ENVELOPE_WEIGHT,
    iterations=ENVELOPE_ITERATIONS,
    min_level=ENVELOPE_MIN_LEVEL,
):
    """Same as `smooth_cycles`, with the upper envelope of `whittaker_envelope`.

    Returns the same columns, so the two can be compared directly by swapping
    the call.
    """
    return _smooth(
        df, seasons, index, lam, cloud_weight, min_observations,
        fit=lambda v, w: whittaker_envelope(
            v, w, lam=lam, envelope_weight=envelope_weight,
            iterations=iterations, min_level=min_level,
        ),
    )
