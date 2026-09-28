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
        where growth stops.
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

        smooth = whittaker(values.to_numpy(), weights.to_numpy(), lam=lam)
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
