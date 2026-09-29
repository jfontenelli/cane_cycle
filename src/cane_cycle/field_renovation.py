"""Sugarcane field renovation (reforma) and the ratoon number of a cycle.

Sugarcane is not replanted every year. One planting is cut several times: the
first harvest takes the plant cane, and each regrowth after it is a ratoon,
which yields a little less than the one before. When the stand no longer pays
for itself the field is renovated - ploughed, usually rested or rotated, and
replanted - and the count starts over. Knowing which cut a cycle carries is
what makes a yield comparable to another: a first ratoon and a fifth ratoon in
the same year are not the same crop.

The renovation is read from the ground, not from the calendar. A green harvest
leaves a trash blanket over the ratoon, so the cover measured in the weeks
after an ordinary cut is high. A renovation harvest is followed by tillage,
which buries the straw and exposes the soil, so the cover collapses. The threshold
is the lowest class of the Zarc Niveis de Manejo, 45%, the same boundary that
separates straw from bare soil elsewhere in this package: below it the ground
is no longer protected, which is exactly the state tillage produces.

The reading is taken over the `RESIDUE_WINDOW_DAYS` window after the cut and
summarised by the median, not by a single image, because single images lie in
both directions. Measured on these two fields:

* every image below 45% would detect nothing - both renovation cuts keep one
  image above the threshold (53% and 45%), a strip of remaining straw or a
  thin cloud;
* any image below 45% would raise a false renovation on the 2021-09-06 cut of
  field 1460, which shows one isolated 32% among values of 60% to 67%;
* the median of the window marks both renovations and rejects the isolated dip.

The cut that carries the signal is the *last cut of the old stand*. What
follows it is the fallow and the replanting, so the cycle that opens on that
cut is the plant cane, and the harvest closing it is the plant cane harvest.
The next cycle is the first ratoon.

Before the first renovation in the record the count is unknown, and stays so:
the series starts in the middle of the field's history and nothing in the
imagery says how many cuts came before it. An unknown is reported as an unknown rather
than guessed, since a wrong ratoon number is worse than none.
"""

import pandas as pd

from cane_cycle.residue import STRAW_COVER_MIN

# Residue cover below which the cut is taken to have been followed by tillage.
# The lowest class of the Zarc Niveis de Manejo, and the same boundary that
# separates straw from bare soil in `cane_cycle.residue`.
FIELD_RENOVATION_COVER_MAX = STRAW_COVER_MIN

# Label of the cycle that follows a renovation.
PLANT_CANE = "plant cane"


def flag_field_renovation(residue, cover_max=FIELD_RENOVATION_COVER_MAX):
    """Mark the harvests that opened a sugarcane field renovation.

    Parameters
    ----------
    residue : pandas.DataFrame
        One row per harvest, from `cane_cycle.residue.summarize_residue`,
        with the median `residue_cover_pct` of the window after the cut.
    cover_max : float, default FIELD_RENOVATION_COVER_MAX
        Cover below which the straw is taken to have been ploughed under.

    Returns
    -------
    pandas.DataFrame
        Copy of `residue` with `field_renovation`: True when the cover fell
        below the threshold, False when it did not, and null when the window
        held no clear image, where neither answer can be defended.
    """
    residue = residue.copy()
    renovated = (residue["residue_cover_pct"] < cover_max).astype("boolean")
    residue["field_renovation"] = renovated.mask(
        residue["residue_cover_pct"].isna()
    )
    return residue


def count_cuts(seasons, field_renovation):
    """Number the cut each cycle carries, from the last renovation onwards.

    Parameters
    ----------
    seasons : pandas.DataFrame
        Cycles from `cane_cycle.harvest.build_seasons`.
    field_renovation : pandas.DataFrame
        Harvests with the `field_renovation` flag, from `flag_field_renovation`.

    Returns
    -------
    pandas.DataFrame
        Copy of `seasons` with `ratoon_number` (0 for the plant cane, 1 for
        the first ratoon, and so on) and `stage`, its name. Both are null on
        the cycles before the first renovation of the record, and on every cycle
        after a harvest whose renovation flag could not be decided.
    """
    indexed = field_renovation.set_index(["field_id", "harvest_date"])
    flags = indexed["field_renovation"]
    seasons = seasons.sort_values(["field_id", "start_date"]).copy()

    numbers = []
    for field_id, cycles in seasons.groupby("field_id", sort=False):
        ratoon = None
        for cycle in cycles.itertuples():
            # The cut that opened this cycle. The first cycle of a field opens
            # on the limit of the record instead, and has no flag.
            opening = flags.get((field_id, cycle.start_date), pd.NA)
            if pd.isna(opening):
                # The limit of the record, or a cut whose cover could not be
                # read: the chain is broken and the count is unknown again.
                ratoon = None
            elif opening:
                ratoon = 0
            elif ratoon is not None:
                ratoon += 1
            numbers.append(ratoon)

    seasons["ratoon_number"] = pd.array(numbers, dtype="Int64")
    seasons["stage"] = seasons["ratoon_number"].map(_stage_name, na_action="ignore")
    return seasons.reset_index(drop=True)


def _stage_name(ratoon_number):
    """Name a ratoon number: 0 is the plant cane, 1 the first ratoon."""
    # `Series.map` hands the Int64 value over as a float.
    ratoon_number = int(ratoon_number)
    if ratoon_number == 0:
        return PLANT_CANE
    return f"{_ordinal(ratoon_number)} ratoon"


def _ordinal(number):
    """Write an integer as an English ordinal: 1st, 2nd, 3rd, 4th."""
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"
