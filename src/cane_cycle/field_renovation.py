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

The reading is taken over every bare image the cycle holds, summarised by the
median rather than by a single image, because single images lie in both
directions. The window is the cycle's own bare stretch and not a fixed number
of days after the cut, for two reasons: a renovation keeps the ground open for
months, well past any fixed window; and the first cycle of a record opens on
the limit of the series rather than on a cut, so a window anchored to a cut
cannot be asked of it at all. Measured on these two fields, the median over the
bare stretch gives 27.8%, 33.3% and 34.4% on the three renovations against
51.6% to 68.2% on every ratoon cycle, a gap of 17 points.

A cycle whose bare stretch holds fewer than `CYCLE_MIN_BARE_IMAGES` images is
not decided either way. That is what separates a first cycle the record caught
in the bare period, which reads like any other, from one it caught with the
canopy already closed, which holds almost no bare image and whose earlier
history is simply not in the data.

The flag sits on the cycle, not on the cut: the tillage happens after the last
cut of the old stand, inside the cycle that follows it, and that cycle is the
plant cane. The next one is the first ratoon.

Before the first renovation in the record the count is unknown, and stays so:
the series starts in the middle of the field's history and nothing in the
imagery says how many cuts came before it. An unknown is reported as an unknown
rather than guessed, since a wrong ratoon number is worse than none.
"""

import pandas as pd

from cane_cycle.residue import CYCLE_MIN_BARE_IMAGES, STRAW_COVER_MIN

# Residue cover below which the cut is taken to have been followed by tillage.
# The lowest class of the Zarc Niveis de Manejo, and the same boundary that
# separates straw from bare soil in `cane_cycle.residue`.
FIELD_RENOVATION_COVER_MAX = STRAW_COVER_MIN

# Label of the cycle that follows a renovation.
PLANT_CANE = "plant cane"


def flag_field_renovation(cycle_residue, cover_max=FIELD_RENOVATION_COVER_MAX):
    """Mark the cycles that grew out of a sugarcane field renovation.

    Parameters
    ----------
    cycle_residue : pandas.DataFrame
        One row per cycle, from `cane_cycle.residue.summarize_cycle_residue`,
        with the median `residue_cover_pct` of its bare stretch and
        `enough_images`.
    cover_max : float, default FIELD_RENOVATION_COVER_MAX
        Cover below which the straw is taken to have been ploughed under.

    Returns
    -------
    pandas.DataFrame
        Copy of `cycle_residue` with `field_renovation`: True when the cover
        fell below the threshold, False when it did not, and null when the
        bare stretch was too thin to read, where neither answer can be
        defended.
    """
    cycle_residue = cycle_residue.copy()
    renovated = (cycle_residue["residue_cover_pct"] < cover_max).astype("boolean")
    undecided = (
        cycle_residue["residue_cover_pct"].isna() | ~cycle_residue["enough_images"]
    )
    cycle_residue["field_renovation"] = renovated.mask(undecided)
    return cycle_residue


def count_cuts(seasons, field_renovation):
    """Number the cut each cycle carries, from the last renovation onwards.

    Parameters
    ----------
    seasons : pandas.DataFrame
        Cycles from `cane_cycle.harvest.build_seasons`.
    field_renovation : pandas.DataFrame
        Cycles with the `field_renovation` flag, from `flag_field_renovation`.

    Returns
    -------
    pandas.DataFrame
        Copy of `seasons` with `ratoon_number` (0 for the plant cane, 1 for
        the first ratoon, and so on) and `stage`, its name. Both are null on
        the cycles before the first renovation of the record, and from any
        cycle whose renovation flag could not be decided onwards.
    """
    flags = field_renovation.set_index(["field_id", "season"])["field_renovation"]
    seasons = seasons.sort_values(["field_id", "start_date"]).copy()

    numbers = []
    for field_id, cycles in seasons.groupby("field_id", sort=False):
        ratoon = None
        for cycle in cycles.itertuples():
            renovated = flags.get((field_id, cycle.season), pd.NA)
            if pd.isna(renovated):
                # A cycle whose bare stretch could not be read: the chain is
                # broken and the count is unknown again.
                ratoon = None
            elif renovated:
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
