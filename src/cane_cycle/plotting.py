"""Time series plots of vegetation indices per field."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from cane_cycle.vegetation_indices import INDEX_GROUPS

# Colors of the first and second index (fixed order).
INDEX_COLORS = ["#2a78d6", "#eb6834"]

# One hue per season, assigned in this fixed order. Seasons are contiguous in
# time and each segment carries its own label, so colour is a redundant cue and
# a ninth season may reuse the first hue without ambiguity.
SEASON_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

# One colour per surface state. Violet replaces the intuitive brown for bare
# soil: brown and green sit 3.2 units apart under protanopia, while this trio
# clears every check of the palette validator (worst pair 16.2).
SURFACE_COLORS = {
    "cloud": "#c9c9c9",
    "canopy": "#008300",
    "straw": "#eda100",
    "bare soil": "#4a3aa7",
}

# Colour of a clear observation when the smoothed curve carries the season.
POINT_COLOR_CLEAR = "#6b6b6b"

# One colour per phenological phase, running cold to warm in the order the
# phases occur, so the cycle reads as a progression rather than as four
# unrelated categories.
PHASE_COLORS = {
    "post_harvest": "#4a3aa7",
    "tillering": "#1baf7a",
    "grand_growth": "#008300",
    "maturation": "#eda100",
}
PHASE_UNCLASSIFIED_COLOR = "#b0b0b0"

# Markers of the cycle boundaries on the curve.
CYCLE_START_COLOR = "#000000"
CYCLE_END_COLOR = "#e34948"

# Style of the cut lines on the cut plot: an ordinary cut is grey and quiet,
# the cut that opens a renovation is heavy, because it is the event that
# restarts the count.
CUT_COLOR = "#909090"
# Black, not red: red is the eighth season hue, and in field 1460 the
# renovation falls exactly on the cycle that carries it.
FIELD_RENOVATION_CUT_COLOR = "#000000"

# Written over a cycle whose place in the sequence of cuts is unknown.
UNKNOWN_STAGE = "undefined"

# Headroom added above the series, as a fraction of its range, to leave room
# for the labels written over each cycle.
LABEL_HEADROOM = 0.45

# Gap, in points, between the cut line and the stage written beside it.
STAGE_LABEL_OFFSET_PT = 4

# Columns of the legend of the plots that name every season, which carry one
# entry per season plus a few style entries.
SEASON_LEGEND_COLUMNS = 5

# Y-axis label for each index group.
GROUP_LABELS = {
    "vegetation": "Vegetation index",
    "moisture": "Moisture index",
    "tillage": "Tillage index",
}

FIGURE_SIZE = (14, 5)
FIGURE_DPI = 150

# Style of the harvest markers.
HARVEST_COLOR = "red"
HARVEST_MARKER = "*"
HARVEST_SIZE = 200


def plot_index_timeseries(
    df, indices, field_id, connect_points=True, harvests=None, save_path=None
):
    """Plot the raw time series of 1 or 2 indices for one field.

    Indices of the same group (e.g. NDVI and EVI) share the y-axis; an index
    of a different group (e.g. NDTI) is drawn on a second y-axis on the right.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with `field_id`, `date` and the index columns.
    indices : list of str
        One or two index names, e.g. ["NDVI"] or ["NDVI", "NDTI"].
    field_id : int
        Field to plot.
    connect_points : bool, default True
        Draw a thin line connecting the observations.
    harvests : pandas.DataFrame, optional
        Candidate harvests from `cane_cycle.harvest.detect_harvests`. Each one
        is marked with a red asterisk on the first index, at its `harvest_date`.
    save_path : str or Path, optional
        If given, the figure is saved to this file.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if len(indices) not in (1, 2):
        raise ValueError(f"Choose 1 or 2 indices, got {len(indices)}: {indices}")
    missing = [name for name in indices if name not in df.columns]
    if missing:
        raise ValueError(f"Indices not computed yet: {missing}. Run add_indices first.")

    data = df[df["field_id"] == field_id].sort_values("date")
    if data.empty:
        raise ValueError(f"No data for field_id {field_id}")

    fig, ax_left = plt.subplots(figsize=FIGURE_SIZE)
    left_group = INDEX_GROUPS[indices[0]]
    axes_by_group = {left_group: ax_left}

    for name, color in zip(indices, INDEX_COLORS):
        group = INDEX_GROUPS[name]
        if group not in axes_by_group:
            # Different kind of index: new y-axis on the right.
            axes_by_group[group] = ax_left.twinx()
        ax = axes_by_group[group]

        if connect_points:
            ax.plot(data["date"], data[name], color=color, linewidth=0.8, alpha=0.5)
        ax.scatter(data["date"], data[name], color=color, s=14, label=name, zorder=3)
        ax.set_ylabel(GROUP_LABELS.get(group, group))

    if harvests is not None and len(harvests) > 0:
        # Mark each candidate on the first index, at the first image showing
        # the cut, so the marker sits on a real observation.
        field_harvests = harvests[harvests["field_id"] == field_id]
        marked = data[data["date"].isin(field_harvests["harvest_date"])]
        ax_left.scatter(
            marked["date"],
            marked[indices[0]],
            color=HARVEST_COLOR,
            marker=HARVEST_MARKER,
            s=HARVEST_SIZE,
            zorder=5,
            label="Harvest (candidate)",
        )

    # X axis: a tick every 3 months, labels rotated.
    ax_left.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    ax_left.tick_params(axis="x", labelrotation=90)
    ax_left.set_xlabel("Date")
    ax_left.grid(True, color="#dddddd", linewidth=0.6)
    ax_left.set_axisbelow(True)
    ax_left.set_title(f"Time series | field_id: {field_id}")

    # One legend below the plot with the entries of all axes.
    handles, labels = [], []
    for ax in axes_by_group.values():
        h, l = ax.get_legend_handles_labels()
        handles += h
        labels += l
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI)
    return fig


def plot_seasons_timeseries(df, index, field_id, seasons, save_path=None):
    """Plot one index per field, coloured by crop cycle.

    Each cycle of `seasons` runs from one harvest to the next and gets its own
    colour, so the whole record reads as a sequence of seasons. The boundaries
    are marked on the curve: black opens a cycle, red closes it. Because one
    cycle's end is the next one's start, only the first black marker stays
    visible; the last red marker is the end of the record, not a harvest.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with `field_id`, `date` and the index column.
    index : str
        Name of the index to plot, e.g. "NDVI".
    field_id : int
        Field to plot.
    seasons : pandas.DataFrame
        Output of `cane_cycle.harvest.build_seasons`.
    save_path : str or Path, optional
        If given, the figure is saved to this file.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if index not in df.columns:
        raise ValueError(f"Index not computed yet: {index}. Run add_indices first.")

    data = df[df["field_id"] == field_id].sort_values("date")
    cycles = seasons[seasons["field_id"] == field_id].sort_values("start_date")
    if data.empty or cycles.empty:
        raise ValueError(f"No data or no season for field_id {field_id}")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    for position, cycle in enumerate(cycles.itertuples()):
        color = SEASON_COLORS[position % len(SEASON_COLORS)]
        # Both boundaries are included, so consecutive cycles share a point
        # and the curve stays continuous across the colour change.
        segment = data[
            data["date"].between(cycle.start_date, cycle.end_date)
        ]
        ax.plot(
            segment["date"],
            segment[index],
            color=color,
            linewidth=1.5,
            zorder=2,
            label=f"Season {cycle.season}",
        )
        ax.scatter(
            segment["date"], segment[index], color=color, s=12, alpha=0.5, zorder=3
        )

    opens = data[data["date"].isin(cycles["start_date"])]
    closes = data[data["date"].isin(cycles["end_date"])]
    ax.scatter(
        opens["date"], opens[index], color=CYCLE_START_COLOR, s=45, zorder=5,
        label="Cycle start",
    )
    ax.scatter(
        closes["date"], closes[index], color=CYCLE_END_COLOR, s=45, zorder=6,
        label="Cycle end (harvest)",
    )

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.set_xlabel("Date")
    ax.set_ylabel(index)
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"{index} | field_id: {field_id}")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI)
    return fig


def plot_surface_timeseries(df, index, field_id, harvests=None, save_path=None):
    """Plot one index per field, with every date coloured by surface state.

    Green is a covered ground, yellow the straw a harvest left behind and
    violet the bare soil of a renovation. Violet stands in for the intuitive
    brown because brown and green are 3.2 units apart under protanopia, which
    makes them the same colour for part of the audience.

    The straw / bare soil split is a cut through a continuum. About half of
    the bare dates sit between the pure endmembers of the spectral library,
    so read the colours as a summary and `residue_cover_pct` as the measure.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with `field_id`, `date`, the index column and `surface`, from
        `cane_cycle.residue.classify_surface`.
    index : str
        Name of the index to plot, e.g. "NDVI".
    field_id : int
        Field to plot.
    harvests : pandas.DataFrame, optional
        Harvests from `cane_cycle.harvest.detect_harvests`, marked as vertical
        lines.
    save_path : str or Path, optional
        If given, the figure is saved to this file.

    Returns
    -------
    matplotlib.figure.Figure
    """
    for column in (index, "surface"):
        if column not in df.columns:
            raise ValueError(f"Missing column: {column}")

    data = df[df["field_id"] == field_id].sort_values("date")
    if data.empty:
        raise ValueError(f"No data for field_id {field_id}")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    if harvests is not None and len(harvests) > 0:
        for date in harvests.loc[harvests["field_id"] == field_id, "harvest_date"]:
            ax.axvline(date, color="#b0b0b0", linewidth=1, linestyle="--", zorder=1)
        # One proxy artist so the legend names the lines once.
        ax.plot([], [], color="#b0b0b0", linewidth=1, linestyle="--", label="Harvest")

    ax.plot(data["date"], data[index], color="#cccccc", linewidth=1, zorder=2)
    for position, (surface, color) in enumerate(SURFACE_COLORS.items()):
        marks = data[data["surface"] == surface]
        if marks.empty:
            continue
        # Cloud is drawn first and smaller: it says where the surface could
        # not be seen, without competing with the states that were measured.
        ax.scatter(
            marks["date"],
            marks[index],
            color=color,
            s=10 if surface == "cloud" else 16,
            zorder=3 + position,
            label=surface,
        )

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.set_xlabel("Date")
    ax.set_ylabel(index)
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_title(f"Surface state | field_id: {field_id} | {index}")

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI)
    return fig


def plot_cuts_timeseries(df, index, field_id, cycles, harvests=None, save_path=None):
    """Plot the raw series with the cut each cycle carries written over it.

    One colour per cycle, named in the legend by its season, and beside the cut
    that opened each cycle the stage of the crop it grew: the plant cane, a
    numbered ratoon, or `undefined` where the count is unknown because the
    record starts in the middle of the field's history.

    The cuts are drawn as vertical lines, and the one that opened a renovation
    is drawn apart: it is the last cut of the old stand, the tillage that follows
    it is what the residue cover detected, and the cycle after it is the plant
    cane. Nothing here is smoothed - the cuts are read on the observations
    themselves, as they were detected.

    Parameters
    ----------
    df : pandas.DataFrame
        Table with `field_id`, `date` and the index column.
    index : str
        Name of the index to plot, e.g. "NDVI".
    field_id : int
        Field to plot.
    cycles : pandas.DataFrame
        Cycles with `stage` and `ratoon_number`, from
        `cane_cycle.field_renovation.count_cuts`.
    harvests : pandas.DataFrame, optional
        Harvest dates. When given, only these get a cut line, which keeps the
        limits of the record from being drawn as if they were cuts.
    save_path : str or Path, optional
        If given, the figure is saved to this file.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if index not in df.columns:
        raise ValueError(f"Index not computed yet: {index}. Run add_indices first.")

    data = df[df["field_id"] == field_id].sort_values("date")
    field_cycles = cycles[cycles["field_id"] == field_id].sort_values("start_date")
    if data.empty or field_cycles.empty:
        raise ValueError(f"No data or no cycle for field_id {field_id}")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    cuts = []
    for position, cycle in enumerate(field_cycles.itertuples()):
        color = SEASON_COLORS[position % len(SEASON_COLORS)]
        # Both boundaries are included, so consecutive cycles share a point
        # and the curve stays continuous across the colour change.
        segment = data[data["date"].between(cycle.start_date, cycle.end_date)]
        ax.plot(
            segment["date"], segment[index], color=color, linewidth=1.5, zorder=2,
            label=str(cycle.season),
        )
        ax.scatter(
            segment["date"], segment[index], color=color, s=12, alpha=0.5, zorder=3
        )
        # A renovation cycle is the plant cane, so the cut that opened it is the
        # renovation cut.
        is_field_renovation = pd.notna(cycle.ratoon_number) and cycle.ratoon_number == 0
        cuts.append((cycle.start_date, is_field_renovation))
        cuts.append((cycle.end_date, False))

        # The stage is written against the cut that opened the cycle, because
        # that cut is what the number counts: the season itself is read from
        # the colour and the legend, and is not repeated here.
        stage = cycle.stage if isinstance(cycle.stage, str) else UNKNOWN_STAGE
        ax.annotate(
            stage,
            # `annotate` does not convert dates itself, unlike `text`.
            xy=(mdates.date2num(cycle.start_date), 0.98),
            xycoords=ax.get_xaxis_transform(),
            xytext=(STAGE_LABEL_OFFSET_PT, 0), textcoords="offset points",
            rotation=90, ha="left", va="top", fontsize=8, color=color, zorder=4,
        )

    if harvests is not None:
        dates = set(harvests.loc[harvests["field_id"] == field_id, "harvest_date"])
        cuts = [(date, renovation) for date, renovation in cuts if date in dates]

    for date, is_field_renovation in cuts:
        ax.axvline(
            date,
            color=FIELD_RENOVATION_CUT_COLOR if is_field_renovation else CUT_COLOR,
            linewidth=1.8 if is_field_renovation else 1.0,
            linestyle="--", zorder=1,
        )
    ax.plot([], [], color=CUT_COLOR, linewidth=1.0, linestyle="--", label="cut")
    ax.plot(
        [], [], color=FIELD_RENOVATION_CUT_COLOR, linewidth=1.8, linestyle="--",
        label="cut that opened a renovation",
    )

    # Room for the labels, which are written along the top of the axes.
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + LABEL_HEADROOM * (top - bottom))

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.set_xlabel("Date")
    ax.set_ylabel(index)
    ax.set_title(f"Cuts and ratoon number | field_id: {field_id} | {index}")
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=SEASON_LEGEND_COLUMNS,
        frameon=False, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.14, 1, 1))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI)
    return fig


def plot_smoothed_timeseries(
    df, smoothed, index, field_id, harvests=None, save_path=None
):
    """Plot the observations against the smoothed curve and its derivative.

    The top panel carries the observations and the curve fitted through them,
    one colour per cycle, named in the legend by its season. The bottom panel carries the first derivative, which
    is what the phenological phases are read from: where it crosses zero the
    crop stops growing, and its extremes are the fastest growth and the
    fastest decline.

    Parameters
    ----------
    df : pandas.DataFrame
        Observations, with `surface` so cloud can be drawn apart.
    smoothed : pandas.DataFrame
        Daily curve from `cane_cycle.smoothing.smooth_cycles`.
    index : str
        Index that was smoothed, e.g. "NDVI".
    field_id : int
        Field to plot.
    harvests : pandas.DataFrame, optional
        Harvest dates, drawn as vertical lines.
    save_path : str or Path, optional
        If given, the figure is saved to this file.

    Returns
    -------
    matplotlib.figure.Figure
    """
    data = df[df["field_id"] == field_id].sort_values("date")
    curve = smoothed[smoothed["field_id"] == field_id].sort_values("date")
    if data.empty or curve.empty:
        raise ValueError(f"No data or no smoothed curve for field_id {field_id}")

    fig, (ax_index, ax_slope) = plt.subplots(
        2, 1, figsize=(FIGURE_SIZE[0], FIGURE_SIZE[1] * 1.35),
        sharex=True, height_ratios=[2, 1],
    )

    if harvests is not None and len(harvests) > 0:
        for date in harvests.loc[harvests["field_id"] == field_id, "harvest_date"]:
            for ax in (ax_index, ax_slope):
                ax.axvline(date, color="#b0b0b0", linewidth=1, linestyle="--", zorder=1)

    clear = data[data["surface"] != "cloud"]
    cloudy = data[data["surface"] == "cloud"]
    ax_index.scatter(
        cloudy["date"], cloudy[index], color=SURFACE_COLORS["cloud"], s=10,
        zorder=2, label="cloud (weight 0)",
    )
    ax_index.scatter(
        clear["date"], clear[index], color=POINT_COLOR_CLEAR, s=14,
        zorder=3, label="observation",
    )

    for position, (season, segment) in enumerate(curve.groupby("season", sort=False)):
        color = SEASON_COLORS[position % len(SEASON_COLORS)]
        # The curve of each season carries its own name, so the legend below
        # the figure reads as the key to the colours of both panels.
        ax_index.plot(
            segment["date"], segment[f"{index}_smooth"], color=color,
            linewidth=1.8, zorder=4, label=str(season),
        )
        ax_slope.plot(
            segment["date"], segment[f"{index}_slope"], color=color, linewidth=1.4
        )

    ax_index.set_ylabel(index)
    ax_index.set_title(f"Smoothed {index} | field_id: {field_id}")

    ax_slope.axhline(0, color="#888888", linewidth=0.8)
    ax_slope.set_ylabel(f"d{index}/dt")
    ax_slope.set_xlabel("Date")

    for ax in (ax_index, ax_slope):
        ax.grid(True, color="#eeeeee", linewidth=0.6)
        ax.set_axisbelow(True)
    ax_slope.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax_slope.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    ax_slope.tick_params(axis="x", labelrotation=90, labelsize=8)

    # The two point entries are drawn first but read last: the seasons are the
    # key to the colours, the points are a footnote to them.
    handles, labels = ax_index.get_legend_handles_labels()
    order = list(range(2, len(handles))) + [1, 0]
    fig.legend(
        [handles[i] for i in order], [labels[i] for i in order],
        loc="lower center", ncol=SEASON_LEGEND_COLUMNS, frameon=False, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI)
    return fig


def plot_phenology_timeseries(df, curve, index, field_id, save_path=None):
    """Plot the smoothed curve with every day coloured by phenological phase.

    The colours run cold to warm in the order the phases occur, so the cycle
    reads left to right as a progression. A cycle the phases could not be
    read from - a field in renovation has no crop to have phenology - is drawn
    in grey.

    Parameters
    ----------
    df : pandas.DataFrame
        Observations, for the points behind the curve.
    curve : pandas.DataFrame
        Daily curve with a `phase` column, from
        `cane_cycle.phenology.add_phenology`.
    index : str
        Index that was smoothed, e.g. "NDVI".
    field_id : int
        Field to plot.
    save_path : str or Path, optional
        If given, the figure is saved to this file.

    Returns
    -------
    matplotlib.figure.Figure
    """
    points = df[df["field_id"] == field_id]
    line = curve[curve["field_id"] == field_id].sort_values("date").reset_index(drop=True)
    if line.empty:
        raise ValueError(f"No smoothed curve for field_id {field_id}")

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    clear = points[points["surface"] != "cloud"]
    ax.scatter(clear["date"], clear[index], color="#c9c9c9", s=10, zorder=2)

    # One segment per run of the same phase, each carrying the first day of
    # the next run so the curve never breaks between colours.
    column = f"{index}_smooth"
    run = (line["phase"] != line["phase"].shift()).cumsum()
    for _, segment in line.groupby(run):
        phase = segment["phase"].iloc[0]
        # Carry the first day of the next run so the colours meet with no gap.
        after = segment.index[-1] + 1
        drawn = pd.concat([segment, line.iloc[after : after + 1]])
        ax.plot(
            drawn["date"], drawn[column],
            color=PHASE_COLORS.get(phase, PHASE_UNCLASSIFIED_COLOR),
            linewidth=2, zorder=3, solid_capstyle="round",
        )

    for phase, color in PHASE_COLORS.items():
        ax.plot([], [], color=color, linewidth=2, label=phase)
    if line["phase"].isna().any():
        ax.plot([], [], color=PHASE_UNCLASSIFIED_COLOR, linewidth=2, label="unclassified")

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    ax.set_xlabel("Date")
    ax.set_ylabel(index)
    ax.set_title(f"Phenological phases | field_id: {field_id} | {index}")
    ax.grid(True, color="#eeeeee", linewidth=0.6)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.09, 1, 1))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=FIGURE_DPI)
    return fig
