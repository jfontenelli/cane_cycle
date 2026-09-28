"""Time series plots of vegetation indices per field."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

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

# Markers of the cycle boundaries on the curve.
CYCLE_START_COLOR = "#000000"
CYCLE_END_COLOR = "#e34948"

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
