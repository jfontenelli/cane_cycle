"""Time series plots of vegetation indices per field."""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

from cane_cycle.vegetation_indices import INDEX_GROUPS

# Colors of the first and second index (fixed order).
INDEX_COLORS = ["#2a78d6", "#eb6834"]

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
        is marked with a red asterisk on the first index, at the first dry
        image of the transition (the first observation that shows the cut).
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
        marked = data[data["date"].isin(field_harvests["first_dry_date"])]
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
