# =============================================================================
# figure_utilities.py — Matplotlib plotting utilities (AJB-instrumented)
#
# Origin: upstream/multi-gpu-sort-merge-join/scripts/figure_utilities.py (229 lines)
# AJB adaptation (~20%):
#   - configure_plot: dict-dispatch replaces the 7-branch if/elif chain
#   - plot_lines: per-series range dump + early-exit on empty data
#   - plot_bars: overflow-safe annotation (clamps absurd bar heights)
#   - ajb_validate_plot_data: column-wise diagnostics with percentile snapshot
#   - LaTeX fallback + tier-palette for NVLink/PCIe comparison figures
# =============================================================================

import os
import sys
import traceback
import matplotlib as plotlib
import matplotlib.container as container
import matplotlib.pyplot as pyplot
import numpy
import pandas

from typing import Any, List, Tuple

# AJB: graceful LaTeX fallback — detect once, cache result
_latex_available = None

def _check_latex():
    global _latex_available
    if _latex_available is not None:
        return _latex_available
    try:
        plotlib.rcParams.update({
            "text.usetex": True,
            "text.latex.preamble": "\\usepackage{amsmath}\\usepackage{lmodern}",
        })
        _latex_available = True
    except Exception:
        _latex_available = False
        print("[AJB_WARN] LaTeX unavailable, using default text renderer",
              file=sys.stderr)
    return _latex_available

_check_latex()

plotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman"],
    "hatch.linewidth": 0.5
})

figure_width = 4.85
figure_height = 3.85

legend_font_size = 17
small_font_size = 21
large_font_size = 23

colors = ["#3193C6", "#05AD97", "#AAC56C", "#F7AB13", "#CD4E38", "#7D52A5"]
hatches = ["///", "\\\\\\", "xxx", "...", "oo"]
markers = ["o", "v", "s", "d", "h"]

# AJB: bandwidth-tier palette — one dict instead of ad-hoc per-plot color picks
ajb_tier_colors = {
    "nvlink":   "#05AD97",
    "pcie":     "#F7AB13",
    "host":     "#CD4E38",
    "ajb":      "#3193C6",
    "baseline": "#7D52A5",
}

def ajb_validate_plot_data(data, label="plot_data"):
    """Column-wise data validation with percentile snapshot.

    Unlike upstream (which silently plots NaN as gaps), this catches bad
    data before matplotlib sees it and prints a structured diagnostic.
    Returns True if data is clean.

    AJB algorithm change: instead of a single NaN/Inf flag, we walk numeric
    columns and emit per-column p5/p50/p95 so you can spot outliers or
    unit-conversion bugs (e.g. nanoseconds mixed with seconds) in the log.
    """
    issues = []
    snapshots = []

    if isinstance(data, pandas.DataFrame):
        numeric_cols = data.select_dtypes(include=[numpy.number]).columns
        for col in numeric_cols:
            vals = data[col].dropna().values
            nan_ct = int(data[col].isna().sum())
            inf_ct = int(numpy.isinf(vals).sum()) if len(vals) > 0 else 0
            neg_ct = int((vals < 0).sum()) if len(vals) > 0 else 0
            if nan_ct:
                issues.append(f"{col}: {nan_ct} NaN")
            if inf_ct:
                issues.append(f"{col}: {inf_ct} Inf")
            if neg_ct:
                issues.append(f"{col}: {neg_ct} negative")
            # percentile snapshot for non-trivial columns
            if len(vals) >= 3:
                p5, p50, p95 = numpy.percentile(vals, [5, 50, 95])
                snapshots.append(f"{col}[p5={p5:.4g} p50={p50:.4g} p95={p95:.4g}]")
    elif isinstance(data, (list, numpy.ndarray)):
        arr = numpy.asarray(data, dtype=float)
        finite = arr[numpy.isfinite(arr)]
        if numpy.any(numpy.isnan(arr)):
            issues.append("NaN detected")
        if numpy.any(numpy.isinf(arr)):
            issues.append("Inf detected")
        if len(finite) >= 3:
            p5, p50, p95 = numpy.percentile(finite, [5, 50, 95])
            snapshots.append(f"array[p5={p5:.4g} p50={p50:.4g} p95={p95:.4g}]")

    if issues:
        print(f"[AJB_WARN] {label}: {'; '.join(issues)}", file=sys.stderr)
    if snapshots:
        print(f"[AJB_TRACE] {label} snapshot: {', '.join(snapshots[:6])}",
              file=sys.stderr)
    if not issues:
        print(f"[AJB_STATE] {label}: valid ({len(snapshots)} numeric cols)",
              file=sys.stderr)
    return len(issues) == 0


def scale_figure_size(width_factor: float, height_factor: float):
    pyplot.figure(num=1, figsize=(figure_width * width_factor, figure_height * height_factor))


def calculate_means(rows: List[str]):
    return [numpy.array([[float(value) for value in row.split(",")] for row in rows]).mean(axis=0)]


def annotate_bars(bars: container.BarContainer, precision: int, height: float = None, rotation: str = None):
    bar = bars[0]
    bar_height = height or bar.get_height()

    # AJB: clamp annotation for absurdly tall bars so text stays on-canvas
    max_display = bar_height
    canvas_ylim = pyplot.gca().get_ylim()
    if canvas_ylim[1] > 0 and bar_height > canvas_ylim[1] * 2:
        max_display = canvas_ylim[1] * 0.95

    value = str(int(bar_height)) if precision == 0 else ("{:.%sf}" % (precision)).format(bar_height)

    pyplot.annotate(value,
                    xy=(bar.get_x() + bar.get_width() / 2, min(bar_height, max_display)),
                    xytext=(0, 5 if rotation is not None else 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=rotation,
                    fontsize=small_font_size,
                    zorder=3)


def plot_bars(y_values: List[float],
              precision: int,
              colors: List[str],
              hatches: List[str],
              single_width: float = 0.9,
              total_width: float = 0.8,
              bar_label_rotation: str = None):
    handles = []

    num_bars = len(y_values)
    if num_bars == 0:
        print("[AJB_WARN] plot_bars: empty y_values, nothing to draw",
              file=sys.stderr)
        return handles

    bar_width = total_width / num_bars

    for index, values in enumerate(y_values):
        x_offset = (index - num_bars / 2) * bar_width + bar_width / 2

        for x, y in enumerate(values):
            bar = pyplot.bar(x + x_offset,
                             y,
                             width=bar_width * single_width,
                             color=colors[index % len(colors)],
                             hatch=hatches[index % len(hatches)] if hatches else None,
                             alpha=0.99 if hatches is not None else 1,
                             zorder=2)

            if x == 0:
                handles.append(bar[0])

            annotate_bars(bar, precision, rotation=bar_label_rotation)

    return handles


def plot_lines(x_values: List[int], y_values: List[float], colors: List[str], markers: List[str], labels: List[str]):
    if not x_values:
        print("[AJB_WARN] plot_lines: no series to plot", file=sys.stderr)
        return

    zorder = 2 + len(x_values)
    for index in range(len(x_values)):
        xv, yv = x_values[index], y_values[index]
        # AJB: per-series range diagnostic — catch empty or mismatched data early
        if len(xv) == 0 or len(yv) == 0:
            print(f"[AJB_WARN] plot_lines: series '{labels[index]}' is empty, skipping",
                  file=sys.stderr)
            continue
        if len(xv) != len(yv):
            print(f"[AJB_WARN] plot_lines: series '{labels[index]}' "
                  f"len(x)={len(xv)} != len(y)={len(yv)}", file=sys.stderr)

        pyplot.plot(xv,
                    yv,
                    linestyle="-",
                    color=colors[index],
                    marker=markers[index],
                    markersize=4,
                    label=labels[index],
                    zorder=zorder)

        zorder -= 1


def plot_stacked_bars(data: pandas.DataFrame, segment_columns: List[str], segment_colors: List[str],
                      segment_hatches: List[str], segment_labels: List[str], column: str, precision: int):
    handles = []

    col_values = data[column].tolist()
    for column_index, column_value in enumerate(col_values):
        column_data = data[data[column] == column_value]

        bottom = 0
        for segment_index, segment_column in enumerate(segment_columns):
            seg_val = column_data[segment_column].tolist()[0]
            bar = pyplot.bar(column_value,
                             seg_val,
                             bottom=bottom,
                             color=segment_colors[segment_index % len(segment_colors)],
                             hatch=segment_hatches[segment_index %
                                                   len(segment_hatches)] if segment_hatches is not None else None,
                             alpha=0.99 if segment_hatches is not None else 1,
                             label=segment_labels[segment_index] if column_index == 0 else "",
                             zorder=2)

            if column_index == 0:
                handles.append(bar)

            bottom += seg_val

            if segment_index == len(segment_columns) - 1:
                annotate_bars(bar, precision, bottom)

    return handles


def plot_stacked_lines(data: pandas.DataFrame, segment_columns: List[str], segment_colors: List[str],
                       segment_hatches: List[str], segment_labels: List[str], column: str):
    x_values = data[column].tolist()

    y_values = []
    for segment_column in segment_columns:
        y_values.append(data[segment_column].tolist())

    segments = pyplot.stackplot(x_values, *y_values, baseline="zero", colors=segment_colors, labels=segment_labels)

    for segment, segment_hatch in zip(segments, segment_hatches):
        segment.set_hatch(segment_hatch)

    return segments


def configure_plot(x_ticks_color: str = None,
                   x_ticks_labels: List[Any] = None,
                   x_ticks_ticks: List[Any] = None,
                   y_ticks_color: str = None,
                   y_ticks_labels: List[Any] = None,
                   y_ticks_ticks: List[Any] = None,
                   set_y_limits: bool = False,
                   x_label: str = None,
                   y_label: str = None,
                   legend: bool = False,
                   legend_anchor: tuple = None,
                   legend_mode: str = None,
                   legend_location: str = None,
                   legend_handles: List[Any] = None,
                   legend_labels: List[str] = None,
                   legend_columns: int = None):
    pyplot.xticks(fontsize=small_font_size)
    if x_ticks_color is not None:
        pyplot.tick_params(axis="x", colors=x_ticks_color)

    if x_ticks_labels is not None and x_ticks_ticks is not None:
        pyplot.xticks(ticks=x_ticks_ticks, labels=x_ticks_labels)
    elif x_ticks_ticks is not None:
        pyplot.xticks(ticks=x_ticks_ticks)

    pyplot.yticks(fontsize=small_font_size)
    if y_ticks_color is not None:
        pyplot.tick_params(axis="y", colors=y_ticks_color)

    if y_ticks_labels is not None and y_ticks_ticks is not None:
        pyplot.yticks(ticks=y_ticks_ticks, labels=y_ticks_labels)
    elif y_ticks_ticks is not None:
        pyplot.yticks(ticks=y_ticks_ticks)

    if y_ticks_ticks is not None and set_y_limits:
        padding = ((y_ticks_ticks[1] - y_ticks_ticks[0]) / 5) - 0.1
        pyplot.ylim(y_ticks_ticks[0] - padding, y_ticks_ticks[-1] + padding)

    pyplot.minorticks_on()
    pyplot.tick_params(axis="x", which="minor", bottom=False)

    if x_label is not None:
        pyplot.xlabel(x_label, fontsize=large_font_size)

    if y_label is not None:
        pyplot.ylabel(y_label, fontsize=large_font_size)

    # AJB: dict-dispatch legend configuration replaces the upstream 7-branch
    # if/elif chain. Build a kwargs dict from whatever the caller provided,
    # then make a single pyplot.legend() call. Same behavior, half the lines,
    # and no chance of a missing combination silently dropping the legend.
    if legend:
        leg_kwargs = {"fontsize": legend_font_size, "labelspacing": 0.4}
        _opt_map = {
            "bbox_to_anchor": legend_anchor,
            "mode":           legend_mode,
            "loc":            legend_location,
            "handles":        legend_handles,
            "labels":         legend_labels,
            "ncol":           legend_columns,
        }
        for k, v in _opt_map.items():
            if v is not None:
                leg_kwargs[k] = v
        try:
            pyplot.legend(**leg_kwargs)
        except Exception as exc:
            print(f"[AJB_WARN] legend failed: {exc}", file=sys.stderr)
