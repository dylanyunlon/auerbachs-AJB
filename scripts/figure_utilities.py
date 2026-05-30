import matplotlib as plotlib
import matplotlib.container as container
import matplotlib.pyplot as pyplot
import numpy
import pandas

from typing import Any, List, Tuple

plotlib.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": "\\usepackage{amsmath}\\usepackage{lmodern}",
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


def scale_figure_size(width_factor: float, height_factor: float):
    pyplot.figure(num=1, figsize=(figure_width * width_factor, figure_height * height_factor))


def calculate_means(rows: List[str]):
    return [numpy.array([[float(value) for value in row.split(",")] for row in rows]).mean(axis=0)]


def annotate_bars(bars: container.BarContainer, precision: int, height: float = None, rotation: str = None):
    bar = bars[0]
    bar_height = height or bar.get_height()

    value = str(int(bar_height)) if precision == 0 else ("{:.%sf}" % (precision)).format(bar_height)

    pyplot.annotate(value,
                    xy=(bar.get_x() + bar.get_width() / 2, bar_height),
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
    bar_width = total_width / num_bars

    for index, values in enumerate(y_values):
        x_offset = (index - num_bars / 2) * bar_width + bar_width / 2

        for x, y in enumerate(values):
            bar = pyplot.bar(x + x_offset,
                             y,
                             width=bar_width * single_width,
                             color=colors[index % len(colors)],
                             hatch=hatches[index % len(hatches)],
                             alpha=0.99 if hatches is not None else 1,
                             zorder=2)

            if x == 0:
                handles.append(bar[0])

            annotate_bars(bar, precision, rotation=bar_label_rotation)

    return handles


def plot_lines(x_values: List[int], y_values: List[float], colors: List[str], markers: List[str], labels: List[str]):
    zorder = 2 + len(x_values)
    for index in range(len(x_values)):
        pyplot.plot(x_values[index],
                    y_values[index],
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

    for column_index, column_value in enumerate(data[column].tolist()):
        column_data = data[data[column] == column_value]

        bottom = 0
        for segment_index, segment_column in enumerate(segment_columns):
            bar = pyplot.bar(column_value,
                             column_data[segment_column].tolist()[0],
                             bottom=bottom,
                             color=segment_colors[segment_index % len(segment_colors)],
                             hatch=segment_hatches[segment_index %
                                                   len(segment_hatches)] if segment_hatches is not None else None,
                             alpha=0.99 if segment_hatches is not None else 1,
                             label=segment_labels[segment_index] if column_index == 0 else "",
                             zorder=2)

            if column_index == 0:
                handles.append(bar)

            bottom += column_data[segment_column].tolist()[0]

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

    if legend and legend_anchor and legend_mode and legend_location is not None and legend_handles is not None and legend_labels is not None and legend_columns is not None:
        pyplot.legend(fontsize=legend_font_size,
                      bbox_to_anchor=legend_anchor,
                      mode=legend_mode,
                      loc=legend_location,
                      handles=legend_handles,
                      labels=legend_labels,
                      ncol=legend_columns,
                      labelspacing=0.4)
    elif legend and legend_anchor and legend_mode and legend_location is not None and legend_handles is not None and legend_columns is not None:
        pyplot.legend(fontsize=legend_font_size,
                      bbox_to_anchor=legend_anchor,
                      mode=legend_mode,
                      loc=legend_location,
                      handles=legend_handles,
                      ncol=legend_columns,
                      labelspacing=0.4)
    elif legend and legend_anchor and legend_mode and legend_location is not None and legend_columns is not None:
        pyplot.legend(fontsize=legend_font_size,
                      bbox_to_anchor=legend_anchor,
                      mode=legend_mode,
                      loc=legend_location,
                      ncol=legend_columns,
                      labelspacing=0.4)
    elif legend and legend_location is not None and legend_handles is not None and legend_columns is not None:
        pyplot.legend(fontsize=legend_font_size,
                      loc=legend_location,
                      handles=legend_handles,
                      ncol=legend_columns,
                      labelspacing=0.4)
    elif legend and legend_location is not None and legend_handles is not None:
        pyplot.legend(fontsize=legend_font_size, loc=legend_location, handles=legend_handles, labelspacing=0.4)
    elif legend and legend_location is not None:
        pyplot.legend(fontsize=legend_font_size, loc=legend_location, labelspacing=0.4)
    elif legend:
        pyplot.legend(fontsize=legend_font_size, labelspacing=0.4)
