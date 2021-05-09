import numpy as np
from typing import List


def tidy_legend(ax):
    """
    Tidy up a plot's legend by removing duplicate entries

    :param ax: The matplotlib axes object where the legend labels reside.
    :return:
    """
    handles, labels = ax.get_legend_handles_labels()
    labels, ids = np.unique(labels, return_index=True)
    handles = [handles[i] for i in ids]
    # Add labels to plot
    ax.legend(handles, labels, loc="best")
    return ax


def glow(lines: List, ax, n_glow_lines: int = 20, diff_linewidth: float=1.05, alpha_line: float=0.3):
    """
    Heavily based on the `make_lines_glow` from the mplcyberpunk package - the primary difference being that this
    function is applied to potentially a singular line as, when sample paths are plotted, the glow effect can be
    overwhelming.

    :param lines: The set of line(s) t
    :param ax: The figure axes
    :param n_glow_lines:
    :param diff_linewidth:
    :param alpha_line:
    :return:
    """
    alpha_value = alpha_line / n_glow_lines
    for line in lines:
        data = line.get_data()
        linewidth = line.get_linewidth()
        for n in range(1, n_glow_lines + 1):
            glow_line, = ax.plot(*data)
            glow_line.update_from(line)  # line properties are copied as seen in this solution: https://stackoverflow.com/a/54688412/3240855
            glow_line.set_alpha(alpha_value)
            glow_line.set_linewidth(linewidth + (diff_linewidth * n))
            glow_line.is_glow_line = True  # mark the glow lines, to disregard them in the underglow function.
    return ax
