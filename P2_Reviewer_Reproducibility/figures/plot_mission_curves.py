"""Draw the four-panel monochrome P2 result figure from checked replay observations."""
from argparse import ArgumentParser
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np


BUDGET_STYLE = (0, (5, 2, 1, 2))
SELECTORS = (
    ("hard_thermal", "Thermal baseline", (0, (1.0, 2.0)), "o"),
    ("soft_RF", "Soft rainflow penalty", (0, (4.2, 2.5)), "^"),
    ("hard_RF", "Hard rainflow budget", "-", "s"),
)


def axes_style(ax, title, ylabel, yticks):
    ax.set_title(title, loc="left", fontweight="bold", pad=7)
    ax.set_ylabel(ylabel, labelpad=4)
    ax.set_yticks(yticks)
    ax.set_xlim(0, 30)
    ax.set_xticks(np.arange(0, 31, 5))
    ax.tick_params(direction="out", length=3, width=0.55, pad=2.5)
    ax.spines[["top", "right"]].set_visible(False)
    for name in ("left", "bottom"):
        ax.spines[name].set_linewidth(0.58)


def trace_panel(ax, centres, series, quantity):
    for key, _, style, _ in reversed(SELECTORS):
        ax.plot(centres, series[key][quantity], color="black", linestyle=style,
                linewidth=0.85, dash_capstyle="butt")


def add_braking_zoom(ax, centres, series, quantity, position, ylim, yticks, title_inside=False):
    """Show the same eight complete late-braking windows on a local scale."""
    left, right = 25.8, 26.6
    in_window = (centres >= left) & (centres < right)
    assert np.count_nonzero(in_window) == 8
    zoom = ax.inset_axes(position, zorder=6)
    zoom.set_facecolor("white")
    zoom.set_xlim(left, right)
    zoom.set_ylim(*ylim)
    zoom.set_xticks([25.8, 26.2, 26.6])
    zoom.set_yticks(yticks)
    if title_inside:
        zoom.text(0.045, 0.94, "Late braking", transform=zoom.transAxes,
                  fontsize=7.1, va="top")
    else:
        zoom.set_title("Late braking", fontsize=7.4, pad=2)
    zoom.tick_params(labelsize=7, direction="out", length=2, width=0.5, pad=1.5)
    for spine in zoom.spines.values():
        spine.set_linewidth(0.55)
    for index, (key, _, style, marker) in enumerate(SELECTORS):
        points = np.asarray(series[key][quantity])[in_window]
        assert ylim[0] <= points.min() <= points.max() <= ylim[1]
        zoom.plot(centres[in_window], points, color="black", linestyle=style,
                  linewidth=0.74, dash_capstyle="butt", marker=marker,
                  markersize=3.05, markerfacecolor="white", markeredgewidth=0.7,
                  markevery=(index, 3))
    ax.add_patch(Rectangle((left, ylim[0]), right - left, ylim[1] - ylim[0],
                           fill=False, edgecolor="black", linewidth=0.62,
                           linestyle=(0, (2, 1.5)), zorder=4))


def terminal_charge_panel(ax, series, budget):
    """Place terminal charges on a separate, expanded vertical scale."""
    ax.set_title("Terminal", fontsize=8.25, pad=7)
    ax.set_xlim(0.52, 3.48)
    ax.set_ylim(6.48, 7.02)
    ax.set_xticks([])
    ax.set_yticks([6.5, 6.7, 6.9])
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", labelsize=7.0, direction="out", length=2.2, width=0.5, pad=1.7)
    ax.spines[["top", "left", "bottom"]].set_visible(False)
    ax.spines["right"].set_linewidth(0.55)
    ax.axhline(budget, color="black", linewidth=0.69, linestyle=BUDGET_STYLE, zorder=1)
    ax.text(0.04, budget + 0.016, f"Budget {budget:.3f}", transform=ax.get_yaxis_transform(),
            va="bottom", fontsize=7.0)
    for i, (key, _, _, symbol) in enumerate(SELECTORS, 1):
        charge = series[key]["terminal_charge"]
        ax.plot(i, charge, marker=symbol, color="black", markerfacecolor="white",
                markeredgewidth=0.9, markersize=5.3, linestyle="none", zorder=3)
        above = key != "soft_RF"
        ax.annotate(f"{charge:.3f}", (i, charge),
                    xytext=(-3.0 if key == "hard_RF" else 0, 4.6 if above else -4.9),
                    textcoords="offset points", ha="center",
                    va="bottom" if above else "top", fontsize=7.1)


def draw(data, output):
    times = np.asarray(data["window_edges_s"])
    centres = (times[:-1] + times[1:]) / 2
    series = data["selectors"]
    budget = data["budget"]
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 8.45,
        "axes.titlesize": 9.25,
        "axes.labelsize": 8.45,
        "pdf.fonttype": 42,
        "mathtext.fontset": "cm",
        "text.color": "black",
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
    })

    fig = plt.figure(figsize=(6.08, 4.95), facecolor="white")
    grid = fig.add_gridspec(2, 2, left=0.095, right=0.948, bottom=0.098,
                            top=0.870, wspace=0.33, hspace=0.49)
    demand = fig.add_subplot(grid[0, 0])
    error = fig.add_subplot(grid[0, 1])
    temp = fig.add_subplot(grid[1, 0])
    rf_grid = grid[1, 1].subgridspec(1, 2, width_ratios=[1.63, 1.05], wspace=0.32)
    settled = fig.add_subplot(rf_grid[0])
    terminal = fig.add_subplot(rf_grid[1])

    axes_style(demand, "(a) Current demand", "Reference amplitude (A)", [0, 200, 400])
    demand.set_ylim(0, 555)
    demand.step(data["demand_time_s"], data["reference_crest_A"], where="post",
                color="black", linewidth=0.95)
    limits = [0.0, *data["phase_changes_s"], 30.0]
    for boundary in limits[1:-1]:
        demand.axvline(boundary, color="black", linewidth=0.55, linestyle=(0, (2.0, 2.6)))
    for index, label in enumerate(("Traction", "Braking", "Rest")):
        centre = (limits[index] + limits[index + 1]) / 2 + (0.0, -0.53, 0.25)[index]
        demand.text(centre, 521, label, ha="center", va="center", fontsize=7.1,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.16})

    axes_style(error, "(b) Current tracking", "Windowed RMS error (A)", [0, 10, 20, 30, 40])
    error.set_ylim(0, 43)
    trace_panel(error, centres, series, "window_rms_A")
    add_braking_zoom(error, centres, series, "window_rms_A",
                     [0.15, 0.14, 0.55, 0.41], (34.0, 36.8), [34, 35, 36])

    axes_style(temp, "(c) Thermal upper bound", "Peak bound (°C)", [40, 55, 70, 85, 100])
    temp.set_ylim(37, 101)
    trace_panel(temp, centres, series, "window_peak_bound_C")
    add_braking_zoom(temp, centres, series, "window_peak_bound_C",
                     [0.20, 0.10, 0.55, 0.286], (86.6, 94.5), [87, 90, 93], title_inside=True)
    temp.set_xlabel("Mission time (s)", labelpad=4)

    axes_style(settled, "(d) Rainflow charge", "Maximum device charge", [0, 2, 4, 6])
    settled.set_xticks([0, 10, 20, 30])
    settled.set_ylim(0, 7.6)
    settled.set_xlabel("Mission time (s)", labelpad=4)
    for key, _, style, _ in reversed(SELECTORS):
        settled.step(times, series[key]["settled_charge"], where="post",
                     color="black", linestyle=style, linewidth=0.9)
    settled.axhline(budget, color="black", linewidth=0.69, linestyle=BUDGET_STYLE, zorder=1)
    settled.text(0.8, budget + 0.15, "Budget", va="bottom", fontsize=7.5)
    terminal_charge_panel(terminal, series, budget)

    handles = [Line2D([0], [0], color="black", linestyle=style, linewidth=0.85,
                      marker=symbol, markerfacecolor="white", markeredgewidth=0.85,
                      markersize=4.2, label=label) for _, label, style, symbol in SELECTORS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.988),
               ncol=3, frameon=False, fontsize=8.4, handlelength=2.55,
               columnspacing=1.85, borderaxespad=0.15, handletextpad=0.52)

    assert max(max(p["window_rms_A"]) for p in series.values()) <= 43
    assert max(max(p["window_peak_bound_C"]) for p in series.values()) <= 101
    fig.savefig(output, metadata={"Title": "Time-resolved synthetic rail duty results",
                                 "Subject": "Checked deterministic replays of three archived schedules",
                                 "CreationDate": None, "ModDate": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("curves", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    draw(json.loads(args.curves.read_text(encoding="utf-8")), args.output)
