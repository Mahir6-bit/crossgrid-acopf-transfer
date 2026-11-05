"""Common journal style for the redrawn Results figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

FONT = "Times New Roman" if any("Times New Roman" in f.name for f in font_manager.fontManager.ttflist) else "Liberation Serif"
FS = 14
LW = 3.5
MS = 9
PAL = {"blue": "#1f4e9c", "red": "#d1453b", "green": "#2e9d4a", "orange": "#e08a00",
       "purple": "#7b3fa0", "teal": "#0f9aa8", "grey": "#555555"}
# one colour per model or case, the same in every figure
MODEL = {"ft": PAL["blue"], "scratch": PAL["red"], "mlp": PAL["green"], "ref": PAL["grey"],
         "dcopf": PAL["grey"], "mlp_onehot": PAL["purple"], "mlp_unseen": PAL["orange"]}

plt.rcParams.update({
    "font.family": FONT, "font.weight": "bold", "font.size": FS,
    "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.titlesize": FS, "axes.labelsize": FS,
    "xtick.labelsize": FS, "ytick.labelsize": FS, "legend.fontsize": FS,
    "mathtext.fontset": "custom", "mathtext.rm": f"{FONT}:bold", "mathtext.it": f"{FONT}:bold:italic",
    "mathtext.bf": f"{FONT}:bold", "mathtext.default": "bf",
    "axes.linewidth": 2.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.major.width": 2.0, "ytick.major.width": 2.0, "xtick.major.size": 7, "ytick.major.size": 7,
    "xtick.minor.width": 1.5, "ytick.minor.width": 1.5, "xtick.minor.size": 4, "ytick.minor.size": 4,
    "legend.handlelength": 2.5, "legend.framealpha": 1.0, "legend.edgecolor": "#333333", "legend.fancybox": False,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def new_axes(w=8.0, h=5.2):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(True, which="major", color="#c8c8c8", linewidth=0.8, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    return fig, ax


def bold_ticks(ax):
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


def framed_legend(ax, **kw):
    leg = ax.legend(frameon=True, **kw)
    leg.get_frame().set_linewidth(1.6)
    leg.get_frame().set_edgecolor("#333333")
    for t in leg.get_texts():
        t.set_fontweight("bold")
    return leg


def callout(ax, x, y, text, **kw):
    return ax.text(x, y, text, transform=ax.transAxes, fontsize=FS, fontweight="bold", zorder=20,
                   bbox=dict(boxstyle="square,pad=0.35", facecolor="white", edgecolor="black", linewidth=1.8, linestyle="--"), **kw)
