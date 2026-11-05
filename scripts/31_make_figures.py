"""
31_make_figures.py
Paper figures from results\\*\\metrics.json (IEEE 30 TEST, Nov-Dec).

  transfer_curves.png/.pdf  (plan Fig. 6, data-efficiency curve)
      metric vs share of IEEE 30 training data, fine-tune vs from scratch,
      mean line + shaded band = +-1 std over seeds, log-scale y,
      dashed reference = from scratch with 100 % data, zero-shot value printed in each panel.
  ablation.png/.pdf         (plan Fig. 11, E5) - only if ablation runs (tagged) exist
      zero-shot and fine-tune at 1 % / 5 % for every variant vs the full model.

Run from the project folder:
    python scripts\\31_make_figures.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SUB = json.load(open(RES.parents[0] / "data" / "processed" / "finetune_subsets_case30.json"))
N_HOURS = {round(float(f) * 100): len(v) for f, v in SUB["seeds"]["0"]["train"].items()}

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FT, SC, MLPC = "#2a78d6", "#eb6834", "#1baf7a"   # validated categorical slots 1, 2, 3
MM = 1 / 25.4
DOUBLE = 183 * MM                                 # Nature Portfolio double-column width
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0,
                     "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def panel(ax, letter):
    """Nature-style panel label: bold lower-case letter, no brackets."""
    ax.text(-0.02, 1.03, letter, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold", color=INK)

# ---------------------------------------------------------------- collect
rows = []
for f in RES.glob("*/metrics.json"):
    parts = f.parent.name.split("_")
    for m in json.load(open(f)):
        if m["case"] != "case30":
            continue
        if parts[0] in ("ft30", "scratch30") and len(parts) >= 3:
            m.update(kind=parts[0], pct=float(parts[1][1:]) * 100, seed=int(parts[2][1:]),
                     tag="_".join(parts[3:]))
        elif parts[0] == "src14" and len(parts) >= 2:
            m.update(kind="zero", pct=0.0, seed=int(parts[1][1:]), tag="_".join(parts[2:]))
        else:
            continue
        rows.append(m)
df = pd.DataFrame(rows)
main = df[df.tag == ""]

PANELS = [("pg_mae_mw", "Generator P MAE (MW)"), ("va_mae_deg", "Voltage angle MAE (deg)"),
          ("vm_mae_pu", "Voltage magnitude MAE (pu)"), ("cost_gap_abs_pct", "|Optimality gap| (%)"),
          ("pb_p_mean_mw", "Mean |ΔP| mismatch (MW)"), ("pb_p_max_mw", "Max |ΔP| mismatch (MW)")]


SUP = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


def tick(v):
    """0.05 -> '0.05', 5e-05 -> '5\u00d710\u207b\u2075' (no computer notation on a journal axis)."""
    if v <= 0 or v >= 1e-3:
        return f"{v:g}"
    e = int(np.floor(np.log10(v)))
    m = v / 10 ** e
    mant = f"{m:g}\\times" if abs(m - 1) > 1e-9 else ""
    return f"${mant}10^{{{e}}}$"              # mathtext: renders with any font (no missing glyphs)


def style(ax):
    ax.grid(True, color=GRID, lw=1.0, which="major")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- Fig. 6: data efficiency
fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 112 * MM), constrained_layout=True)
zs = main[main.kind == "zero"]
xt = [1, 5, 10, 25, 100]
for ax, (key, title), letter in zip(axes.flat, PANELS, "abcdef"):
    for kind, color, marker, label in [("ft30", FT, "o", "Fine-tune (from IEEE 14)"),
                                       ("scratch30", SC, "s", "From scratch")]:
        g = main[main.kind == kind].groupby("pct")[key].agg(["mean", "std"]).fillna(0)
        lo = np.maximum(g["mean"] - g["std"], g["mean"] * 0.05)
        ax.fill_between(g.index, lo, g["mean"] + g["std"], color=color, alpha=0.15, lw=0)
        ax.plot(g.index, g["mean"], color=color, marker=marker, ms=4, lw=1.5, label=label)
        if kind == "scratch30" and 100 in g.index:
            ax.axhline(g.loc[100, "mean"], color=SC, lw=1, ls="--",
                       label="From scratch, 100 % data")
    mlp = df[(df.kind == "scratch30") & (df.tag == "mlp")]
    if len(mlp):                                   # grid-specific reference (3 seeds)
        g = mlp.groupby("pct")[key].agg(["mean", "std"]).fillna(0)
        lo = np.maximum(g["mean"] - g["std"], g["mean"] * 0.05)
        ax.fill_between(g.index, lo, g["mean"] + g["std"], color=MLPC, alpha=0.12, lw=0)
        ax.plot(g.index, g["mean"], color=MLPC, marker="^", ms=4, lw=1.5, ls="-.",
                label="MLP from scratch (grid-specific, cannot transfer)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{p}\n{N_HOURS.get(p, '?')}" for p in xt], fontsize=6.5)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: tick(v)))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("IEEE 30 training data (% and hours)")
    ax.set_title(title, color=INK, loc="left")
    style(ax); panel(ax, letter)
    if len(zs):
        ax.text(0.98, 0.96, f"zero-shot: {zs[key].mean():.3g}", transform=ax.transAxes,
                ha="right", va="top", color=MUTED, fontsize=6.5)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="outside upper center", ncol=2, frameon=False)
fig.savefig(OUT / "transfer_curves.png", dpi=600)
fig.savefig(OUT / "transfer_curves.pdf")
print(f"Saved {OUT / 'transfer_curves.png'} and .pdf")

# ---------------------------------------------------------------- Fig. 11: ablation (E5)
NAMES = {"": "Full model", "nope": "w/o ER-PE", "noattn": "w/o global attention",
         "nophys": "w/o physics loss", "augnone": "w/o augmentation",
         "augL": "load aug. only (L)", "augLP": "load + parameter aug. (LP)"}
abl = df[df.tag.isin(NAMES) & ((df.kind == "zero") | ((df.kind == "ft30") & df.pct.isin([1, 5])))]
if abl.tag.nunique() > 1:
    abl = abl[abl.seed.isin([0, 1, 2])]          # ablations use seeds 0-2: compare like with like
    order = [t for t in NAMES if t in set(abl.tag)]
    settings = [("zero", 0, "Zero-shot"), ("ft30", 1, "Fine-tune, 1 %"), ("ft30", 5, "Fine-tune, 5 %")]
    metrics = [("pg_mae_mw", "Generator P MAE (MW)"), ("pb_p_mean_mw", "Mean |ΔP| mismatch (MW)")]
    fig, axes = plt.subplots(len(metrics), 3, figsize=(DOUBLE, 100 * MM), constrained_layout=True)
    for r, (key, mlabel) in enumerate(metrics):
        for c, (kind, pct, slabel) in enumerate(settings):
            ax = axes[r, c]
            d = abl[(abl.kind == kind) & (abl.pct == pct)]
            g = d.groupby("tag")[key].agg(["mean", "std"]).reindex(order)
            y = np.arange(len(order))
            colors = [FT if t == "" else "#a9a8a2" for t in order]
            ax.barh(y, g["mean"], xerr=g["std"].fillna(0), color=colors, height=0.6, lw=0,
                    error_kw=dict(elinewidth=1, capsize=2, ecolor=MUTED))
            ax.set_yticks(y)
            ax.set_yticklabels([NAMES[t] for t in order] if c == 0 else [])
            ax.invert_yaxis()                      # bars start at zero: linear axis
            ax.set_title(f"{slabel}" if r == 0 else "", color=INK, loc="left")
            ax.set_xlabel(mlabel)                  # each row has its own metric
            style(ax); panel(ax, "abcdef"[r * 3 + c])
    fig.savefig(OUT / "ablation.png", dpi=600)
    fig.savefig(OUT / "ablation.pdf")
    print(f"Saved {OUT / 'ablation.png'} and .pdf")
