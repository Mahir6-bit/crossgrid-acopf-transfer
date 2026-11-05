"""
32_make_paper_figures.py
All paper figures in one consistent style (Elsevier widths: 90 mm single, 140 mm 1.5, 190 mm double
column; 8 pt text). Each figure is saved as PNG (600 dpi, for Word) and PDF (vector).

  fig01_framework        framework diagram
  fig02_grids            IEEE 14 / IEEE 30 graphs
  fig03_load_profile     yearly load profile with train / validation / test periods + one test week
  fig04_distribution     distribution shift IEEE 14 vs IEEE 30 (inputs and labels)
  fig05_training_curves  validation loss vs step, fine-tune vs from scratch
  fig06                  data-efficiency curve -> results\\figures\\transfer_curves.png (make_figures.py)
  fig07_parity           predicted vs true Vm, Va, Pg, Qg (IEEE 30 test)
  fig08_bus_profile      Vm / Va over all 30 buses at the peak-cost test hour, and errors
  fig09_test_week        hourly generation cost in one test week, and relative error
  fig10_violations       constraint violations before and after power-flow post-processing
  fig11                  ablation -> results\\figures\\ablation.png (make_figures.py)
  fig12_runtime          runtime per sample (log scale)
  fig13_n1               N-1 test: no outage data / with outage data / unseen outages

Run from the project folder (after predict_test.py):
    python scripts\\32_make_paper_figures.py
Output: results\\figures\\paper\\
"""

import json
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, PROC = ROOT / "results", ROOT / "data" / "processed"
OUT = RES / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

MM = 1 / 25.4
SINGLE, ONEHALF, DOUBLE = 89 * MM, 120 * MM, 183 * MM      # Nature Portfolio widths
INK, MUTED, GRID, NEUTRAL = "#0b0b0b", "#52514e", "#e4e3df", "#a9a8a2"
FT, SC, MLPC, YEL, MAG, VIO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 1.0, "xtick.major.width": 1.0, "ytick.major.width": 1.0, "lines.linewidth": 1.5,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def style(ax, grid="y"):
    if grid:
        ax.grid(True, axis=grid, color=GRID, lw=1.0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=600)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


def panel(ax, letter):
    """Nature-style panel label: bold lower-case letter, no brackets."""
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight="bold", color=INK)


SUP = str.maketrans("-0123456789", "\u207b\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")


def sci(v):
    """2 significant digits; 6.9e-05 -> '6.9\u00d710\u207b\u2075' (no computer notation)."""
    if v == 0 or abs(v) >= 1e-3:
        return f"{v:#.2g}".rstrip(".")
    e = int(np.floor(np.log10(abs(v))))
    return f"${v / 10 ** e:.1f}\\times10^{{{e}}}$"   # mathtext: renders with any font


def runs_df():
    df = pd.read_csv(RES / "all_runs.csv")
    return df


def tensors(case, split=None):
    import torch
    from opf_dataset import OPFGraphDataset
    return OPFGraphDataset(case, split=split)


def pred(run, case="case30"):
    return np.load(RES / "predictions" / f"{run}_{case}.npz", allow_pickle=True)


def meanstd(d, key):
    return d[key].mean(), (d[key].std() if len(d) > 1 else 0.0)


# ---------------------------------------------------------------- Fig. 1 framework
def fig01():
    fig, ax = plt.subplots(figsize=(DOUBLE, 72 * MM))
    ax.set_xlim(0, 4); ax.set_ylim(-0.05, 2.3); ax.axis("off")
    top = [("Hourly load profiles", "year 2007, net bus loads\nIEEE 14 and IEEE 30\n8 760 h per grid"),
           ("Load scenarios", "per-bus factors mapped\nto 0.6\u20131.0 \u00d7 nominal\nQ at constant p.f."),
           ("AC-OPF labels", "pandapower (PIPS)\nVm, Va, Pg, Qg\nflows, cost"),
           ("Source augmentation", "IEEE 14 only\nload (L), r/x/cost (P)\nN-1 outages (T)")]
    bot = [("Evaluation", "accuracy, power balance\nlimits, PF feasibility\nruntime, N-1"),
           ("Transfer to IEEE 30", "zero-shot, or fine-tune\nwith 1\u2013100 % of\ntarget training data"),
           ("Physics-informed GNN", "GATv2 + ER-biased\nglobal attention; MSE +\npower balance + limits"),
           ("Heterogeneous graphs", "bus, generator and\nbranch features (p.u.)\nER positional encoding")]
    w, h = 0.8, 0.8
    def box(x, y, title, sub, edge):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                    fc="#f6f6f4", ec=edge, lw=1.2))
        ax.text(x, y + 0.25, title, ha="center", va="center", fontsize=7, fontweight="bold", color=INK)
        ax.text(x, y - 0.09, sub, ha="center", va="center", fontsize=6.4, color=MUTED, linespacing=1.3)
    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=MUTED))
    xs = [0.5, 1.5, 2.5, 3.5]
    for i, (t, s) in enumerate(top):
        box(xs[i], 1.68, t, s, NEUTRAL)
        if i:
            arrow((xs[i - 1] + w / 2, 1.68), (xs[i] - w / 2, 1.68))
    for i, (t, s) in enumerate(bot):
        box(xs[i], 0.5, t, s, FT if i < 3 else NEUTRAL)
        if i:
            arrow((xs[i] - w / 2, 0.5), (xs[i - 1] + w / 2, 0.5))
    arrow((3.5, 1.68 - h / 2), (3.5, 0.5 + h / 2))
    ax.text(0.1, 2.16, "Data and labels", fontsize=7.5, color=MUTED, fontstyle="italic")
    ax.text(0.1, -0.04, "Learning, transfer and evaluation", fontsize=7.5, color=MUTED, fontstyle="italic")
    save(fig, "fig01_framework")


# ---------------------------------------------------------------- Fig. 2 grids
def fig02():
    import networkx as nx
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 82 * MM), gridspec_kw=dict(width_ratios=[1, 1.35]))
    for ax, case, title, letter in [(axes[0], "case14", "IEEE 14-bus (source, PGLib)", "a"),
                                    (axes[1], "case30", "IEEE 30-bus (target, MATPOWER)", "b")]:
        br = pd.read_csv(PROC / f"network_{case}_branches.csv", index_col=0)
        gens = pd.read_csv(PROC / f"network_{case}_gens.csv", index_col=0)
        sc = pd.read_parquet(PROC / f"load_scenarios_{case}.parquet", columns=None)
        load_buses = {int(c[5:]) for c in sc.columns if c.startswith("P_bus")}
        nb = int(max(br.from_bus.max(), br.to_bus.max()))
        g = nx.Graph()
        g.add_nodes_from(range(1, nb + 1))
        g.add_edges_from(zip(br.from_bus, br.to_bus))
        pos = nx.kamada_kawai_layout(g)
        for _, e in br.iterrows():
            (x1, y1), (x2, y2) = pos[e.from_bus], pos[e.to_bus]
            ax.plot([x1, x2], [y1, y2], color=NEUTRAL, lw=1.0, ls="--" if e.type == "trafo" else "-", zorder=1)
        slack = set(gens[gens.type == "ext_grid"].bus)
        disp = set(gens[(gens.type == "gen") & (gens.p_max_mw > 0)].bus)
        sync = set(gens[(gens.type == "gen") & (gens.p_max_mw <= 0)].bus)
        for b in range(1, nb + 1):
            x, y = pos[b]
            if b in slack:
                ax.scatter(x, y, s=70, marker="s", c=SC, edgecolors="white", linewidths=1.2, zorder=3)
            elif b in disp:
                ax.scatter(x, y, s=60, marker="s", c=FT, edgecolors="white", linewidths=1.2, zorder=3)
            elif b in sync:
                ax.scatter(x, y, s=60, marker="s", facecolors="white", edgecolors=FT, linewidths=1.4, zorder=3)
            else:
                filled = b in load_buses
                ax.scatter(x, y, s=34, marker="o", facecolors=INK if filled else "white",
                           edgecolors=INK, linewidths=1.0, zorder=3)
            nb_xy = np.array([pos[n] for n in g.neighbors(b)]).mean(axis=0)
            d = np.array([x, y]) - nb_xy
            d = d / (np.linalg.norm(d) + 1e-9)
            ax.text(x + 0.11 * d[0], y + 0.11 * d[1], str(b), ha="center", va="center", fontsize=6,
                    color=MUTED, zorder=4)
        ax.set_title(title, color=INK, loc="left")
        ax.axis("off"); ax.set_aspect("equal"); ax.set_anchor("N")
        ax.margins(0.08)
        panel(ax, letter)
    handles = [Line2D([], [], marker="s", ls="", color=SC, ms=6, label="slack generator"),
               Line2D([], [], marker="s", ls="", color=FT, ms=6, label="dispatchable generator"),
               Line2D([], [], marker="s", ls="", mfc="white", mec=FT, ms=6, label="synchronous condenser"),
               Line2D([], [], marker="o", ls="", color=INK, ms=5, label="load bus"),
               Line2D([], [], marker="o", ls="", mfc="white", mec=INK, ms=5, label="bus without load"),
               Line2D([], [], color=NEUTRAL, lw=1.0, label="line"),
               Line2D([], [], color=NEUTRAL, lw=1.0, ls="--", label="transformer")]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    save(fig, "fig02_grids")


# ---------------------------------------------------------------- Fig. 3 load profile
def fig03():
    split = pd.read_csv(PROC / "splits.csv", index_col=0, parse_dates=True).split
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE, 92 * MM), gridspec_kw=dict(height_ratios=[1.35, 1]))
    nominal = {"case14": 259.0, "case30": 189.2}
    for case, color, label in [("case14", FT, "IEEE 14"), ("case30", SC, "IEEE 30")]:
        sc = pd.read_parquet(PROC / f"load_scenarios_{case}.parquet")
        sc.index = pd.to_datetime(sc.index)
        tot = sc.filter(like="P_bus").sum(axis=1) / nominal[case] * 100
        day = tot.resample("D")
        axes[0].plot(day.mean().index, day.mean().values, color=color, lw=1.2, label=f"{label} (daily mean)")
        if case == "case30":
            axes[0].fill_between(day.min().index, day.min().values, day.max().values, color=color,
                                 alpha=0.15, lw=0, label="IEEE 30 (daily min\u2013max)")
        wk = tot["2007-12-03":"2007-12-09 23:00"]
        axes[1].plot(wk.index, wk.values, color=color, lw=1.2, label=label)
    spans = [("2007-01-01", "2007-08-31 23:00", "#2a78d6", "training (Jan\u2013Aug)"),
             ("2007-09-01", "2007-10-31 23:00", "#eda100", "validation (Sep\u2013Oct)"),
             ("2007-11-01", "2007-12-31 23:00", "#eb6834", "test (Nov\u2013Dec)")]   # chronological split
    for a, b, color, txt in spans:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        axes[0].axvspan(a, b, color=color, alpha=0.07, lw=0)
        axes[0].text(a + (b - a) / 2, 1.0, txt, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=7, color=MUTED)
    axes[0].set_ylabel("Total load (% of nominal)")
    axes[0].set_ylim(50, 103)
    axes[0].legend(loc="lower left", ncol=3)
    axes[1].set_ylabel("Total load (% of nominal)")
    axes[1].set_title("One test week, 3\u20139 December 2007 (hourly)", loc="left", color=INK)
    axes[1].set_ylim(52, 103)
    axes[1].legend(loc="lower center", ncol=2)
    import matplotlib.dates as mdates
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    for ax, l in zip(axes, "ab"):
        style(ax); panel(ax, l)
    fig.tight_layout()
    save(fig, "fig03_load_profile")


# ---------------------------------------------------------------- Fig. 4 distribution shift
def fig04():
    d14, d30 = tensors("case14", "train"), tensors("case30", "train")
    def pick(ds, what):
        t, i = ds.t, ds.idx
        if what == "pd":
            x = t["pd"][i].numpy().ravel() * 100; return x[x > 1e-6]
        if what == "pe":
            return t["er_pe"][:, 2].numpy()
        if what == "vm":
            return t["vm"][i].numpy().ravel()
        if what == "va":
            return np.rad2deg(t["va"][i].numpy().ravel())
        if what == "pg":
            gx = t["gen_x"]; disp = (gx[:, 1] > gx[:, 0]).numpy()
            return t["pg"][i].numpy()[:, disp].ravel() * 100
    items = [("pd", "Bus load (MW)"), ("pe", "ER-PE mean (\u2013)"), ("vm", "Vm label (pu)"),
             ("va", "Va label (deg)"), ("pg", "Pg label (MW)")]
    fig, axes = plt.subplots(1, 5, figsize=(DOUBLE, 52 * MM))
    for ax, (k, lab), l in zip(axes, items, "abcde"):
        bp = ax.boxplot([pick(d14, k), pick(d30, k)], widths=0.55, showfliers=False, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.2), whiskerprops=dict(color=MUTED),
                        capprops=dict(color=MUTED), boxprops=dict(lw=1.0))
        for patch, c in zip(bp["boxes"], [FT, SC]):
            patch.set_facecolor(c); patch.set_alpha(0.55); patch.set_edgecolor(c)
        ax.set_xticks([1, 2]); ax.set_xticklabels(["IEEE 14", "IEEE 30"])
        ax.set_title(lab, loc="left", color=INK)
        style(ax); panel(ax, l)
    fig.tight_layout()
    save(fig, "fig04_distribution")


# ---------------------------------------------------------------- Fig. 5 training curves
def fig05():
    n_seeds = 0
    fig, axes = plt.subplots(1, 2, figsize=(ONEHALF, 55 * MM), sharey=True)
    for ax, f, l in zip(axes, ["0.05", "1.0"], "ab"):
        for kind, color, lab in [("ft30", FT, "Fine-tune (from IEEE 14)"), ("scratch30", SC, "From scratch")]:
            hs = []
            for s in range(5):
                p = RES / f"{kind}_f{f}_s{s}" / "history.csv"
                if p.exists():
                    hs.append(pd.read_csv(p).set_index("step")["val"])
            if not hs:
                continue
            h = pd.concat(hs, axis=1)
            ax.fill_between(h.index, h.min(axis=1), h.max(axis=1), color=color, alpha=0.15, lw=0)
            ax.plot(h.index, h.median(axis=1), color=color, lw=1.4, label=lab)
            n_seeds = h.shape[1]
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_title(f"{float(f) * 100:.0f} % of IEEE 30 training data", loc="left", color=INK)
        style(ax, "both"); panel(ax, l)
    axes[0].set_ylabel("Validation loss")
    hd, lb = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(hd, lb, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig05_training_curves")


# ---------------------------------------------------------------- Fig. 7 parity
def fig07():
    rows = [("ft30_f0.05_s0", "Fine-tune, 5 % data"), ("scratch30_f1.0_s0", "From scratch, 100 % data")]
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE, 92 * MM))
    for r, (run, rlab) in enumerate(rows):
        z = pred(run)
        disp = z["gen_pmax"] > z["gen_pmin"]
        items = [("vm", "Vm (pu)", 1, None), ("va", "Va (deg)", 180 / np.pi, None),
                 ("pg", "Pg (MW)", 100, disp), ("qg", "Qg (Mvar)", 100, None)]
        for c, (k, lab, scale, mask) in enumerate(items):
            ax = axes[r, c]
            t, p = z[f"true_{k}"] * scale, z[f"pred_{k}"] * scale
            if mask is not None:
                t, p = t[:, mask], p[:, mask]
            keep = t.std(axis=0) > 1e-9          # drop constant columns (slack-bus angle)
            t, p = t[:, keep].ravel(), p[:, keep].ravel()
            lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
            ax.hexbin(t, p, gridsize=55, bins="log", cmap="Blues", mincnt=1, linewidths=0, extent=(lo, hi, lo, hi))
            ax.plot([lo, hi], [lo, hi], color=SC, lw=1.0, ls="--")
            mae = np.abs(p - t).mean()
            p99 = np.percentile(np.abs(p - t), 99)
            ax.text(0.97, 0.04, f"MAE {sci(mae)}\nP99 {sci(p99)}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=6.8, color=INK)   # lower right: clear of the diagonal
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            if r == 1:
                ax.set_xlabel(f"True {lab}")
            if c == 0:
                ax.set_ylabel(f"{rlab}\npredicted")
            if r == 0:
                ax.set_title(lab, loc="left", color=INK)
            style(ax, None)
    fig.tight_layout()
    save(fig, "fig07_parity")


# ---------------------------------------------------------------- Fig. 8 bus profile
def fig08():
    models = [("ft30_f0.05_s0", FT, "o", "Fine-tune, 5 %"), ("scratch30_f1.0_s0", SC, "s", "From scratch, 100 %"),
              ("scratch30_f1.0_s0_mlp", MLPC, "^", "MLP, 100 %")]
    base = pred(models[0][0])
    h = int(np.argmax(base["true_cost"]))
    nb = base["true_vm"].shape[1]
    buses = np.arange(1, nb + 1)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE, 88 * MM), sharex=True)
    for col, (k, lab, scale) in enumerate([("vm", "Vm (pu)", 1), ("va", "Va (deg)", 180 / np.pi)]):
        axes[0, col].plot(buses, base[f"true_{k}"][h] * scale, color=INK, lw=1.4, marker="o", ms=3, label="AC-OPF")
        for run, color, mk, mlab in models:
            if not (RES / "predictions" / f"{run}_case30.npz").exists():
                continue
            z = pred(run)
            axes[0, col].plot(buses, z[f"pred_{k}"][h] * scale, color=color, lw=0, marker=mk, ms=3.5, label=mlab)
            axes[1, col].plot(buses, (z[f"pred_{k}"][h] - z[f"true_{k}"][h]) * scale, color=color, lw=1.0,
                              marker=mk, ms=3, label=mlab)
        axes[1, col].axhline(0, color=MUTED, lw=1.0)
        axes[0, col].set_title(lab, loc="left", color=INK)
        axes[1, col].set_title(f"Prediction error, {lab}", loc="left", color=INK)
        axes[1, col].set_xlabel("Bus")
    for ax, l in zip(axes.flat, "abcd"):
        style(ax); panel(ax, l)
    hd, lb = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.legend(hd, lb, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.965))
    print(f"  (for the legend) highest-cost test hour: {str(base['timestamps'][h])[:16]}")
    save(fig, "fig08_bus_profile")


# ---------------------------------------------------------------- Fig. 9 test week
def fig09():
    models = [("ft30_f0.05_s0", FT, "Fine-tune, 5 %"), ("scratch30_f1.0_s0", SC, "From scratch, 100 %"),
              ("scratch30_f1.0_s0_mlp", MLPC, "MLP, 100 %")]
    base = pred(models[0][0])
    ts = pd.to_datetime(base["timestamps"].astype(str))
    sel = (ts >= "2007-12-03") & (ts < "2007-12-10")
    if sel.sum() < 24:                                   # week not in the test set: first 168 test hours
        sel = np.zeros(len(ts), bool); sel[:168] = True
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE, 82 * MM), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.4, 1]))
    axes[0].plot(ts[sel], base["true_cost"][sel], color=INK, lw=1.6, label="AC-OPF")
    for run, color, lab in models:
        if not (RES / "predictions" / f"{run}_case30.npz").exists():
            continue
        z = pred(run)
        axes[0].plot(ts[sel], z["pred_cost"][sel], color=color, lw=1.0, ls="--", label=lab)
        err = (z["pred_cost"][sel] - z["true_cost"][sel]) / z["true_cost"][sel] * 100
        axes[1].plot(ts[sel], err, color=color, lw=1.0, label=lab)
    axes[1].axhline(0, color=MUTED, lw=1.0)
    axes[0].set_ylabel("Generation cost ($/h)")
    axes[1].set_ylabel("Cost error (%)")
    hd, lb = axes[0].get_legend_handles_labels()
    import matplotlib.dates as mdates
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
    for ax, l in zip(axes, "ab"):
        style(ax); panel(ax, l)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.legend(hd, lb, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    save(fig, "fig09_test_week")


# ---------------------------------------------------------------- Fig. 10 violations
def fig10():
    df = runs_df()
    df = df[df.case == "case30"]
    pf = pd.read_csv(RES / "pf_postprocess_runs.csv")
    models = [("fine-tune", 0.01, "Fine-tune 1 %", "Fine-tune 1 % (E3)", FT),
              ("fine-tune", 0.05, "Fine-tune 5 %", "Fine-tune 5 % (E3)", FT),
              ("fine-tune", 1.0, "Fine-tune 100 %", "Fine-tune 100 % (E3)", FT),
              ("scratch", 0.05, "Scratch 5 %", "From scratch 5 % (E4)", SC),
              ("scratch", 1.0, "Scratch 100 %", "From scratch 100 % (E4)", SC),
              (None, None, "DC-OPF", "DC-OPF baseline (case30)", NEUTRAL)]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE, 58 * MM))
    y = np.arange(len(models))
    for j, (key, title) in enumerate([("samples_with_branch_viol_pct", "Hours with a branch overload,\nbefore PF (%)"),
                                      ("pb_p_mean_mw", "Mean |\u0394P| mismatch,\nbefore PF (MW)"),
                                      ("infeasible", "Hours not fully feasible,\nafter PF (%)")]):
        ax = axes[j]
        for i, (m, f, lab, grp, color) in enumerate(models):
            if key == "infeasible":
                d = pf[pf.group == grp]
                if not len(d):
                    ax.text(0, i, " n/a", va="center", fontsize=6.8, color=MUTED)
                    continue
                v = 100 - d.feasible_pct
                mu, sd = v.mean(), (v.std() if len(v) > 1 else 0)
            else:
                d = df[(df.method == m) & np.isclose(df.fraction, f)] if m is not None else df.iloc[:0]
                if not len(d):
                    ax.text(0, i, " n/a", va="center", fontsize=6.8, color=MUTED)
                    continue
                mu, sd = meanstd(d, key)
            ax.barh(i, mu, xerr=sd, color=color, height=0.6, error_kw=dict(elinewidth=1.0, capsize=2, ecolor=MUTED))
            ax.text(mu + sd, i, f" {mu:.3g}", va="center", fontsize=6.8, color=INK)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.15)       # room for the value labels
        ax.set_yticks(y); ax.set_yticklabels([m[2] for m in models] if j == 0 else [])
        ax.set_ylim(len(models) - 0.5, -0.5)
        ax.set_title(title, loc="left", color=INK)
        style(ax, "x"); panel(ax, "abc"[j])
    fig.tight_layout()
    save(fig, "fig10_violations")


# ---------------------------------------------------------------- Fig. 12 runtime
def fig12():
    rt = pd.read_csv(RES / "runtime_benchmark.csv", index_col=0)
    pf = pd.read_csv(RES / "pf_postprocess_runs.csv")
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE, 55 * MM), sharey=True)
    for ax, case, l in [(axes[0], "case14", "a"), (axes[1], "case30", "b")]:
        r = rt[case].astype(float)
        pf_ms = pf[(pf.case == case) & (pf.run != "oracle") & (pf.run != "dcopf")].pf_ms.mean()
        dc = pf[(pf.case == case) & (pf.run == "dcopf")]
        items = [("AC-OPF solver (pandapower)", r["solver_ms_median"], INK),
                 ("DC-OPF (pandapower)", dc["dc_dc_ms"].iloc[0] if len(dc) and "dc_dc_ms" in dc else np.nan, NEUTRAL),
                 ("GNN + power flow (CPU)", r.get("gnn_cpu_single_ms_median", np.nan) + pf_ms, FT),
                 ("GNN, single sample, GPU", r.get("gnn_gpu_single_ms_median", np.nan), FT),
                 ("GNN, single sample, CPU", r.get("gnn_cpu_single_ms_median", np.nan), FT),
                 ("GNN, batched, GPU (per sample)", r.get("gnn_gpu_batched_ms_per_sample", np.nan), FT)]
        for i, (lab, v, color) in enumerate(items):
            if not np.isfinite(v):
                continue
            ax.plot([1e-3, v], [i, i], color=GRID, lw=1.2, zorder=1)
            ax.scatter(v, i, s=36, color=color, zorder=3)
            speed = r["solver_ms_median"] / v
            ax.text(v * 1.35, i, f"{v:.3g} ms" + ("" if i == 0 else f"  ({speed:,.0f}\u00d7)"),
                    va="center", fontsize=6.8, color=INK)
        ax.set_xscale("log"); ax.set_xlim(0.02, 3000)
        ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items])
        ax.invert_yaxis()
        ax.set_xlabel("Time per OPF sample (ms, log scale)")
        ax.set_title({"case14": "IEEE 14", "case30": "IEEE 30"}[case], loc="left", color=INK)
        style(ax, "x"); panel(ax, l)
    fig.tight_layout()
    save(fig, "fig12_runtime")


# ---------------------------------------------------------------- Fig. 13 N-1
def fig13():
    rows = []
    n1r = pd.read_csv(RES / "n1_runs.csv")
    n1r = n1r[np.isclose(n1r.fraction, 0.25)]
    for model, method, lab in [("proposed", "fine-tune", "GNN fine-tune"), ("proposed", "scratch", "GNN from scratch"),
                               ("MLP", "scratch", "MLP topology-blind")]:
        for _, x in n1r[(n1r.model == model) & (n1r.method == method)].iterrows():
            rows.append(dict(scen="No outage data", model=lab, pg=x.n1_pg_mae_mw, pb=x.n1_pb_p_mean_mw))
    for tag, kind, lab in [("n1", "ft30", "GNN fine-tune"), ("n1", "scratch30", "GNN from scratch"),
                           ("n1mlp", "scratch30", "MLP topology-blind"), ("n1mlptopo", "scratch30", "MLP + outage one-hot")]:
        for f in sorted(RES.glob(f"{kind}_f0.25_s?_{tag}/metrics.json")):
            for m in json.load(open(f)):
                if m["case"] == "case30_n1":
                    rows.append(dict(scen="With outage data", model=lab, pg=m["pg_mae_mw"], pb=m["pb_p_mean_mw"]))
    un = pd.read_csv(RES / "unseen_runs.csv")
    un = un[un.subset == "unseen"]
    for src, lab in [("GNN fine-tune, outages held out", "GNN fine-tune"),
                     ("GNN from scratch, outages held out", "GNN from scratch"),
                     ("MLP one-hot, outages held out (raw)", "MLP + outage one-hot"),
                     ("MLP one-hot, outages held out (unseen -> intact)", "MLP one-hot, unseen \u2192 intact")]:
        for _, x in un[un.model == src].iterrows():
            rows.append(dict(scen="Unseen outages", model=lab, pg=x.pg_mae_mw, pb=x.pb_p_mean_mw))
    floor_un = un[un.model.str.contains("floor")]
    df = pd.DataFrame(rows)
    floor = n1r[(n1r.model == "MLP") & (n1r.method == "scratch")]
    floors = {"No outage data": (floor.n1_pg_mae_mw.mean(), floor.n1_pb_p_mean_mw.mean()),
              "With outage data": (floor.n1_pg_mae_mw.mean(), floor.n1_pb_p_mean_mw.mean()),
              "Unseen outages": (floor_un.pg_mae_mw.mean(), floor_un.pb_p_mean_mw.mean())}
    colors = {"GNN fine-tune": FT, "GNN from scratch": SC, "MLP topology-blind": MLPC,
              "MLP + outage one-hot": VIO, "MLP one-hot, unseen \u2192 intact": MAG}
    scens = ["No outage data", "With outage data", "Unseen outages"]
    models = list(colors)
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 82 * MM), sharex="row", sharey=True)
    for r, (key, lab) in enumerate([("pg", "Pg MAE (MW)"), ("pb", "Mean |\u0394P| (MW)")]):
        for c, sname in enumerate(scens):
            ax = axes[r, c]
            d = df[df.scen == sname] if len(df) else df
            for i, m in enumerate(models):
                v = d[d.model == m][key] if len(d) else []
                if len(v):
                    ax.errorbar(v.mean(), i, xerr=v.std() if len(v) > 1 else 0, fmt="o", color=colors[m], ms=5,
                                elinewidth=1, capsize=2)
                else:
                    ax.text(0.02, i, "\u2013", transform=ax.get_yaxis_transform(), va="center", color=NEUTRAL)
            if np.isfinite(floors[sname][r]):
                ax.axvline(floors[sname][r], color=MUTED, lw=1.0, ls="--")
            ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
            ax.set_ylim(len(models) - 0.5, -0.5)
            if r == 0:
                ax.set_title(sname, loc="left", color=INK)
            ax.set_xlabel(lab)
            style(ax, "x"); panel(ax, "abcdef"[r * 3 + c])
        vals = list(df[key]) + [floors[k][r] for k in scens] if len(df) else [1.0]
        g = df.groupby(["scen", "model"])[key] if len(df) else None
        hi = np.nanmax(vals + (list((g.mean() + g.std().fillna(0)).values) if g is not None else []))
        axes[r, 0].set_xlim(0, 1.12 * hi)
    fig.tight_layout()
    pass                                  # notes go into the figure legend
    save(fig, "fig13_n1")


SCIREP = {  # final name : source file (results/figures/paper unless a path is given)
    # numbering of the revised manuscript: ten main-text figures, three supplementary figures
    "Fig1": "fig01_framework", "Fig2": "fig02_grids", "Fig3": "fig03_load_profile",
    "Fig4": "fig04_distribution", "Fig5": "../transfer_curves", "Fig6": "fig09_test_week",
    "Fig7": "../ablation", "Fig8": "fig10_violations", "Fig9": "fig12_runtime", "Fig10": "fig13_n1",
    "SupplementaryFigS1": "fig05_training_curves", "SupplementaryFigS2": "fig07_parity",
    "SupplementaryFigS3": "fig08_bus_profile"}


def export_scirep():
    import shutil
    dst = RES / "figures" / "scirep"
    dst.mkdir(parents=True, exist_ok=True)
    for name, src in SCIREP.items():
        for ext in ("pdf", "png"):
            f = (OUT / f"{src}.{ext}").resolve()
            if f.exists():
                shutil.copy(f, dst / f"{name}.{ext}")
            else:
                print(f"  missing {f.name} for {name} (run make_figures.py first for Fig5/Fig7)")
    print(f"Scientific Reports file names in {dst}")


if __name__ == "__main__":
    for f in [fig01, fig02, fig03, fig04, fig05, fig07, fig08, fig09, fig10, fig12, fig13]:
        print(f"{f.__name__}:")
        try:
            f()
        except Exception as e:
            print(f"  SKIPPED ({type(e).__name__}: {e})")
            traceback.print_exc(limit=1)
    print(f"\nFigures in {OUT}")
    export_scirep()
