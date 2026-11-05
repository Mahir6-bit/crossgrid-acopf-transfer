"""
28_stats_ci.py   (revision: replaces stats_tests.py for the paper; stats_tests.py can stay)
Paired, seed-matched comparisons on the IEEE 30 TEST set with everything a reviewer asks for:

  per comparison and metric
    n_seeds, mean_A, mean_B, rel_diff_pct (= mean_A / mean_B - 1)
    mean_diff, ci95_low, ci95_high      95 % CI of the paired difference A - B (Student t, n-1 dof)
    rel_ci95_low, rel_ci95_high         the same CI divided by mean_B (in %)
    d_z                                 standardized paired effect size = mean(diff) / sd(diff)
    A_better_in_seeds                   e.g. "5/5"
    p_value (two-sided paired t-test), p_holm (Holm-Bonferroni within the family), sig_holm_0.05
    p_wilcoxon                          exact Wilcoxon signed-rank p (n<6 cannot go below 0.0625; reported
                                        for transparency only)
Families (computed when the runs exist; ALL common seeds are used, e.g. 0-9 after run_more_seeds.ps1):
  E3 vs E4   fine-tune vs from scratch at every fraction
  headline   fine-tune 5 % vs from scratch 100 %
  FT vs MLP  proposed fine-tune vs grid-specific MLP from scratch at every fraction
  baselines  proposed vs GCN / GAT / MLP at the same method and fraction
  E5         ablation variants vs the full model (zero-shot, FT 1 %, FT 5 %)

Run from the project folder (after summarize_results.py):
    python scripts\\28_stats_ci.py
Writes results\\stats_ci.csv (all metrics) and results\\tables\\table_S2_stats_ci.csv (three key metrics,
formatted for Supplementary Table S2).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path(__file__).resolve().parents[1] / "results"
METRICS = ["pg_mae_mw", "va_mae_deg", "vm_mae_pu", "pb_p_mean_mw", "pb_p_max_mw", "cost_gap_abs_pct"]  # same six as stats_tests.py
KEY = {"pg_mae_mw": "Pg MAE", "pb_p_mean_mw": "mean |ΔP|", "cost_gap_abs_pct": "|cost gap|"}


def paired(x, y):
    """Statistics of the paired difference x - y (arrays of equal length >= 2)."""
    d = np.asarray(x, float) - np.asarray(y, float)
    n = len(d)
    mean_d, sd_d = d.mean(), d.std(ddof=1)
    half = stats.t.ppf(0.975, n - 1) * sd_d / np.sqrt(n) if n > 1 else np.nan
    t = stats.ttest_rel(x, y)
    try:
        pw = stats.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided").pvalue
    except ValueError:
        pw = np.nan
    mb = np.mean(y)
    return dict(n_seeds=n, mean_A=np.mean(x), mean_B=mb, rel_diff_pct=(np.mean(x) / mb - 1) * 100,
                mean_diff=mean_d, ci95_low=mean_d - half, ci95_high=mean_d + half,
                rel_ci95_low=(mean_d - half) / mb * 100, rel_ci95_high=(mean_d + half) / mb * 100,
                d_z=mean_d / sd_d if sd_d > 0 else np.nan,
                A_better_in_seeds=f"{int((d < 0).sum())}/{n}", p_value=t.pvalue, p_wilcoxon=pw)


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m, adj, running = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def compare(family, label_a, a, label_b, b, metrics=METRICS):
    """a, b: DataFrames indexed by seed. Returns one row per metric over the common seeds."""
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 2:
        return []
    rows = []
    for m in metrics:
        if m not in a.columns or m not in b.columns:
            continue
        r = paired(a.loc[common, m].to_numpy(float), b.loc[common, m].to_numpy(float))
        rows.append(dict(family=family, A=label_a, B=label_b, metric=m, seeds=",".join(map(str, common)), **r))
    return rows


def finish(rows):
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["p_holm"] = np.nan
    for _, idx in res.groupby("family").groups.items():
        res.loc[idx, "p_holm"] = holm(res.loc[idx, "p_value"].to_numpy())
    res["sig_holm_0.05"] = res.p_holm < 0.05
    res["sig_uncorrected_only"] = (res.p_value < 0.05) & ~res["sig_holm_0.05"]
    return res


def s2_table(res):
    k = res[res.metric.isin(KEY)].copy()
    fmt = lambda v: f"{v:+.1f}"
    return pd.DataFrame({
        "Family": k.family, "A": k.A, "B": k.B, "Metric": k.metric.map(KEY), "n": k.n_seeds,
        "Mean A": k.mean_A.map("{:.3g}".format), "Mean B": k.mean_B.map("{:.3g}".format),
        "Δ (%)": k.rel_diff_pct.map(fmt),
        "95% CI of Δ (%)": [f"[{lo:+.1f}, {hi:+.1f}]" for lo, hi in zip(k.rel_ci95_low, k.rel_ci95_high)],
        "d_z": k.d_z.map("{:+.2f}".format), "A better": k.A_better_in_seeds,
        "p": k.p_value.map("{:.3g}".format), "p (Holm)": k.p_holm.map("{:.3g}".format),
        "Sig.": np.where(k["sig_holm_0.05"], "yes", np.where(k["sig_uncorrected_only"], "uncorr. only", "")),
    })


if __name__ == "__main__":
    df = pd.read_csv(RES / "all_runs.csv")
    df = df[df.case == "case30"]

    def pick(method, fraction, seeds=None):
        d = df[(df.method == method) & np.isclose(df.fraction, fraction)]
        if seeds is not None:
            d = d[d.seed.isin(seeds)]
        return d.drop_duplicates("seed").set_index("seed")

    rows = []
    for f in sorted(df[df.method == "fine-tune"].fraction.unique()):
        rows += compare("E3 vs E4", f"fine-tune {f:.0%}", pick("fine-tune", f), f"scratch {f:.0%}", pick("scratch", f))
    rows += compare("headline", "fine-tune 5%", pick("fine-tune", 0.05), "scratch 100%", pick("scratch", 1.0))
    variants = sorted({m.split("[")[1].rstrip("]") for m in df.method.unique() if "[" in m})
    if "mlp" in variants:
        for f in sorted(df[df.method == "scratch [mlp]"].fraction.unique()):
            rows += compare("FT vs MLP", f"proposed fine-tune {f:.0%}", pick("fine-tune", f),
                            f"MLP scratch {f:.0%}", pick("scratch [mlp]", f))
    for v in [v for v in variants if v in ("gcn", "gat", "mlp")]:
        for method, fr in [("zero-shot (E2)", 0.0), ("fine-tune", 0.05), ("fine-tune", 1.0),
                           ("scratch", 0.05), ("scratch", 1.0)]:
            rows += compare("baselines", f"proposed ({method} {fr:.0%})", pick(method, fr),
                            f"{v} ({method} {fr:.0%})", pick(f"{method} [{v}]", fr))
    for v in [v for v in variants if v in ("nope", "noattn", "nophys", "augnone", "augL", "augLP")]:
        rows += compare("E5 ablation", "full (zero-shot)", pick("zero-shot (E2)", 0.0),
                        f"{v} (zero-shot)", pick(f"zero-shot (E2) [{v}]", 0.0))
        for f in (0.01, 0.05):
            rows += compare("E5 ablation", f"full (FT {f:.0%})", pick("fine-tune", f),
                            f"{v} (FT {f:.0%})", pick(f"fine-tune [{v}]", f))
    res = finish(rows)
    if res.empty:
        raise SystemExit("No comparable runs found (run summarize_results.py first).")
    res.to_csv(RES / "stats_ci.csv", index=False)
    (RES / "tables").mkdir(exist_ok=True)
    s2_table(res).to_csv(RES / "tables" / "table_S2_stats_ci.csv", index=False)
    show = res[res.metric.isin(KEY)][["family", "A", "B", "metric", "n_seeds", "rel_diff_pct", "rel_ci95_low",
                                      "rel_ci95_high", "d_z", "A_better_in_seeds", "p_holm", "sig_holm_0.05"]]
    with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.3g}".format):
        print(show.to_string(index=False))
    print(f"\n{len(res)} tests, {int(res['sig_holm_0.05'].sum())} significant after Holm correction; "
          f"{int(res['sig_uncorrected_only'].sum())} significant only before correction.")
    print(f"Saved {RES / 'stats_ci.csv'} and {RES / 'tables' / 'table_S2_stats_ci.csv'}")
