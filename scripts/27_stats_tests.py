"""
27_stats_tests.py
Paired, seed-matched significance tests for the paper's main claims (IEEE 30 TEST set).

Pairing: runs with the same seed share the same data subset (and, for fine-tuning, the
source model of that seed), so A_seed - B_seed differences are paired.
Test: two-sided paired t-test per metric; Holm correction within each comparison family.
Small n (3-5 seeds) gives low power - report effect sizes (relative difference) together
with p-values, and do not over-interpret p ~ 0.05-0.1.

Comparison families (only those whose runs exist are computed):
  E3 vs E4   fine-tune vs from scratch at every fraction                    (5 seeds)
  headline   fine-tune 5 % vs from scratch 100 %                           (5 seeds)
  E5         every ablation variant vs the full model, zero-shot / 1 % / 5 % (seeds 0-2)
  baselines  proposed model vs GCN / GAT / MLP at the same method and fraction (seeds 0-2)
  FT vs MLP  proposed fine-tune vs grid-specific MLP from scratch at every fraction (seeds 0-2)

Run from the project folder (after summarize_results.py):
    python scripts\\27_stats_tests.py
Writes results\\stats_tests.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path(__file__).resolve().parents[1] / "results"
METRICS = ["pg_mae_mw", "va_mae_deg", "vm_mae_pu", "pb_p_mean_mw", "pb_p_max_mw", "cost_gap_abs_pct"]

df = pd.read_csv(RES / "all_runs.csv")
df = df[df.case == "case30"]


def pick(method, fraction, seeds=None):
    d = df[(df.method == method) & np.isclose(df.fraction, fraction)]
    if seeds is not None:
        d = d[d.seed.isin(seeds)]
    return d.set_index("seed")


def compare(family, label_a, a, label_b, b):
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 2:
        return []
    out = []
    for m in METRICS:
        x, y = a.loc[common, m].to_numpy(float), b.loc[common, m].to_numpy(float)
        t, p = stats.ttest_rel(x, y)
        out.append(dict(family=family, A=label_a, B=label_b, metric=m, n_seeds=len(common),
                        mean_A=x.mean(), mean_B=y.mean(),
                        rel_diff_pct=(x.mean() / y.mean() - 1) * 100,
                        A_better_in_seeds=f"{int((x < y).sum())}/{len(common)}", p_value=p))
    return out


rows = []
fracs = sorted(df[df.method == "fine-tune"].fraction.unique())
for f in fracs:
    rows += compare("E3 vs E4", f"fine-tune {f:.0%}", pick("fine-tune", f),
                    f"scratch {f:.0%}", pick("scratch", f))
rows += compare("headline", "fine-tune 5%", pick("fine-tune", 0.05), "scratch 100%", pick("scratch", 1.0))

seeds3 = [0, 1, 2]
variants = sorted({m.split("[")[1].rstrip("]") for m in df.method.unique() if "[" in m})
ablations = [v for v in variants if v in ("nope", "noattn", "nophys", "augnone", "augL", "augLP")]
baselines = [v for v in variants if v in ("gcn", "gat", "mlp")]
for v in ablations:
    rows += compare("E5 ablation", "full (zero-shot)", pick("zero-shot (E2)", 0.0, seeds3),
                    f"{v} (zero-shot)", pick(f"zero-shot (E2) [{v}]", 0.0, seeds3))
    for f in (0.01, 0.05):
        rows += compare("E5 ablation", f"full (FT {f:.0%})", pick("fine-tune", f, seeds3),
                        f"{v} (FT {f:.0%})", pick(f"fine-tune [{v}]", f, seeds3))
for v in baselines:
    for method, fr in [("zero-shot (E2)", 0.0), ("fine-tune", 0.05), ("fine-tune", 1.0),
                       ("scratch", 0.05), ("scratch", 1.0)]:
        rows += compare("baselines", f"proposed ({method} {fr:.0%})", pick(method, fr, seeds3),
                        f"{v} ({method} {fr:.0%})", pick(f"{method} [{v}]", fr, seeds3))

# transferable GNN (fine-tuned) vs grid-specific MLP trained from scratch on the same target data
if "mlp" in baselines:
    for f in sorted(df[df.method == "scratch [mlp]"].fraction.unique()):
        rows += compare("FT vs MLP", f"proposed fine-tune {f:.0%}", pick("fine-tune", f, seeds3),
                        f"MLP scratch {f:.0%}", pick("scratch [mlp]", f, seeds3))

res = pd.DataFrame(rows)
if res.empty:
    raise SystemExit("No comparable runs found.")
# Holm-Bonferroni correction within each comparison family
def holm(p):
    order = np.argsort(p)
    m, adj, running = len(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


res["p_holm"] = np.nan
for fam, idx in res.groupby("family").groups.items():
    res.loc[idx, "p_holm"] = holm(res.loc[idx, "p_value"].to_numpy())
res["sig_holm_0.05"] = res.p_holm < 0.05
res.to_csv(RES / "stats_tests.csv", index=False)

with pd.option_context("display.width", 250, "display.max_rows", 500, "display.float_format", "{:.4g}".format):
    for fam, d in res.groupby("family", sort=False):
        print(f"\n=== {fam} ===  (A vs B; rel_diff = A/B - 1; negative = A has lower error)")
        print(d.drop(columns="family").to_string(index=False))
print(f"\n{len(res)} tests; {int(res['sig_holm_0.05'].sum())} significant after Holm correction (within family). "
      f"Saved {RES / 'stats_tests.csv'}")
