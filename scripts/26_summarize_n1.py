"""
26_summarize_n1.py
N-1 adaptation experiment: models trained on IEEE 30 hours that include, for every training
hour, the intact case AND one N-1 case (tags _n1, _n1mlp, _n1mlptopo), evaluated on the
IEEE 30 N-1 test set and on the intact test set. Models trained without any N-1 data
(from results\\n1_runs.csv, all available seeds) are shown as reference.
Paired tests use only the seeds both models have.

Run from the project folder:
    python scripts\\26_summarize_n1.py
Writes results\\n1_adaptation_summary.csv and results\\n1_adaptation_stats.csv
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RES = Path(__file__).resolve().parents[1] / "results"
KEYS = ["pg_mae_mw", "va_mae_deg", "vm_mae_pu", "pb_p_mean_mw", "pb_p_max_mw", "branch_viol_pct",
        "cost_gap_abs_pct"]
NAMES = {("ft30", "n1"): "Proposed GNN, fine-tune (+N-1 data)",
         ("scratch30", "n1"): "Proposed GNN, from scratch (+N-1 data)",
         ("scratch30", "n1mlp"): "MLP, topology-blind (+N-1 data)",
         ("scratch30", "n1mlptopo"): "MLP + outage one-hot (+N-1 data)"}

rows = []
for f in RES.glob("*/metrics.json"):
    p = f.parent.name.split("_")
    if p[0] not in ("ft30", "scratch30") or len(p) < 4 or (p[0], p[3]) not in NAMES:
        continue
    for m in json.load(open(f)):
        test = {"case30": "intact", "case30_n1": "N-1"}.get(m["case"])
        if test:
            rows.append(dict(model=NAMES[(p[0], p[3])], fraction=float(p[1][1:]), seed=int(p[2][1:]),
                             test=test, **{k: m[k] for k in KEYS}))
ref = RES / "n1_runs.csv"
if ref.exists():
    r = pd.read_csv(ref)
    r = r[r.method.isin(["fine-tune", "scratch"]) & r.model.isin(["proposed", "MLP"])]
    for _, x in r.iterrows():
        name = {"proposed": f"Proposed GNN, {x.method} (no N-1 data)", "MLP": "MLP, topology-blind (no N-1 data)"}[x.model]
        if x.model == "MLP" and x.method != "scratch":
            continue
        for test, pre in [("N-1", "n1_"), ("intact", "intact_")]:
            rows.append(dict(model=name, fraction=x.fraction, seed=x.seed, test=test,
                             **{k: x.get(pre + k, np.nan) for k in KEYS}))
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No N-1 adaptation runs found (tags _n1, _n1mlp, _n1mlptopo).")
fr = sorted(df[df.model.str.contains(r"\+N-1")].fraction.unique())
df = df[df.fraction.isin(fr)]

g = df.groupby(["test", "fraction", "model"])
mean, std = g[KEYS].mean(), g[KEYS].std()
table = mean.astype(object)
for k in KEYS:
    table[k] = [f"{m:.4g} ± {s:.2g}" if pd.notna(s) else f"{m:.4g}" for m, s in zip(mean[k], std[k])]
table.insert(0, "n_seeds", g.size())
table.to_csv(RES / "n1_adaptation_summary.csv")

# paired tests on the N-1 test set: fine-tuned GNN (+N-1) vs every other model, same seeds
st = []
for f in fr:
    d = df[(df.test == "N-1") & (df.fraction == f)]
    a = d[d.model == NAMES[("ft30", "n1")]].set_index("seed")
    for other in sorted(set(d.model) - {NAMES[("ft30", "n1")]}):
        b = d[d.model == other].set_index("seed")
        s = sorted(set(a.index) & set(b.index))
        if len(s) < 2:
            continue
        for k in KEYS:
            x, y = a.loc[s, k].to_numpy(float), b.loc[s, k].to_numpy(float)
            st.append(dict(fraction=f, A="GNN fine-tune (+N-1)", B=other, metric=k, n=len(s),
                           mean_A=x.mean(), mean_B=y.mean(), rel_diff_pct=(x.mean() / y.mean() - 1) * 100,
                           A_better=f"{int((x < y).sum())}/{len(s)}", p_value=stats.ttest_rel(x, y).pvalue))
st = pd.DataFrame(st)
st.to_csv(RES / "n1_adaptation_stats.csv", index=False)

with pd.option_context("display.width", 300, "display.max_columns", 30, "display.max_rows", 200):
    print("N-1 ADAPTATION - IEEE 30 N-1 TEST SET (one branch out per hour)\n")
    print(table.loc["N-1"].to_string())
    print("\nSame models on the INTACT test set\n")
    print(table.loc["intact"].to_string())
    if len(st):
        print("\nPaired tests on the N-1 test set (A = proposed GNN fine-tuned with N-1 data; negative = A better)\n")
        print(st[st.metric.isin(["pg_mae_mw", "pb_p_mean_mw", "cost_gap_abs_pct"])]
              .round(4).to_string(index=False))
print(f"\nSaved {RES / 'n1_adaptation_summary.csv'} and {RES / 'n1_adaptation_stats.csv'}")
