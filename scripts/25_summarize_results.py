"""
25_summarize_results.py
Collects results\\*\\metrics.json into the paper tables (mean +- std over seeds):
  * IEEE 30 TEST: zero-shot (source models), fine-tune and from scratch per data fraction  (E2-E4)
  * IEEE 14 TEST: source models (E1) and after fine-tuning (forgetting)
Run variants with a tag (e.g. "_nophys") appear as separate methods, e.g. "fine-tune [nophys]".

Run from the project folder:
    python scripts\\25_summarize_results.py
Writes results\\summary_case30.csv, results\\summary_case14.csv, results\\all_runs.csv
"""

import json
from pathlib import Path

import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
KEY = ["vm_mae_pu", "va_mae_deg", "pg_mae_mw", "qg_mae_mvar", "vm_max_err_pu", "va_max_err_rad",
       "pb_p_mean_mw", "pb_q_mean_mvar", "pb_p_max_mw", "branch_viol_pct",
       "samples_with_branch_viol_pct", "cost_gap_abs_pct", "best_step"]


def parse(name):
    """-> (kind, fraction, seed, tag) or None."""
    parts = name.split("_")
    if parts[0] == "src14" and len(parts) >= 2 and parts[1].startswith("s"):
        return "src", 0.0, int(parts[1][1:]), "_".join(parts[2:])
    if parts[0] in ("ft30", "scratch30") and len(parts) >= 3:
        return parts[0], float(parts[1][1:]), int(parts[2][1:]), "_".join(parts[3:])
    return None


rows = []
for f in RES.glob("*/metrics.json"):
    p = parse(f.parent.name)
    if p is None:
        continue
    kind, frac, seed, tag = p
    for m in json.load(open(f)):
        if kind == "src":
            method = "source (E1)" if m["case"] == "case14" else "zero-shot (E2)"
        else:
            method = "fine-tune" if kind == "ft30" else "scratch"
        if tag:
            method += f" [{tag}]"
        m.update(run=f.parent.name, method=method, fraction=frac, seed=seed)
        rows.append(m)
df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No results found (expected folders like results\\src14_s0, ft30_f0.05_s0).")
df.to_csv(RES / "all_runs.csv", index=False)


def table(d):
    keys = [k for k in KEY if k in d.columns]
    g = d.groupby(["method", "fraction"])
    mean, std, n = g[keys].mean(), g[keys].std(), g.size().rename("n_seeds")
    out = mean.copy().astype(object)
    for k in keys:
        out[k] = [f"{m:.4g} ± {s:.2g}" if pd.notna(s) else f"{m:.4g}" for m, s in zip(mean[k], std[k])]
    return pd.concat([n, out], axis=1)


with pd.option_context("display.width", 300, "display.max_columns", 30):
    t30 = table(df[df.case == "case30"])
    t30.to_csv(RES / "summary_case30.csv")
    print("IEEE 30 TEST (Nov-Dec), mean ± std over seeds; fraction = share of IEEE 30 training data\n")
    print(t30.to_string())

    t14 = table(df[df.case == "case14"])
    t14.to_csv(RES / "summary_case14.csv")
    print("\nIEEE 14 TEST: source models (E1); fine-tune rows = forgetting after adapting to IEEE 30;"
          " scratch rows = IEEE 30-only models applied to IEEE 14\n")
    print(t14[["n_seeds", "vm_mae_pu", "pg_mae_mw", "pb_p_mean_mw", "cost_gap_abs_pct"]].to_string())
print(f"\nSaved {RES / 'summary_case30.csv'}, {RES / 'summary_case14.csv'}, {RES / 'all_runs.csv'}")
