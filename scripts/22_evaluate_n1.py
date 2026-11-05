"""
22_evaluate_n1.py
Evaluates every trained IEEE 30 model (and the zero-shot source models) on the N-1 test set
(graph_tensors_case30_n1test.pt from make_n1_testset.py) and compares with the same model on
the intact test set.

Run from the project folder:
    python scripts\\22_evaluate_n1.py
Writes results\\n1_runs.csv (one row per model) and results\\n1_summary.csv (mean +- std)
"""

import json
from pathlib import Path

import pandas as pd
import torch

from evaluate import evaluate, load_checkpoint
from opf_dataset import OPFGraphDataset

RES = Path(__file__).resolve().parents[1] / "results"
KEYS = ["pg_mae_mw", "va_mae_deg", "vm_mae_pu", "pb_p_mean_mw", "pb_p_max_mw",
        "branch_viol_pct", "cost_gap_abs_pct"]


def label(name):
    parts = name.split("_")
    tag = "_".join(parts[3:]) if parts[0] in ("ft30", "scratch30") else "_".join(parts[2:])
    arch = {"gcn": "GCN", "gat": "GAT", "mlp": "MLP"}.get(tag, "proposed" if tag == "" else None)
    if arch is None:                       # ablation variants are not part of this comparison
        return None
    if parts[0] == "src14":
        return arch, "zero-shot", 0.0, int(parts[1][1:])
    if parts[0] in ("ft30", "scratch30"):
        return arch, "fine-tune" if parts[0] == "ft30" else "scratch", float(parts[1][1:]), int(parts[2][1:])
    return None


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = OPFGraphDataset("case30", tensor_file="graph_tensors_case30_n1test.pt")
    print(f"N-1 test set: {len(ds)} hours")
    rows = []
    for ck in sorted(RES.glob("*/best.pt")):
        name = ck.parent.name
        lab = label(name)
        if lab is None or not name.startswith(("src14_s", "ft30_", "scratch30_")):
            continue
        model, _ = load_checkpoint(ck, dev)
        n1 = evaluate(model, "case30", device=dev, ds=ds, timing=False)
        intact = {}
        mfile = ck.parent / "metrics.json"
        if mfile.exists():
            intact = next((m for m in json.load(open(mfile)) if m["case"] == "case30"), {})
        r = dict(run=name, model=lab[0], method=lab[1], fraction=lab[2], seed=lab[3])
        r.update({f"n1_{k}": n1[k] for k in KEYS})
        r.update({f"intact_{k}": intact.get(k) for k in KEYS})
        rows.append(r)
        print(f"  {name:26s} N-1: Pg MAE {n1['pg_mae_mw']:.3f} MW, mean |dP| {n1['pb_p_mean_mw']:.3f} MW, "
              f"gap {n1['cost_gap_abs_pct']:.3f} % | intact Pg MAE {intact.get('pg_mae_mw', float('nan')):.3f}",
              flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(RES / "n1_runs.csv", index=False)

    cols = [f"n1_{k}" for k in KEYS] + ["intact_pg_mae_mw", "intact_pb_p_mean_mw"]
    for k in ["pg_mae_mw", "pb_p_mean_mw"]:
        df[f"ratio_{k}"] = df[f"n1_{k}"] / df[f"intact_{k}"]
        cols.append(f"ratio_{k}")
    g = df.groupby(["model", "method", "fraction"])
    mean, std = g[cols].mean(), g[cols].std()
    table = mean.astype(object)
    for c in cols:
        table[c] = [f"{m:.4g} ± {s:.2g}" if pd.notna(s) else f"{m:.4g}" for m, s in zip(mean[c], std[c])]
    table.insert(0, "n_seeds", g.size())
    table.to_csv(RES / "n1_summary.csv")
    with pd.option_context("display.width", 300, "display.max_columns", 30):
        print("\nIEEE 30 N-1 TEST (one branch out per hour); ratio = N-1 error / intact error\n")
        print(table.to_string())
    print(f"\nSaved {RES / 'n1_runs.csv'} and {RES / 'n1_summary.csv'}")
