"""
24_evaluate_unseen.py
Unseen-outage experiment (IEEE 30, 25 % of training hours + N-1 data).

The N-1 test set is split into
  unseen : test hours whose outaged branch was HELD OUT of all N-1 training/validation data
           (runs tagged _n1ho / _n1homlptopo, trained with --n1-holdout 8)
  seen   : all other N-1 test hours
Models compared (25 %, all available seeds):
  GNN fine-tune / from scratch, trained with held-out outages            (_n1ho)
  MLP + outage one-hot, trained with held-out outages                    (_n1homlptopo)
      - "raw": the one-hot column of an unseen branch was never trained
      - "unseen -> intact": an unseen outage is fed as 'no outage' (the best a practitioner could do)
References: the same models trained with ALL outages (_n1, _n1mlptopo) and the topology-blind
MLP without N-1 data (the 'ignore the outage' floor).

Run from the project folder:
    python scripts\\24_evaluate_unseen.py
Writes results\\unseen_runs.csv, unseen_summary.csv, unseen_stats.csv
"""

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from evaluate import evaluate, load_checkpoint
from opf_dataset import OPFGraphDataset, n1_holdout_branches

RES = Path(__file__).resolve().parents[1] / "results"
PROC = RES.parents[0] / "data" / "processed"
KEYS = ["pg_mae_mw", "va_mae_deg", "vm_mae_pu", "pb_p_mean_mw", "pb_p_max_mw", "branch_viol_pct",
        "cost_gap_abs_pct"]
GROUPS = [  # (label, run-name glob, mlp variant)
    ("GNN fine-tune, outages held out", "ft30_f0.25_s?_n1ho", None),
    ("GNN from scratch, outages held out", "scratch30_f0.25_s?_n1ho", None),
    ("MLP one-hot, outages held out (raw)", "scratch30_f0.25_s?_n1homlptopo", "raw"),
    ("MLP one-hot, outages held out (unseen -> intact)", "scratch30_f0.25_s?_n1homlptopo", "intact"),
    ("REF GNN fine-tune, all outages trained", "ft30_f0.25_s?_n1", None),
    ("REF MLP one-hot, all outages trained", "scratch30_f0.25_s?_n1mlptopo", "raw"),
    ("REF MLP topology-blind, no N-1 data (floor)", "scratch30_f0.25_s?_mlp", None),
]

if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    held = n1_holdout_branches(8, 7)
    full = torch.load(PROC / "graph_tensors_case30_n1test.pt", weights_only=False)
    br = pd.read_csv(PROC / "network_case30_branches.csv", index_col=0)
    out = full["outage"].numpy()
    ts = np.array(full["timestamps"])
    subsets = {"unseen": ts[np.isin(out, held)].tolist(), "seen": ts[~np.isin(out, held)].tolist()}
    print("Held-out branches: " + ", ".join(f"{b} ({br.from_bus[b]}-{br.to_bus[b]})" for b in held))
    print(f"N-1 test hours: unseen {len(subsets['unseen'])}, seen {len(subsets['seen'])}")

    ds = {k: OPFGraphDataset("case30", tensor_file="graph_tensors_case30_n1test.pt", timestamps=v)
          for k, v in subsets.items()}
    ds_intact = copy.copy(ds["unseen"])                 # unseen outage fed to the MLP as 'intact'
    ds_intact.t = dict(ds["unseen"].t)
    o = ds_intact.t["outage"].clone()
    o[torch.as_tensor(ds_intact.idx)] = -1
    ds_intact.t["outage"] = o

    rows = []
    for label, pattern, variant in GROUPS:
        for ck in sorted(RES.glob(f"{pattern}/best.pt")):
            model, _ = load_checkpoint(ck, dev)
            seed = int(ck.parent.name.split("_")[2][1:])
            for sub in ["unseen", "seen"]:
                d = ds_intact if (variant == "intact" and sub == "unseen") else ds[sub]
                m = evaluate(model, "case30", device=dev, ds=d, timing=False)
                rows.append(dict(model=label, seed=seed, subset=sub, run=ck.parent.name,
                                 **{k: m[k] for k in KEYS}))
        if not any(r["model"] == label for r in rows):
            print(f"  (no runs for '{label}')")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "unseen_runs.csv", index=False)

    g = df.groupby(["subset", "model"], sort=False)
    mean, std = g[KEYS].mean(), g[KEYS].std()
    table = mean.astype(object)
    for k in KEYS:
        table[k] = [f"{m:.4g} ± {s:.2g}" if pd.notna(s) else f"{m:.4g}" for m, s in zip(mean[k], std[k])]
    table.insert(0, "n_seeds", g.size())
    table.to_csv(RES / "unseen_summary.csv")

    st = []
    a_lab = GROUPS[0][0]
    for sub in ["unseen", "seen"]:
        d = df[df.subset == sub]
        a = d[d.model == a_lab].set_index("seed")
        for other in [x for x in d.model.unique() if x != a_lab]:
            b = d[d.model == other].set_index("seed")
            s = sorted(set(a.index) & set(b.index))
            if len(s) < 2:
                continue
            for k in KEYS:
                x, y = a.loc[s, k].to_numpy(float), b.loc[s, k].to_numpy(float)
                st.append(dict(subset=sub, B=other, metric=k, n=len(s), mean_A=x.mean(), mean_B=y.mean(),
                               rel_diff_pct=(x.mean() / y.mean() - 1) * 100,
                               A_better=f"{int((x < y).sum())}/{len(s)}", p_value=stats.ttest_rel(x, y).pvalue))
    st = pd.DataFrame(st)
    st.to_csv(RES / "unseen_stats.csv", index=False)

    with pd.option_context("display.width", 300, "display.max_columns", 30, "display.max_rows", 200):
        for sub in ["unseen", "seen"]:
            print(f"\nN-1 TEST, {sub.upper()} outages\n")
            print(table.loc[sub].to_string())
        if len(st):
            print(f"\nPaired tests (A = {a_lab}; negative = A better)\n")
            print(st[st.metric.isin(["pg_mae_mw", "pb_p_mean_mw", "cost_gap_abs_pct"])].round(4).to_string(index=False))
    print(f"\nSaved {RES / 'unseen_runs.csv'}, {RES / 'unseen_summary.csv'}, {RES / 'unseen_stats.csv'}")
