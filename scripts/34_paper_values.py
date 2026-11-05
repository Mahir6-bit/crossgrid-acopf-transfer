"""
34_paper_values.py
Prints values that are reported in the text of the paper:
  1) the eight branches withheld in the unseen-outage experiment (E8) and the AC-OPF convergence
     count of the N-1 test set,
  2) facts about the hourly load series on 1 January (completeness and load level),
  3) the time to build the input graph and the effective-resistance matrix (IEEE 30-bus, CPU).
It only reads files written by the earlier steps.

    python scripts\34_paper_values.py
"""

import statistics
import time

import numpy as np
import pandas as pd
import torch

from augment_case14 import branch_ends
from networks import DATASET, PROJECT_DIR, get_net
from opf_dataset import GraphBatcher, OPFGraphDataset, effective_resistance, n1_holdout_branches

PROC = PROJECT_DIR / "data" / "processed"
RAW = PROJECT_DIR / "data" / "raw"

# ------------------------------------------------------------------ 1) N-1 placeholders
ends = branch_ends(get_net("case30")) + 1                       # 1-based buses, MATPOWER branch order
held = n1_holdout_branches(8, 7)                                 # same call as the *_n1ho training runs
print("1a) Withheld branches (MATPOWER branch number: from-to bus):")
print("    " + ", ".join(f"{k + 1}: {ends[k][0]}–{ends[k][1]}" for k in held))

lab = pd.read_parquet(PROC / "n1test_case30.parquet")
ok = int(lab.converged.fillna(False).astype(bool).sum())
print(f"1b) N-1 test set: AC-OPF converged for {ok} of {len(lab)} hours ({100 * ok / len(lab):.1f}%)")

# ------------------------------------------------------------------ 2) 1 January spike
print("\n2) 1 January load spike (raw per-bus factor = Cons_i / nominal Pd_i, mean over pure load buses):")
for case in ("case14", "case30"):
    meta = DATASET[case]
    df = pd.read_csv(RAW / meta["file"], index_col=0)
    df.index = pd.to_datetime(df.index)
    Pd = np.array(meta["Pd"], float)
    pure = [b for b in range(1, meta["n_bus"] + 1) if Pd[b - 1] > 0 and b not in meta["gen_buses"]]
    lam = pd.DataFrame({b: df[f"Cons_{b}"] / Pd[b - 1] for b in pure}).mean(axis=1)
    jan1 = lam[lam.index.normalize() == pd.Timestamp("2007-01-01")]
    rest = lam[(lam.index >= "2007-01-02") & (lam.index < "2007-01-15")]
    full = pd.date_range("2007-01-01", "2007-12-31 23:00", freq="h")
    print(f"  {case}: rows in file {len(df)}, missing hourly timestamps {len(full.difference(df.index))}, "
          f"duplicate timestamps {int(df.index.duplicated().sum())}, empty cells {int(df.isna().sum().sum())}")
    print(f"    hours present on 1 Jan: {len(jan1)}; mean factor 1 Jan {jan1.mean():.3f} "
          f"vs 2-14 Jan {rest.mean():.3f}; 1 Jan min/max {jan1.min():.3f}/{jan1.max():.3f}")
    print("    1 Jan hourly factors: " + " ".join(f"{v:.2f}" for v in jan1.to_numpy()))
    same = (df[[f"Cons_{b}" for b in pure]].loc[jan1.index].nunique() == 1).sum() if len(jan1) else 0
    print(f"    load buses whose value is constant over 1 Jan: {int(same)} of {len(pure)}")

# ------------------------------------------------------------------ 3) graph-building time
print("\n3) Graph construction time per operating point, IEEE 30-bus, CPU (N-1 test hours, outage removed):")
torch.set_num_threads(1)
ds = OPFGraphDataset("case30", tensor_file="graph_tensors_case30_n1test.pt")
t = ds.t
gb = GraphBatcher(ds, torch.device("cpu"))
n = min(200, len(ds.idx))
er_t, mk_t = [], []
for rep in range(n + 20):                                        # first 20 = warm-up
    i = rep % n
    ea = t["edge_attr"][i:i + 1] if t["edge_attr"].dim() == 3 else t["edge_attr"]
    t0 = time.perf_counter()
    effective_resistance(ea, t["edge_index"], t["n_bus"], t["edge_mask"][i:i + 1])
    t1 = time.perf_counter()
    gb.make(torch.tensor([i]))
    t2 = time.perf_counter()
    if rep >= 20:
        er_t.append((t1 - t0) * 1e3); mk_t.append((t2 - t1) * 1e3)
er, mk = statistics.median(er_t), statistics.median(mk_t)
print(f"    effective-resistance matrix (pseudo-inverse): median {er:.3f} ms")
print(f"    graph assembly:                               median {mk:.3f} ms")
print(f"    total per operating point:                    {er + mk:.2f} ms  <- value for [X] ms")
