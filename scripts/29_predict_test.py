"""
29_predict_test.py
Saves test-set predictions of representative models (seed 0) for the parity, bus-profile and
time-series figures.

Run from the project folder:
    python scripts\\29_predict_test.py
Writes results\\predictions\\<run>_<case>.npz with predicted and true Vm, Va, Pg, Qg, cost, timestamps
"""

from pathlib import Path

import numpy as np
import torch

from evaluate import load_checkpoint
from opf_dataset import GraphBatcher, OPFGraphDataset
from physics import generation_cost, scatter_to_bus

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "predictions"
RUNS = [("ft30_f0.05_s0", "case30"), ("ft30_f1.0_s0", "case30"), ("scratch30_f0.05_s0", "case30"),
        ("scratch30_f1.0_s0", "case30"), ("scratch30_f1.0_s0_mlp", "case30"), ("src14_s0", "case30"),
        ("src14_s0", "case14")]


@torch.no_grad()
def predict(run, case, dev):
    model, _ = load_checkpoint(RES / run / "best.pt", dev)
    model.eval()
    ds = OPFGraphDataset(case, split="test")
    gb = GraphBatcher(ds, dev)
    keys = ["vm", "va", "pg", "qg"]
    pred = {k: [] for k in keys}
    true = {k: [] for k in keys}
    cost_p, cost_t = [], []
    for b in gb.batches(512):
        p = model(b)
        n = b.num_graphs
        by, gy, gx = b["bus"].y, b["gen"].y, b["gen"].x
        for k, src in [("vm", by[:, 0]), ("va", by[:, 1]), ("pg", gy[:, 0]), ("qg", gy[:, 1])]:
            pred[k].append(p[k].view(n, -1).cpu()); true[k].append(src.view(n, -1).cpu())
        c = generation_cost(p["pg"], gx[:, 4], gx[:, 5], gx[:, 6])
        cost_p.append(scatter_to_bus(c, b["gen"].batch, n).cpu()); cost_t.append(b.cost.cpu())
    out = {f"pred_{k}": torch.cat(v).numpy() for k, v in pred.items()}
    out.update({f"true_{k}": torch.cat(v).numpy() for k, v in true.items()})
    out.update(pred_cost=torch.cat(cost_p).numpy(), true_cost=torch.cat(cost_t).numpy(),
               timestamps=np.array([ds.t["timestamps"][i] for i in ds.idx]),
               gen_pmax=ds.t["gen_x"][:, 1].numpy(), gen_pmin=ds.t["gen_x"][:, 0].numpy(),
               gen_bus=ds.t["gen_bus"].numpy())
    return out


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    for run, case in RUNS:
        if not (RES / run / "best.pt").exists():
            print(f"  {run}: not found - skipped"); continue
        np.savez(OUT / f"{run}_{case}.npz", **predict(run, case, dev))
        print(f"  saved {OUT / f'{run}_{case}.npz'}")
