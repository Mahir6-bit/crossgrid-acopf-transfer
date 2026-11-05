"""
evaluate.py
Test-set metrics for a trained checkpoint (also imported by train.py).

Metrics (physical units):
  vm_mae_pu, va_mae_deg, pg_mae_mw (dispatchable gens only), qg_mae_mvar
  vm/va/pg/qg MSE (pu^2, rad^2), vm_max_err_pu, va_max_err_rad
  pb_p_mean_mw / pb_p_max_mw, pb_q_mean_mvar / pb_q_max_mvar   power-balance mismatch per bus
  branch_viol_pct    % of branch-end/hour pairs with current above rating (pandapower's limit)
  branch_mean_over_pct / branch_max_over_pct   mean and worst overload above rating
  samples_with_branch_viol_pct   % of test hours with at least one overloaded branch
  vm_limit_viol_pct  % of bus voltages outside limits (0 by construction - sigmoid decoder)
  max_branch_angle_deg, angle_viol_pct   largest |theta_i - theta_j| and % of branch-ends above 30 deg
  cost_gap_pct (signed mean) and cost_gap_abs_pct          optimality gap vs AC-OPF cost
  gnn_ms_per_sample_batched, gnn_ms_per_sample_single, solver_ms_per_sample (from label run)

Usage:
    python scripts\\evaluate.py --ckpt results\\<run>\\best.pt --cases case14 case30
"""

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from model import BRANCH, G2B, build_model
from networks import PROJECT_DIR
from opf_dataset import GraphBatcher, OPFGraphDataset
from physics import edge_flows, generation_cost, power_balance_residual, scatter_to_bus

BASE = 100.0


def physics_terms(batch, pred):
    b, e = batch["bus"], batch[BRANCH]
    g2b = batch[G2B].edge_index
    nb = b.num_nodes
    pg_bus = scatter_to_bus(pred["pg"], g2b[1], nb)
    qg_bus = scatter_to_bus(pred["qg"], g2b[1], nb)
    dp, dq = power_balance_residual(pred["vm"], pred["va"], b.x[:, 0], b.x[:, 1], b.x[:, 2], b.x[:, 3],
                                    pg_bus, qg_bus, e.edge_index, e.y_phys)
    p, q = edge_flows(pred["vm"], pred["va"], e.edge_index, e.y_phys)
    i_mag = torch.sqrt(p * p + q * q + 1e-12) / pred["vm"][e.edge_index[0]]
    over = torch.relu(i_mag - e.edge_attr[:, 3])
    return dp, dq, i_mag, over


def angle_terms(batch, pred):
    """Voltage-angle difference across every directed branch edge (rad)."""
    s, d = batch[BRANCH].edge_index
    return pred["va"][s] - pred["va"][d]


@torch.no_grad()
def evaluate(model, case, split="test", device="cpu", timestamps=None, batch_size=256, ds=None,
             timing=True):
    """ds: optional prepared dataset (e.g. the IEEE 30 N-1 test set); timing=False skips latency."""
    model.eval()
    if ds is None:
        ds = OPFGraphDataset(case, split=None if timestamps else split, timestamps=timestamps)
    gb = GraphBatcher(ds, device)
    acc = {k: [] for k in ["vm", "va", "pg", "qg", "dp", "dq", "ratio", "gap", "cost",
                           "graph_ratio", "vlim", "dth"]}
    n_graphs, t_gnn = 0, 0.0
    for batch in gb.batches(batch_size):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pred = model(batch)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_gnn += time.perf_counter() - t0
        n_graphs += batch.num_graphs

        gy, by, gx = batch["gen"].y, batch["bus"].y, batch["gen"].x
        disp = gx[:, 1] > gx[:, 0]
        acc["vm"].append((pred["vm"] - by[:, 0]).abs())
        acc["va"].append((pred["va"] - by[:, 1]).abs())
        acc["pg"].append((pred["pg"] - gy[:, 0]).abs()[disp])
        acc["qg"].append((pred["qg"] - gy[:, 1]).abs())
        dp, dq, i_mag, _ = physics_terms(batch, pred)
        acc["dp"].append(dp.abs()); acc["dq"].append(dq.abs())
        ratio = i_mag / batch[BRANCH].edge_attr[:, 3]
        acc["ratio"].append(ratio)
        e_graph = batch["bus"].batch[batch[BRANCH].edge_index[0]]
        acc["graph_ratio"].append(scatter_to_bus(torch.zeros_like(ratio), e_graph, batch.num_graphs)
                                  .scatter_reduce(0, e_graph, ratio, reduce="amax"))
        acc["dth"].append(angle_terms(batch, pred).abs())
        bx = batch["bus"].x
        acc["vlim"].append(((pred["vm"] < bx[:, 4] - 1e-6) | (pred["vm"] > bx[:, 5] + 1e-6)).double())
        c = generation_cost(pred["pg"], gx[:, 4], gx[:, 5], gx[:, 6])
        cost_pred = scatter_to_bus(c, batch["gen"].batch, batch.num_graphs)
        acc["gap"].append((cost_pred - batch.cost) / batch.cost * 100)
    cat = {k: torch.cat(v).double().cpu() for k, v in acc.items() if v}
    ratio = cat["ratio"]
    m = dict(
        case=case, split=split if not timestamps else "custom", n_samples=n_graphs,
        vm_mae_pu=cat["vm"].mean().item(),
        va_mae_deg=np.rad2deg(cat["va"].mean().item()),
        pg_mae_mw=cat["pg"].mean().item() * BASE,
        qg_mae_mvar=cat["qg"].mean().item() * BASE,
        pb_p_mean_mw=cat["dp"].mean().item() * BASE, pb_p_max_mw=cat["dp"].max().item() * BASE,
        pb_q_mean_mvar=cat["dq"].mean().item() * BASE, pb_q_max_mvar=cat["dq"].max().item() * BASE,
        vm_mse_pu2=(cat["vm"] ** 2).mean().item(),
        va_mse_rad2=(cat["va"] ** 2).mean().item(),
        pg_mse_pu2=(cat["pg"] ** 2).mean().item(),
        qg_mse_pu2=(cat["qg"] ** 2).mean().item(),
        vm_max_err_pu=cat["vm"].max().item(),
        va_max_err_rad=cat["va"].max().item(),
        branch_viol_pct=(ratio > 1.001).double().mean().item() * 100,
        branch_mean_over_pct=(torch.relu(ratio - 1) * 100).mean().item(),
        branch_max_over_pct=max(0.0, (ratio.max().item() - 1) * 100),
        samples_with_branch_viol_pct=(cat["graph_ratio"] > 1.001).double().mean().item() * 100,
        vm_limit_viol_pct=cat["vlim"].mean().item() * 100,
        max_branch_angle_deg=np.rad2deg(cat["dth"].max().item()),
        angle_viol_pct=(cat["dth"] > np.deg2rad(30)).double().mean().item() * 100,
        cost_gap_pct=cat["gap"].mean().item(),
        cost_gap_abs_pct=cat["gap"].abs().mean().item(),
        gnn_ms_per_sample_batched=t_gnn / n_graphs * 1000,
    )
    if not timing:
        return m
    # single-sample latency (what replaces one solver call)
    ts = []
    for i in range(min(200, gb.n)):
        b = gb.make(torch.tensor([i], device=device))
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter(); model(b)
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    m["gnn_ms_per_sample_single"] = float(np.median(ts[10:] if len(ts) > 20 else ts) * 1000)
    lab = pd.read_parquet(PROJECT_DIR / "data" / "processed" / f"acopf_labels_{case}.parquet",
                          columns=["time_s"])
    lab.index = lab.index.astype(str)
    sel = [ds.t["timestamps"][i] for i in ds.idx]
    m["solver_ms_per_sample"] = float(lab.loc[sel, "time_s"].mean() * 1000)
    return m


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ck["model_cfg"]).to(device)
    model.load_state_dict(ck["state_dict"])
    return model, ck


def print_metrics(rows):
    df = pd.DataFrame(rows).set_index("case").T
    with pd.option_context("display.float_format", "{:.4g}".format):
        print(df.to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cases", nargs="+", default=["case14", "case30"])
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(a.ckpt, dev)
    rows = [evaluate(model, c, a.split, dev) for c in a.cases]
    print_metrics(rows)
