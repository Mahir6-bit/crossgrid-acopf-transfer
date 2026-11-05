"""
19_benchmark_runtime.py
Clean runtime comparison on the same machine, ONE process, nothing else running:
  * AC-OPF solver: pandapower runopp (PIPS interior point, flat start) per hour
  * GNN: single sample (latency) on GPU and CPU, and batched throughput on GPU
on N test hours of each grid.

Close other heavy programs first. Run from the project folder:
    python scripts\\19_benchmark_runtime.py
Options: --n 200  --ckpt14 results\\src14_s0\\best.pt  --ckpt30 results\\ft30_f1.0_s0\\best.pt
Writes results\\runtime_benchmark.csv
"""

import argparse
import logging
import time
import warnings

import numpy as np
import pandas as pd
import pandapower as pp
import torch

from evaluate import load_checkpoint
from networks import PROJECT_DIR, get_net, load_buses
from opf_dataset import GraphBatcher, OPFGraphDataset

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.CRITICAL)


def time_solver(case, ds, n):
    sc = pd.read_parquet(PROJECT_DIR / "data" / "processed" / f"load_scenarios_{case}.parquet")
    sc.index = sc.index.astype(str)
    net = get_net(case)
    buses = load_buses(net)
    pp.runopp(net)                                    # warm-up (imports, numba compile)
    ts = []
    for i in ds.idx[:n]:
        s = sc.loc[ds.t["timestamps"][i]]
        net.load["p_mw"] = [s[f"P_bus{b}"] for b in buses]
        net.load["q_mvar"] = [s[f"Q_bus{b}"] for b in buses]
        t0 = time.perf_counter(); pp.runopp(net); ts.append(time.perf_counter() - t0)
    return np.array(ts) * 1000


@torch.no_grad()
def time_gnn_single(model, ds, dev, n):
    model = model.to(dev).eval()
    gb = GraphBatcher(ds, dev)
    sync = (lambda: torch.cuda.synchronize()) if dev.type == "cuda" else (lambda: None)
    for i in range(20):                              # warm-up
        model(gb.make(torch.tensor([i], device=dev)))
    ts = []
    for i in range(min(n, gb.n)):
        b = gb.make(torch.tensor([i], device=dev))
        sync(); t0 = time.perf_counter(); model(b); sync()
        ts.append(time.perf_counter() - t0)
    return np.array(ts) * 1000


@torch.no_grad()
def time_gnn_batched(model, ds, dev, bs=512):
    model = model.to(dev).eval()
    gb = GraphBatcher(ds, dev)
    batches = list(gb.batches(bs))
    model(batches[0])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for b in batches:
        model(b)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / gb.n * 1000


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ckpt14", default=str(PROJECT_DIR / "results" / "src14_s0" / "best.pt"))
    ap.add_argument("--ckpt30", default=str(PROJECT_DIR / "results" / "ft30_f1.0_s0" / "best.pt"))
    a = ap.parse_args()
    gpu = torch.device("cuda") if torch.cuda.is_available() else None
    rows = []
    for case, ck in [("case14", a.ckpt14), ("case30", a.ckpt30)]:
        ds = OPFGraphDataset(case, split="test")
        model, _ = load_checkpoint(ck, torch.device("cpu"))
        print(f"[{case}] timing solver on {a.n} test hours ...")
        sol = time_solver(case, ds, a.n)
        cpu = time_gnn_single(model, ds, torch.device("cpu"), a.n)
        r = dict(case=case, solver_ms_mean=sol.mean(), solver_ms_median=np.median(sol),
                 solver_ms_worst=sol.max(), gnn_cpu_single_ms_median=np.median(cpu),
                 gnn_cpu_single_ms_worst=cpu.max())
        if gpu:
            g1 = time_gnn_single(model, ds, gpu, a.n)
            r.update(gnn_gpu_single_ms_mean=g1.mean(), gnn_gpu_single_ms_median=np.median(g1),
                     gnn_gpu_single_ms_worst=g1.max(),
                     gnn_gpu_batched_ms_per_sample=time_gnn_batched(model, ds, gpu))
            r["speedup_gpu_single"] = r["solver_ms_median"] / r["gnn_gpu_single_ms_median"]
            r["speedup_gpu_batched"] = r["solver_ms_median"] / r["gnn_gpu_batched_ms_per_sample"]
        r["speedup_cpu_single"] = r["solver_ms_median"] / r["gnn_cpu_single_ms_median"]
        rows.append(r)
    df = pd.DataFrame(rows).set_index("case").T
    print(df.round(3).to_string())
    df.to_csv(PROJECT_DIR / "results" / "runtime_benchmark.csv")
    print(f"Saved {PROJECT_DIR / 'results' / 'runtime_benchmark.csv'}")
