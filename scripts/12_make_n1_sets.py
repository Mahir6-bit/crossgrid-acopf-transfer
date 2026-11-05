"""
12_make_n1_sets.py
IEEE 30 N-1 sets: for every hour of the chosen split, one randomly chosen branch is out of service
(bridges excluded - no islanding) and the AC-OPF is solved again. No IEEE 30 model has seen an
outage during training, so this tests whether a model can use topology information:
  * the GNNs receive the changed graph (the outaged branch is removed),
  * the MLP only sees the loads and cannot know that a branch is missing.

--split test  (default): N-1 TEST set, Nov-Dec, seed 2027
--split train / val    : N-1 samples of the Jan-Aug / Sep-Oct hours for the N-1 adaptation
                         experiment (different seeds, so outages differ from the test set)

Outputs (data\\processed):
  n1<split>_case30.parquet              definitions (base hour, outage) + AC-OPF labels
  graph_tensors_case30_n1<split>.pt     graph tensors (per-sample admittances + edge mask)

Run from the project folder:
    python scripts\\12_make_n1_sets.py                  (test set, 1464 OPF solves)
    python scripts\\12_make_n1_sets.py --split train    (5832 OPF solves)
    python scripts\\12_make_n1_sets.py --split val      (1464 OPF solves)
Options: --workers N --limit N
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pandapower as pp
import torch

from augment_case14 import branch_ends, non_bridges, sample_y
from networks import PROJECT_DIR, get_net, load_buses
from physics import edge_flows, generation_cost, power_balance_residual, scatter_to_bus

PROC = PROJECT_DIR / "data" / "processed"
CASE = "case30"
CHUNK = 100


def solve_chunk(cid, defs):
    import logging, warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("pandapower").setLevel(logging.CRITICAL)
    sc = pd.read_parquet(PROC / f"load_scenarios_{CASE}.parquet")
    sc.index = sc.index.astype(str)
    base_ppc = np.load(PROC / f"network_{CASE}_ppc.npz")
    base_branch, base_mva = base_ppc["branch"], float(base_ppc["baseMVA"])
    net = get_net(CASE)
    lbus = load_buses(net)
    nl, nt, nb = len(net.line), len(net.trafo), len(net.bus)
    gens = [("ext_grid", i, int(b)) for i, b in net.ext_grid.bus.items()] + \
           [("gen", i, int(b)) for i, b in net.gen.bus.items()]
    load_idx = net.load.bus.to_numpy()
    ends = branch_ends(net)
    ones = np.ones(nl + nt)
    rows = []
    for sid, d in defs.iterrows():
        s = sc.loc[str(d.base_hour)]
        p = np.array([s[f"P_bus{b}"] for b in lbus]); q = np.array([s[f"Q_bus{b}"] for b in lbus])
        net.load["p_mw"], net.load["q_mvar"] = p, q
        net.line["in_service"] = True
        if nt:
            net.trafo["in_service"] = True
        o = int(d.outage)
        (net.line if o < nl else net.trafo).loc[o if o < nl else o - nl, "in_service"] = False
        t0 = time.perf_counter()
        try:
            pp.runopp(net); ok = bool(net.OPF_converged)
        except Exception:
            ok = False
        r = {"sample": sid, "converged": ok, "time_s": time.perf_counter() - t0}
        if ok:
            vm, va = net.res_bus.vm_pu.to_numpy(), net.res_bus.va_degree.to_numpy()
            pg = np.array([net[f"res_{et}"].p_mw[i] for et, i, _ in gens])
            qg = np.array([net[f"res_{et}"].q_mvar[i] for et, i, _ in gens])
            Ytt, Yff, Yft, Ytf = sample_y(base_branch, ones, ones, o)
            V = vm * np.exp(1j * np.deg2rad(va))
            fb, tb = ends[:, 0], ends[:, 1]
            Sbus = np.zeros(nb, complex)
            np.add.at(Sbus, fb, V[fb] * np.conj(Yff * V[fb] + Yft * V[tb]) * base_mva)
            np.add.at(Sbus, tb, V[tb] * np.conj(Ytf * V[fb] + Ytt * V[tb]) * base_mva)
            Sbus += np.abs(V) ** 2 * (base_ppc["bus"][:, 4].real - 1j * base_ppc["bus"][:, 5].real)
            inj = np.zeros(nb, complex)
            np.add.at(inj, [b for _, _, b in gens], pg + 1j * qg)
            np.add.at(inj, load_idx, -(p + 1j * q))
            r.update(cost=net.res_cost, max_mismatch_mva=np.abs(Sbus - inj).max())
            rl = net.res_line
            rt = net.res_trafo if nt else None
            cat = lambda a, b: np.r_[a, b] if nt else np.asarray(a)
            pf = cat(rl.p_from_mw, rt.p_hv_mw if nt else 0); qf = cat(rl.q_from_mvar, rt.q_hv_mvar if nt else 0)
            pt = cat(rl.p_to_mw, rt.p_lv_mw if nt else 0); qt = cat(rl.q_to_mvar, rt.q_lv_mvar if nt else 0)
            for b in range(nb):
                r[f"Vm_bus{b+1}"], r[f"Va_bus{b+1}"] = vm[b], va[b]
            for k in range(len(gens)):
                r[f"Pg_gen{k}"], r[f"Qg_gen{k}"] = pg[k], qg[k]
            for k in range(nl + nt):
                r[f"Pf_br{k}"], r[f"Qf_br{k}"] = np.nan_to_num(pf[k]), np.nan_to_num(qf[k])
                r[f"Pt_br{k}"], r[f"Qt_br{k}"] = np.nan_to_num(pt[k]), np.nan_to_num(qt[k])
        rows.append(r)
    return cid, pd.DataFrame(rows).set_index("sample")


def build_tensors(df, split):
    base_t = torch.load(PROC / f"graph_tensors_{CASE}.pt", weights_only=False)
    ppc = np.load(PROC / f"network_{CASE}_ppc.npz")
    base = float(ppc["baseMVA"])
    nb, ng, nbr = base_t["n_bus"], base_t["n_gen"], base_t["n_branch"]
    df = df[df.converged.fillna(False).astype(bool)]
    N = len(df)
    sc = pd.read_parquet(PROC / f"load_scenarios_{CASE}.parquet")
    sc.index = sc.index.astype(str)
    s = sc.loc[df.base_hour.astype(str)]
    pd_ = np.zeros((N, nb)); qd_ = np.zeros((N, nb))
    for c in s.columns:
        if c.startswith("P_bus"):
            pd_[:, int(c[5:]) - 1] += s[c].to_numpy() / base
        elif c.startswith("Q_bus"):
            qd_[:, int(c[5:]) - 1] += s[c].to_numpy() / base
    outage = df.outage.to_numpy().astype(int)
    ys = np.zeros((N, 2 * nbr), complex); ym = np.zeros((N, 2 * nbr), complex)
    ones = np.ones(nbr)
    for i in range(N):
        Ytt, Yff, Yft, Ytf = sample_y(ppc["branch"], ones, ones, outage[i])
        ys[i] = np.r_[Yff, Ytt]; ym[i] = np.r_[Yft, Ytf]
    mask = np.ones((N, 2 * nbr), bool)
    mask[np.arange(N), outage] = False
    mask[np.arange(N), outage + nbr] = False
    col = lambda pre, n, off=0: df[[f"{pre}{i + off}" for i in range(n)]].to_numpy()
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))
    data = dict(case=CASE, baseMVA=base, timestamps=df.index.tolist(), split=[split] * N,
                base_hour=df.base_hour.astype(str).tolist(), n_bus=nb, n_gen=ng, n_branch=nbr,
                edge_index=base_t["edge_index"], gen_bus=base_t["gen_bus"],
                bus_static=base_t["bus_static"], gen_x=base_t["gen_x"],
                edge_attr=base_t["edge_attr"].expand(N, -1, -1).clone(),
                y_phys=t(np.stack([ys.real, ys.imag, ym.real, ym.imag], -1)).float(),
                edge_mask=torch.as_tensor(mask),
                pd=t(pd_).float(), qd=t(qd_).float(),
                vm=t(col("Vm_bus", nb, 1)).float(), va=t(np.deg2rad(col("Va_bus", nb, 1))).float(),
                pg=t(col("Pg_gen", ng) / base).float(), qg=t(col("Qg_gen", ng) / base).float(),
                flow_p=t(np.c_[col("Pf_br", nbr), col("Pt_br", nbr)] / base).float(),
                flow_q=t(np.c_[col("Qf_br", nbr), col("Qt_br", nbr)] / base).float(),
                cost=t(df.cost.to_numpy()).float(), outage=torch.as_tensor(outage))
    # verification with the same physics as training
    yp = data["y_phys"].double()
    bs = base_t["bus_static"].double()
    dp, dq = power_balance_residual(data["vm"].double(), data["va"].double(), t(pd_), t(qd_), bs[:, 0],
                                    bs[:, 1], scatter_to_bus(data["pg"].double(), data["gen_bus"], nb),
                                    scatter_to_bus(data["qg"].double(), data["gen_bus"], nb),
                                    data["edge_index"], yp)
    gx = base_t["gen_x"].double()
    cost_c = generation_cost(data["pg"].double(), gx[:, 4], gx[:, 5], gx[:, 6]).sum(-1)
    print(f"  CHECK nodal power balance: max |dP| {dp.abs().max().item() * base:.2e} MW, "
          f"max |dQ| {dq.abs().max().item() * base:.2e} Mvar | cost rel. error "
          f"{((cost_c - data['cost'].double()).abs() / data['cost'].double()).max().item():.1e}")
    out = PROC / f"graph_tensors_{CASE}_n1{split}.pt"
    torch.save(data, out)
    print(f"  saved {out} ({N} samples)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    seed = {"test": 2027, "train": 2028, "val": 2029}[a.split]

    split = pd.read_csv(PROC / "splits.csv", index_col=0).split
    hours = split.index[split == a.split].astype(str).tolist()
    net = get_net(CASE)
    cand = non_bridges(net)
    ends = branch_ends(net) + 1
    rng = np.random.default_rng(seed)
    defs = pd.DataFrame({"base_hour": hours, "outage": rng.choice(cand, len(hours))})
    defs.index = [f"n1{a.split}_{i:05d}" if a.split != "test" else f"n1_{i:05d}" for i in range(len(defs))]
    defs.index.name = "sample"
    if a.limit:
        defs = defs.iloc[:a.limit]
    print(f"IEEE 30 N-1 {a.split} set: {len(defs)} hours, outage drawn from {len(cand)} of "
          f"{len(ends)} branches (bridges excluded)")
    chunks = [(c, defs.iloc[s:s + CHUNK]) for c, s in enumerate(range(0, len(defs), CHUNK))]
    parts = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(solve_chunk, c, d) for c, d in chunks]
        for k, f in enumerate(as_completed(futs), 1):
            parts.append(f.result()[1])
            print(f"  {k}/{len(chunks)} chunks | {(time.time() - t0) / 60:.1f} min", flush=True)
    lab = defs.join(pd.concat(parts), how="left")
    lab.to_parquet(PROC / f"n1{a.split}_{CASE}.parquet")
    ok = lab[lab.converged.fillna(False).astype(bool)]
    print(f"Converged {len(ok)}/{len(lab)} ({100 * len(ok) / len(lab):.1f} %); physics check max |S mismatch| "
          f"{ok.max_mismatch_mva.max():.1e} MVA")
    fails = lab[~lab.converged.fillna(False).astype(bool)].outage.value_counts().head(5)
    if len(fails):
        print("  most frequent outages among infeasible hours (branch: from-to, count): " +
              ", ".join(f"{k} ({ends[k][0]}-{ends[k][1]}): {v}" for k, v in fails.items()))
    build_tensors(lab, a.split)
