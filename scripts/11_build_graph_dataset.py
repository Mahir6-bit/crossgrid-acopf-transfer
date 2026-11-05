"""
11_build_graph_dataset.py
Converts the AC-OPF labels into compact per-unit tensors for heterogeneous graphs,
and VERIFIES them with the PyTorch physics in physics.py (the future physics-informed loss).

Graph per hour:
  bus  nodes  x = [Pd, Qd, Gs, Bs, Vmin, Vmax, is_PQ, is_PV, is_slack]  y = [Vm, Va(rad)]
              (bus type one-hot from the network data; base kV is not used because the
               PGLib case14 file sets every base kV to 1.0, so it carries no information)
  gen  nodes  x = [Pmin, Pmax, Qmin, Qmax, c2, c1, c0, is_slack_gen]  y = [Pg, Qg]
  (bus, branch, bus) directed edges (2 per branch)
        edge_attr = [r, x, b, rate, tap, shift(rad), is_trafo, is_forward]
        y_phys    = [Re Yself, Im Yself, Re Ymut, Im Ymut]     (for the physics loss)
        y         = [P, Q] leaving the source bus              (branch-flow labels)
  (gen, gen_to_bus, bus) and (bus, bus_to_gen, gen) edges
  graph-level: cost [$/h]
All powers in per-unit (baseMVA 100); costs converted to per-unit coefficients.
Raw physical values are stored - normalisation is done later from the TRAIN split only.

Run from the project folder with the venv active:
    python scripts\\11_build_graph_dataset.py
Writes data\\processed\\graph_tensors_case14.pt and graph_tensors_case30.pt
"""

import numpy as np
import pandas as pd
import torch
from pandapower.pypower.makeYbus import branch_vectors

from networks import PROJECT_DIR
from physics import edge_flows, generation_cost, power_balance_residual, scatter_to_bus

PROC = PROJECT_DIR / "data" / "processed"
f32 = torch.float32


def build(case):
    ppc = np.load(PROC / f"network_{case}_ppc.npz")
    base = float(ppc["baseMVA"])
    bus_tab = pd.read_csv(PROC / f"network_{case}_buses.csv")
    br_tab = pd.read_csv(PROC / f"network_{case}_branches.csv", index_col=0)
    gen_tab = pd.read_csv(PROC / f"network_{case}_gens.csv", index_col=0)
    nb, nbr, ng = len(bus_tab), len(br_tab), len(gen_tab)

    # ---------------- static structure ----------------
    fb = br_tab.from_bus.to_numpy() - 1
    tb = br_tab.to_bus.to_numpy() - 1
    branch = ppc["branch"]
    assert np.array_equal(branch[:, 0].real.astype(int), fb) and \
           np.array_equal(branch[:, 1].real.astype(int), tb), "branch table / ppc mismatch"
    Ytt, Yff, Yft, Ytf = branch_vectors(branch, nbr)

    edge_index = np.stack([np.r_[fb, tb], np.r_[tb, fb]])
    y_self = np.r_[Yff, Ytt]
    y_mut = np.r_[Yft, Ytf]
    y_phys = np.stack([y_self.real, y_self.imag, y_mut.real, y_mut.imag], 1)

    tap = br_tab.tap.to_numpy().copy()
    tap[tap == 0] = 1.0
    ea = np.stack([br_tab.r_pu, br_tab.x_pu, br_tab.b_pu, br_tab.rate_mva / base, tap,
                   np.deg2rad(br_tab.shift_deg), (br_tab.type == "trafo").astype(float)], 1)
    edge_attr = np.c_[np.r_[ea, ea], np.r_[np.ones(nbr), np.zeros(nbr)]]

    btype = ppc["bus"][:, 1].real.astype(int)          # 1 = PQ, 2 = PV, 3 = slack
    onehot = np.stack([(btype == 1), (btype == 2), (btype == 3)], 1).astype(float)
    assert onehot.sum(1).min() == 1 and onehot[:, 2].sum() == 1, "unexpected bus types"
    bus_static = np.c_[bus_tab.gs_mw / base, bus_tab.bs_mvar / base,
                       bus_tab.vm_min_pu, bus_tab.vm_max_pu, onehot]

    gen_bus = gen_tab.bus.to_numpy() - 1
    gen_x = np.stack([gen_tab.p_min_mw / base, gen_tab.p_max_mw / base,
                      gen_tab.q_min_mvar / base, gen_tab.q_max_mvar / base,
                      gen_tab.c2 * base**2, gen_tab.c1 * base, gen_tab.c0,
                      (gen_tab.type == "ext_grid").astype(float)], 1)

    # ---------------- per-hour data ----------------
    lab = pd.read_parquet(PROC / f"acopf_labels_{case}.parquet")
    sc = pd.read_parquet(PROC / f"load_scenarios_{case}.parquet").reindex(lab.index)
    split = pd.read_csv(PROC / "splits.csv", index_col=0, parse_dates=True).split.reindex(lab.index)
    assert lab.converged.all() and split.notna().all()
    T = len(lab)

    pd_ = np.zeros((T, nb)); qd_ = np.zeros((T, nb))
    for c in sc.columns:
        if c.startswith("P_bus"):
            pd_[:, int(c[5:]) - 1] += sc[c].to_numpy() / base
        elif c.startswith("Q_bus"):
            qd_[:, int(c[5:]) - 1] += sc[c].to_numpy() / base
    col = lambda pre, n, off=0: lab[[f"{pre}{i + off}" for i in range(n)]].to_numpy()
    vm = col("Vm_bus", nb, 1)
    va = np.deg2rad(col("Va_bus", nb, 1))
    pg = col("Pg_gen", ng) / base
    qg = col("Qg_gen", ng) / base
    flow_p = np.c_[col("Pf_br", nbr), col("Pt_br", nbr)] / base
    flow_q = np.c_[col("Qf_br", nbr), col("Qt_br", nbr)] / base

    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))
    data = dict(
        case=case, baseMVA=base, timestamps=[str(x) for x in lab.index],
        split=split.tolist(), n_bus=nb, n_gen=ng, n_branch=nbr,
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_attr=t(edge_attr).to(f32), y_phys=t(y_phys).to(f32),
        gen_bus=torch.as_tensor(gen_bus, dtype=torch.long),
        bus_static=t(bus_static).to(f32), gen_x=t(gen_x).to(f32),
        pd=t(pd_).to(f32), qd=t(qd_).to(f32), vm=t(vm).to(f32), va=t(va).to(f32),
        pg=t(pg).to(f32), qg=t(qg).to(f32),
        flow_p=t(flow_p).to(f32), flow_q=t(flow_q).to(f32),
        cost=t(lab.cost.to_numpy()).to(f32),
    )

    # ---------------- verification (float64) ----------------
    ei = data["edge_index"]
    yp = t(y_phys)
    p_c, q_c = edge_flows(t(vm), t(va), ei, yp)
    flow_err = max((p_c - t(flow_p)).abs().max().item(), (q_c - t(flow_q)).abs().max().item())
    pg_bus = scatter_to_bus(t(pg), data["gen_bus"], nb)
    qg_bus = scatter_to_bus(t(qg), data["gen_bus"], nb)
    dp, dq = power_balance_residual(t(vm), t(va), t(pd_), t(qd_), t(bus_static[:, 0]),
                                    t(bus_static[:, 1]), pg_bus, qg_bus, ei, yp)
    cost_c = generation_cost(t(pg), t(gen_x[:, 4]), t(gen_x[:, 5]), t(gen_x[:, 6])).sum(-1)
    cost_err = ((cost_c - t(lab.cost.to_numpy())).abs() / t(lab.cost.to_numpy())).max().item()
    s_edge = torch.sqrt(p_c**2 + q_c**2)
    rate = t(edge_attr[:, 3])
    max_load = (s_edge / rate).max().item() * 100

    print(f"\n[{case}] {T} hours | {nb} buses, {ng} generators, {nbr} branches "
          f"({2 * nbr} directed edges)")
    print(f"  split sizes: " + ", ".join(f"{s}={split.tolist().count(s)}" for s in ["train", "val", "test"]))
    print(f"  bus types: PQ={int(onehot[:, 0].sum())}, PV={int(onehot[:, 1].sum())}, slack={int(onehot[:, 2].sum())}")
    print(f"  CHECK branch flows recomputed vs labels : max error {flow_err * base:.2e} MW/Mvar")
    print(f"  CHECK nodal power balance (physics loss): max |dP| {dp.abs().max().item() * base:.2e} MW, "
          f"max |dQ| {dq.abs().max().item() * base:.2e} Mvar")
    print(f"  CHECK cost recomputed from Pg           : max relative error {cost_err:.2e}")
    print(f"  max apparent-power loading |S|/rate     : {max_load:.1f}% "
          f"(slightly >100% possible: pandapower limits current, not MVA)")
    out = PROC / f"graph_tensors_{case}.pt"
    torch.save(data, out)
    print(f"  saved {out}")


def build_aug(pool):
    """Augmented IEEE 14 samples: per-sample branch parameters, costs and N-1 edge mask."""
    from networks import get_net, load_buses
    src = PROC / f"aug_case14_{pool}.parquet"
    if not src.exists():
        print(f"\n[aug {pool}] {src.name} not found - skipped"); return
    base_t = torch.load(PROC / "graph_tensors_case14.pt", weights_only=False)
    ppc = np.load(PROC / "network_case14_ppc.npz")
    base = float(ppc["baseMVA"])
    nb, ng, nbr = base_t["n_bus"], base_t["n_gen"], base_t["n_branch"]
    df_all = pd.read_parquet(src)
    df = df_all[df_all.converged.fillna(False).astype(bool)]
    N = len(df)
    lbus = np.array(load_buses(get_net("case14"))) - 1
    col = lambda pre, n, off=0: df[[f"{pre}{i + off}" for i in range(n)]].to_numpy()

    pd_ = np.zeros((N, nb)); qd_ = np.zeros((N, nb))
    np.add.at(pd_.T, lbus, col("P_load", len(lbus)).T / base)
    np.add.at(qd_.T, lbus, col("Q_load", len(lbus)).T / base)
    fr, fx, fc = col("fr_br", nbr), col("fx_br", nbr), col("fc_gen", ng)
    outage = df.outage.to_numpy().astype(int)

    # per-sample admittances (same construction as the physics check in augment_case14.py)
    br = np.repeat(ppc["branch"][None], N, 0)
    br[:, :, 2] *= fr; br[:, :, 3] *= fx
    ys = np.zeros((N, 2 * nbr), complex); ym = np.zeros((N, 2 * nbr), complex)
    for i in range(N):
        Ytt, Yff, Yft, Ytf = branch_vectors(br[i], nbr)
        ys[i] = np.r_[Yff, Ytt]; ym[i] = np.r_[Yft, Ytf]
    mask = np.ones((N, 2 * nbr))
    has = outage >= 0
    mask[has, outage[has]] = 0; mask[has, outage[has] + nbr] = 0
    ys *= mask; ym *= mask
    y_phys = np.stack([ys.real, ys.imag, ym.real, ym.imag], -1)

    edge_attr = np.repeat(base_t["edge_attr"].double().numpy()[None], N, 0)
    edge_attr[:, :, 0] *= np.c_[fr, fr]
    edge_attr[:, :, 1] *= np.c_[fx, fx]
    gen_x = np.repeat(base_t["gen_x"].double().numpy()[None], N, 0)
    gen_x[:, :, 4] *= fc; gen_x[:, :, 5] *= fc

    vm = col("Vm_bus", nb, 1); va = np.deg2rad(col("Va_bus", nb, 1))
    pg = col("Pg_gen", ng) / base; qg = col("Qg_gen", ng) / base
    flow_p = np.c_[col("Pf_br", nbr), col("Pt_br", nbr)] / base
    flow_q = np.c_[col("Qf_br", nbr), col("Qt_br", nbr)] / base

    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64))
    data = dict(case="case14", pool=pool, baseMVA=base, timestamps=df.index.tolist(),
                base_hour=df.base_hour.astype(str).tolist(), split=["train"] * N,
                n_bus=nb, n_gen=ng, n_branch=nbr, edge_index=base_t["edge_index"],
                gen_bus=base_t["gen_bus"], bus_static=base_t["bus_static"],
                edge_attr=t(edge_attr).float(), y_phys=t(y_phys).float(), edge_mask=t(mask).bool(),
                gen_x=t(gen_x).float(), pd=t(pd_).float(), qd=t(qd_).float(),
                vm=t(vm).float(), va=t(va).float(), pg=t(pg).float(), qg=t(qg).float(),
                flow_p=t(flow_p).float(), flow_q=t(flow_q).float(),
                cost=t(df.cost.to_numpy()).float(), outage=torch.as_tensor(outage))

    # verification with the per-sample admittances (float64)
    ei, yp = data["edge_index"], t(y_phys)
    p_c, q_c = edge_flows(t(vm), t(va), ei, yp)
    flow_err = max((p_c - t(flow_p)).abs().max().item(), (q_c - t(flow_q)).abs().max().item())
    bs = base_t["bus_static"].double()
    dp, dq = power_balance_residual(t(vm), t(va), t(pd_), t(qd_), bs[:, 0], bs[:, 1],
                                    scatter_to_bus(t(pg), data["gen_bus"], nb),
                                    scatter_to_bus(t(qg), data["gen_bus"], nb), ei, yp)
    cost_c = generation_cost(t(pg), t(gen_x[:, :, 4]), t(gen_x[:, :, 5]), t(gen_x[:, :, 6])).sum(-1)
    cost_err = ((cost_c - t(df.cost.to_numpy())).abs() / t(df.cost.to_numpy())).max().item()
    print(f"\n[aug {pool}] {N} converged samples (of {len(df_all)}), "
          f"{int(has.sum())} with an N-1 outage, from {df.base_hour.nunique()} train hours")
    print(f"  CHECK branch flows recomputed vs labels : max error {flow_err * base:.2e} MW/Mvar")
    print(f"  CHECK nodal power balance (physics loss): max |dP| {dp.abs().max().item() * base:.2e} MW, "
          f"max |dQ| {dq.abs().max().item() * base:.2e} Mvar")
    print(f"  CHECK cost recomputed from Pg           : max relative error {cost_err:.2e}")
    out = PROC / f"graph_tensors_case14_aug_{pool}.pt"
    torch.save(data, out)
    print(f"  saved {out}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--aug-only", action="store_true", help="only (re)build the augmented pools")
    ap.add_argument("--pools", nargs="+", default=["L", "LP", "LPT"])
    ap.add_argument("--cases", nargs="+", default=["case14", "case30"],
                    help="grids to build (add case57 for the revision experiment)")
    ap.add_argument("--no-aug", action="store_true", help="skip the IEEE 14 augmented pools")
    a = ap.parse_args()
    if not a.aug_only:
        for c in a.cases:
            build(c)
    if not a.no_aug:
        for p in a.pools:
            build_aug(p)
