"""
08_generate_acopf_labels.py
Solves AC-OPF for every hour of the load scenarios and stores the labels
the GNN will learn, plus a static description of each network.

Outputs (per case, in data\\processed):
  acopf_labels_<case>.parquet   one row per hour (index = timestamp):
      converged, time_s, cost, losses_mw, max_mismatch_mva
      Vm_bus{b} [pu], Va_bus{b} [deg], lamP_bus{b} [LMP, $/MWh]
      Pg_gen{k} [MW], Qg_gen{k} [Mvar]
      Pf_br{k}, Qf_br{k}, Pt_br{k}, Qt_br{k} [MW/Mvar], loading_br{k} [%]
  network_<case>_buses.csv / _gens.csv / _branches.csv   static tables (1-based bus numbers)
  network_<case>_ppc.npz        MATPOWER-format bus/branch/gen arrays in per-unit (for physics losses)
  network_<case>.json           full pandapower network (exact reproducibility)

max_mismatch_mva is an independent physics check: S = V * conj(Ybus V) must equal
(Pg - Pd) + j(Qg - Qd) at every bus. Values ~1e-3 MVA or below mean the labels are consistent.

Work is split into chunks saved under data\\processed\\labels_<case>\\ so the run can be
stopped (Ctrl+C) and restarted - finished chunks are skipped.

Run from the project folder with the venv active:
    python scripts\\08_generate_acopf_labels.py --limit 48      (quick test: first 48 hours)
    python scripts\\08_generate_acopf_labels.py                  (full run, both cases)
Options: --case case14|case30|case57   --workers N   --limit N
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.pypower.makeYbus import makeYbus

from networks import PROJECT_DIR, get_net, load_buses

PROC_DIR = PROJECT_DIR / "data" / "processed"
CHUNK = 240


# ---------------------------------------------------------------- static network tables
def branch_endpoints(net):
    lines = net.line[["from_bus", "to_bus"]].to_numpy()
    trafos = net.trafo[["hv_bus", "lv_bus"]].to_numpy() if len(net.trafo) else np.empty((0, 2), int)
    return np.vstack([lines, trafos]).astype(int)


def gen_table(net):
    rows = []
    for et in ["ext_grid", "gen"]:
        for idx, g in net[et].iterrows():
            c = net.poly_cost[(net.poly_cost.et == et) & (net.poly_cost.element == idx)]
            c = c.iloc[0] if len(c) else None
            rows.append(dict(type=et, element=idx, bus=int(g.bus) + 1,
                             p_min_mw=g.min_p_mw, p_max_mw=g.max_p_mw,
                             q_min_mvar=g.min_q_mvar, q_max_mvar=g.max_q_mvar,
                             c2=c.cp2_eur_per_mw2 if c is not None else 0.0,
                             c1=c.cp1_eur_per_mw if c is not None else 0.0,
                             c0=c.cp0_eur if c is not None else 0.0))
    g = pd.DataFrame(rows)
    g.index.name = "gen"
    return g


def save_static(case):
    net = get_net(case)
    pp.runopp(net)
    ppc = net._ppc
    ends = branch_endpoints(net)
    # our branch order (lines, then trafos) must equal pandapower's internal branch order
    assert np.array_equal(ends, ppc["branch"][:, :2].real.astype(int)), "branch order mismatch"
    assert np.array_equal(ppc["bus"][:, 0].astype(int), net.bus.index.to_numpy()), "bus order mismatch"

    bus = pd.DataFrame({
        "bus": net.bus.index + 1, "vn_kv": net.bus.vn_kv.values,
        "vm_min_pu": ppc["bus"][:, 12], "vm_max_pu": ppc["bus"][:, 11],
        "gs_mw": ppc["bus"][:, 4], "bs_mvar": ppc["bus"][:, 5],
    })
    br = pd.DataFrame({
        "from_bus": ends[:, 0] + 1, "to_bus": ends[:, 1] + 1,
        "type": ["line"] * len(net.line) + ["trafo"] * len(net.trafo),
        "r_pu": ppc["branch"][:, 2].real, "x_pu": ppc["branch"][:, 3].real,
        "b_pu": ppc["branch"][:, 4].real, "rate_mva": ppc["branch"][:, 5].real,
        "tap": ppc["branch"][:, 8].real, "shift_deg": ppc["branch"][:, 9].real,
    })
    br.index.name = "branch"
    bus.to_csv(PROC_DIR / f"network_{case}_buses.csv", index=False)
    br.to_csv(PROC_DIR / f"network_{case}_branches.csv")
    gen_table(net).to_csv(PROC_DIR / f"network_{case}_gens.csv")
    np.savez(PROC_DIR / f"network_{case}_ppc.npz", baseMVA=ppc["baseMVA"],
             bus=ppc["bus"], branch=ppc["branch"], gen=ppc["gen"])
    pp.to_json(get_net(case), str(PROC_DIR / f"network_{case}.json"))
    print(f"[{case}] static tables saved: {len(bus)} buses, {len(br)} branches "
          f"({len(net.line)} lines + {len(net.trafo)} trafos), {len(net.gen) + len(net.ext_grid)} generators")


# ---------------------------------------------------------------- worker
def solve_chunk(case, chunk_id, positions):
    import logging, warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("pandapower").setLevel(logging.CRITICAL)

    sc = pd.read_parquet(PROC_DIR / f"load_scenarios_{case}.parquet")
    net = get_net(case)
    lbus = load_buses(net)
    nb = len(net.bus)
    pp.runopp(net)                                   # build ppc once to get Ybus
    Ybus, _, _ = makeYbus(net._ppc["baseMVA"], net._ppc["bus"], net._ppc["branch"])
    base = net._ppc["baseMVA"]
    gens = [("ext_grid", i, int(b)) for i, b in net.ext_grid.bus.items()] + \
           [("gen", i, int(b)) for i, b in net.gen.bus.items()]
    load_idx = net.load.bus.to_numpy()
    has_trafo = len(net.trafo) > 0

    rows = []
    for i in positions:
        s = sc.iloc[i]
        p = np.array([s[f"P_bus{b}"] for b in lbus])
        q = np.array([s[f"Q_bus{b}"] for b in lbus])
        net.load["p_mw"], net.load["q_mvar"] = p, q
        t0 = time.perf_counter()
        try:
            pp.runopp(net)
            ok = bool(net.OPF_converged)
        except Exception:
            ok = False
        r = {"timestamp": sc.index[i], "converged": ok, "time_s": time.perf_counter() - t0}
        if ok:
            vm, va = net.res_bus.vm_pu.to_numpy(), net.res_bus.va_degree.to_numpy()
            pg = np.array([net[f"res_{et}"].p_mw[idx] for et, idx, _ in gens])
            qg = np.array([net[f"res_{et}"].q_mvar[idx] for et, idx, _ in gens])
            # physics check
            V = vm * np.exp(1j * np.deg2rad(va))
            S_calc = V * np.conj(Ybus @ V) * base
            S_inj = np.zeros(nb, complex)
            np.add.at(S_inj, [b for _, _, b in gens], pg + 1j * qg)
            np.add.at(S_inj, load_idx, -(p + 1j * q))
            r.update(cost=net.res_cost, max_mismatch_mva=np.abs(S_calc - S_inj).max())
            rl, rt = net.res_line, (net.res_trafo if has_trafo else None)
            pf = np.r_[rl.p_from_mw, rt.p_hv_mw] if has_trafo else rl.p_from_mw.to_numpy()
            qf = np.r_[rl.q_from_mvar, rt.q_hv_mvar] if has_trafo else rl.q_from_mvar.to_numpy()
            pt = np.r_[rl.p_to_mw, rt.p_lv_mw] if has_trafo else rl.p_to_mw.to_numpy()
            qt = np.r_[rl.q_to_mvar, rt.q_lv_mvar] if has_trafo else rl.q_to_mvar.to_numpy()
            ld = np.r_[rl.loading_percent, rt.loading_percent] if has_trafo else rl.loading_percent.to_numpy()
            r["losses_mw"] = float((pf + pt).sum())
            for b in range(nb):
                r[f"Vm_bus{b+1}"], r[f"Va_bus{b+1}"] = vm[b], va[b]
                r[f"lamP_bus{b+1}"] = net.res_bus.lam_p.iloc[b]
            for k in range(len(gens)):
                r[f"Pg_gen{k}"], r[f"Qg_gen{k}"] = pg[k], qg[k]
            for k in range(len(pf)):
                r[f"Pf_br{k}"], r[f"Qf_br{k}"] = pf[k], qf[k]
                r[f"Pt_br{k}"], r[f"Qt_br{k}"] = pt[k], qt[k]
                r[f"loading_br{k}"] = ld[k]
        rows.append(r)
    return chunk_id, pd.DataFrame(rows).set_index("timestamp")


# ---------------------------------------------------------------- driver
def run(case, workers, limit):
    save_static(case)
    n = len(pd.read_parquet(PROC_DIR / f"load_scenarios_{case}.parquet", columns=["lambda_mapped_mean"]))
    n = min(n, limit) if limit else n
    part_dir = PROC_DIR / (f"labels_{case}" + (f"_test{limit}" if limit else ""))
    part_dir.mkdir(parents=True, exist_ok=True)
    chunks = [(c, list(range(s, min(s + CHUNK, n)))) for c, s in enumerate(range(0, n, CHUNK))]
    todo = [(c, pos) for c, pos in chunks if not (part_dir / f"part_{c:03d}.parquet").exists()]
    print(f"[{case}] {n} hours in {len(chunks)} chunks; {len(chunks) - len(todo)} already done; "
          f"running {len(todo)} with {workers} worker(s)")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(solve_chunk, case, c, pos) for c, pos in todo]
        for k, f in enumerate(as_completed(futs), 1):
            cid, df = f.result()
            df.to_parquet(part_dir / f"part_{cid:03d}.parquet")
            el = time.time() - t0
            print(f"  chunk {cid:03d} saved ({k}/{len(todo)}) | elapsed {el/60:.1f} min | "
                  f"est. remaining {el/k*(len(todo)-k)/60:.1f} min")

    parts = sorted(part_dir.glob("part_*.parquet"))
    if len(parts) < len(chunks):
        print(f"[{case}] only {len(parts)}/{len(chunks)} chunks present - rerun to finish.")
        return
    lab = pd.concat([pd.read_parquet(p) for p in parts]).sort_index()
    out = PROC_DIR / (f"acopf_labels_{case}" + (f"_test{limit}" if limit else "") + ".parquet")
    lab.to_parquet(out)

    ok = lab[lab.converged]
    print(f"\n[{case}] SUMMARY -> {out.name}")
    print(f"  hours: {len(lab)} | converged: {len(ok)} ({100*len(ok)/len(lab):.2f}%)")
    print(f"  solve time: mean {lab.time_s.mean():.3f} s, median {lab.time_s.median():.3f} s, "
          f"max {lab.time_s.max():.3f} s")
    if len(ok):
        print(f"  cost: {ok.cost.min():.1f} - {ok.cost.max():.1f} | losses: "
              f"{ok.losses_mw.min():.2f} - {ok.losses_mw.max():.2f} MW")
        print(f"  physics check max |S mismatch|: median {ok.max_mismatch_mva.median():.2e}, "
              f"max {ok.max_mismatch_mva.max():.2e} MVA")
        ld = ok.filter(like="loading_br")
        print(f"  hours with any branch >= 99% loaded: {(ld.max(axis=1) >= 99).sum()}")
        vm = ok.filter(like="Vm_bus")
        print(f"  Vm range: {vm.min().min():.3f} - {vm.max().max():.3f} pu")
    if len(ok) < len(lab):
        print("  failed hours:", list(lab.index[~lab.converged][:20]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["case14", "case30", "case57"], default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0, help="only the first N hours (testing)")
    a = ap.parse_args()
    for c in ([a.case] if a.case else ["case14", "case30"]):
        run(c, a.workers, a.limit)
