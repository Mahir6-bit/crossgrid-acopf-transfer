"""
augment_case14.py
Research plan Step 2.3 - data augmentation on the SOURCE grid (IEEE 14) only.
Every augmented sample is derived from one IEEE 14 TRAIN hour (Jan-Aug), so validation/test
months are never touched. IEEE 30 is not augmented (target stays realistic).

Three cumulative augmentation pools (for the novelty-3 ablation: what does each kind add?):
  L    load      : global scaling U(0.7, 1.2) x correlated per-bus noise (sigma 10 %, rho 0.5,
                   clipped to +-20 %); Q scaled with P (constant power factor)
  LP   + params  : L + line/transformer r and x each x U(0.8, 1.2) (independent per branch)
                   + generator cost coefficients c1, c2 x U(0.6, 1.4) per generator
  LPT  + topology: LP + N-1: one random branch out of service (bridges excluded - no islanding)
Default: 4 variants per train hour per pool -> 23,328 samples per pool; with the 5,832 original
hours each training arm has ~29k samples (plan target 30-50k).

Infeasible / non-converged samples are kept in the file with converged=False and reported
(plan Step 2.2); they are dropped when the graph dataset is built.

Outputs (data\\processed):
  aug_case14_<pool>.parquet   one row per sample: definition (base hour, factors, outage) + labels
                              (same label columns as acopf_labels_case14.parquet)
  results\\augmentation_table.csv   ranges / distributions / seeds for the paper (Table 3)

Run from the project folder with the venv active:
    python scripts\\09_augment_source_grid.py --limit 40       (quick test: 40 samples per pool)
    python scripts\\09_augment_source_grid.py                  (full run, ~70k OPF solves)
Options: --pools L LP LPT  --variants 4  --workers N  --seed 2026
"""

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import networkx as nx
import numpy as np
import pandas as pd
import pandapower as pp
from pandapower.pypower.makeYbus import branch_vectors

from networks import PROJECT_DIR, get_net, load_buses

PROC = PROJECT_DIR / "data" / "processed"
RES = PROJECT_DIR / "results"
CHUNK = 250
POOLS = {"L": (False, False), "LP": (True, False), "LPT": (True, True)}   # (params, topology)
RANGES = dict(gscale=(0.7, 1.2), noise_sigma=0.10, noise_rho=0.5, noise_clip=0.20,
              rx=(0.8, 1.2), cost=(0.6, 1.4))


# ---------------------------------------------------------------- helpers
def branch_ends(net):
    return np.r_[net.line[["from_bus", "to_bus"]].to_numpy(),
                 net.trafo[["hv_bus", "lv_bus"]].to_numpy()].astype(int)


def non_bridges(net):
    ends = branch_ends(net)
    g = nx.MultiGraph()
    for k, (a, b) in enumerate(ends):
        g.add_edge(a, b, key=k)
    ok = []
    for k, (a, b) in enumerate(ends):
        h = g.copy(); h.remove_edge(a, b, key=k)
        if nx.is_connected(h):
            ok.append(k)
    return ok


def sample_definitions(pool, variants, seed, limit):
    params, topo = POOLS[pool]
    rng = np.random.default_rng(seed)
    split = pd.read_csv(PROC / "splits.csv", index_col=0).split
    train_hours = split.index[split == "train"].tolist()
    net = get_net("case14")
    n_load, n_br = len(net.load), len(net.line) + len(net.trafo)
    n_gen = len(net.ext_grid) + len(net.gen)
    cand = non_bridges(net)
    rows = []
    for h in train_hours:
        for v in range(variants):
            z = RANGES["noise_rho"] * rng.standard_normal() + \
                np.sqrt(1 - RANGES["noise_rho"] ** 2) * rng.standard_normal(n_load)
            eps = np.clip(RANGES["noise_sigma"] * z, -RANGES["noise_clip"], RANGES["noise_clip"])
            r = dict(pool=pool, base_hour=h, variant=v, gscale=rng.uniform(*RANGES["gscale"]),
                     outage=-1)
            r.update({f"eps_load{k}": eps[k] for k in range(n_load)})
            fr = rng.uniform(*RANGES["rx"], n_br) if params else np.ones(n_br)
            fx = rng.uniform(*RANGES["rx"], n_br) if params else np.ones(n_br)
            fc = rng.uniform(*RANGES["cost"], n_gen) if params else np.ones(n_gen)
            r.update({f"fr_br{k}": fr[k] for k in range(n_br)})
            r.update({f"fx_br{k}": fx[k] for k in range(n_br)})
            r.update({f"fc_gen{k}": fc[k] for k in range(n_gen)})
            if topo:
                r["outage"] = int(rng.choice(cand))
            rows.append(r)
    df = pd.DataFrame(rows)
    df.index = [f"{pool}_{i:06d}" for i in range(len(df))]
    df.index.name = "sample"
    return df.iloc[:limit] if limit else df


def sample_y(base_branch, fr, fx, outage):
    """Per-sample branch admittances (pi-model) from the intact base ppc with r/x scaled."""
    br = base_branch.copy()
    br[:, 2] = br[:, 2] * fr
    br[:, 3] = br[:, 3] * fx
    Ytt, Yff, Yft, Ytf = branch_vectors(br, br.shape[0])
    if outage >= 0:
        Ytt[outage] = Yff[outage] = Yft[outage] = Ytf[outage] = 0
    return Ytt, Yff, Yft, Ytf


# ---------------------------------------------------------------- worker
def solve_chunk(pool, chunk_id, defs):
    import logging, warnings
    warnings.filterwarnings("ignore")
    logging.getLogger("pandapower").setLevel(logging.CRITICAL)

    sc = pd.read_parquet(PROC / "load_scenarios_case14.parquet")
    sc.index = sc.index.astype(str)
    base_ppc = np.load(PROC / "network_case14_ppc.npz")
    base_branch, base_mva = base_ppc["branch"], float(base_ppc["baseMVA"])
    net = get_net("case14")
    lbus = load_buses(net)
    nl, nt, nb = len(net.line), len(net.trafo), len(net.bus)
    r0, x0 = net.line.r_ohm_per_km.to_numpy().copy(), net.line.x_ohm_per_km.to_numpy().copy()
    vk0, vkr0 = net.trafo.vk_percent.to_numpy().copy(), net.trafo.vkr_percent.to_numpy().copy()
    vkx0 = np.sqrt(np.maximum(vk0 ** 2 - vkr0 ** 2, 0))
    gens = [("ext_grid", i, int(b)) for i, b in net.ext_grid.bus.items()] + \
           [("gen", i, int(b)) for i, b in net.gen.bus.items()]
    pc = net.poly_cost.set_index(["et", "element"])
    c1_0 = np.array([pc.loc[(et, i), "cp1_eur_per_mw"] for et, i, _ in gens])
    c2_0 = np.array([pc.loc[(et, i), "cp2_eur_per_mw2"] for et, i, _ in gens])
    cost_rows = [net.poly_cost.index[(net.poly_cost.et == et) & (net.poly_cost.element == i)][0]
                 for et, i, _ in gens]
    load_idx = net.load.bus.to_numpy()
    ends = branch_ends(net)

    rows = []
    for sid, d in defs.iterrows():
        s = sc.loc[str(d.base_hour)]
        f = d.gscale * (1 + np.array([d[f"eps_load{k}"] for k in range(len(lbus))]))
        p = np.array([s[f"P_bus{b}"] for b in lbus]) * f
        q = np.array([s[f"Q_bus{b}"] for b in lbus]) * f
        net.load["p_mw"], net.load["q_mvar"] = p, q
        fr = np.array([d[f"fr_br{k}"] for k in range(nl + nt)])
        fx = np.array([d[f"fx_br{k}"] for k in range(nl + nt)])
        fc = np.array([d[f"fc_gen{k}"] for k in range(len(gens))])
        net.line["r_ohm_per_km"], net.line["x_ohm_per_km"] = r0 * fr[:nl], x0 * fx[:nl]
        vkr, vkx = vkr0 * fr[nl:], vkx0 * fx[nl:]
        net.trafo["vkr_percent"], net.trafo["vk_percent"] = vkr, np.sqrt(vkr ** 2 + vkx ** 2)
        net.poly_cost.loc[cost_rows, "cp1_eur_per_mw"] = c1_0 * fc
        net.poly_cost.loc[cost_rows, "cp2_eur_per_mw2"] = c2_0 * fc
        net.line["in_service"], net.trafo["in_service"] = True, True
        o = int(d.outage)
        if o >= 0:
            (net.line if o < nl else net.trafo).loc[o if o < nl else o - nl, "in_service"] = False

        t0 = time.perf_counter()
        try:
            pp.runopp(net)
            ok = bool(net.OPF_converged)
        except Exception:
            ok = False
        r = {"sample": sid, "converged": ok, "time_s": time.perf_counter() - t0}
        if ok:
            vm, va = net.res_bus.vm_pu.to_numpy(), net.res_bus.va_degree.to_numpy()
            pg = np.array([net[f"res_{et}"].p_mw[i] for et, i, _ in gens])
            qg = np.array([net[f"res_{et}"].q_mvar[i] for et, i, _ in gens])
            # independent physics check with OUR per-sample admittances (as used in training)
            Ytt, Yff, Yft, Ytf = sample_y(base_branch, fr, fx, o)
            V = vm * np.exp(1j * np.deg2rad(va))
            fb, tb = ends[:, 0], ends[:, 1]
            Sf = V[fb] * np.conj(Yff * V[fb] + Yft * V[tb]) * base_mva
            St = V[tb] * np.conj(Ytf * V[fb] + Ytt * V[tb]) * base_mva
            Sbus = np.zeros(nb, complex)
            np.add.at(Sbus, fb, Sf); np.add.at(Sbus, tb, St)
            sh = (base_ppc["bus"][:, 4].real - 1j * base_ppc["bus"][:, 5].real) / base_mva
            Sbus += np.abs(V) ** 2 * sh * base_mva          # bus shunt consumption (MVA)
            inj = np.zeros(nb, complex)
            np.add.at(inj, [b for _, _, b in gens], pg + 1j * qg)
            np.add.at(inj, load_idx, -(p + 1j * q))
            r.update(cost=net.res_cost, max_mismatch_mva=np.abs(Sbus - inj).max())
            rl, rt = net.res_line, net.res_trafo
            pf = np.r_[rl.p_from_mw, rt.p_hv_mw]; qf = np.r_[rl.q_from_mvar, rt.q_hv_mvar]
            pt = np.r_[rl.p_to_mw, rt.p_lv_mw]; qt = np.r_[rl.q_to_mvar, rt.q_lv_mvar]
            ld = np.r_[rl.loading_percent, rt.loading_percent]
            r["losses_mw"] = float(np.nansum(pf + pt))
            for b in range(nb):
                r[f"Vm_bus{b+1}"], r[f"Va_bus{b+1}"] = vm[b], va[b]
            for k in range(len(gens)):
                r[f"Pg_gen{k}"], r[f"Qg_gen{k}"] = pg[k], qg[k]
            for k in range(nl + nt):
                r[f"Pf_br{k}"], r[f"Qf_br{k}"] = np.nan_to_num(pf[k]), np.nan_to_num(qf[k])
                r[f"Pt_br{k}"], r[f"Qt_br{k}"] = np.nan_to_num(pt[k]), np.nan_to_num(qt[k])
                r[f"loading_br{k}"] = np.nan_to_num(ld[k])
            for k in range(len(lbus)):
                r[f"P_load{k}"], r[f"Q_load{k}"] = p[k], q[k]
        rows.append(r)
    return chunk_id, pd.DataFrame(rows).set_index("sample")


# ---------------------------------------------------------------- driver
def run_pool(pool, variants, seed, workers, limit):
    defs = sample_definitions(pool, variants, seed, limit)
    tag = f"_test{limit}" if limit else ""
    part_dir = PROC / f"aug_case14_{pool}{tag}_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    chunks = [(c, defs.iloc[s:s + CHUNK]) for c, s in enumerate(range(0, len(defs), CHUNK))]
    todo = [(c, d) for c, d in chunks if not (part_dir / f"part_{c:03d}.parquet").exists()]
    print(f"\n[{pool}] {len(defs)} samples in {len(chunks)} chunks; {len(chunks) - len(todo)} done; "
          f"running {len(todo)} with {workers} workers")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(solve_chunk, pool, c, d) for c, d in todo]
        for k, fu in enumerate(as_completed(futs), 1):
            cid, df = fu.result()
            df.to_parquet(part_dir / f"part_{cid:03d}.parquet")
            el = time.time() - t0
            if k % 5 == 0 or k == len(todo):
                print(f"  {k}/{len(todo)} chunks | {el/60:.1f} min | est. remaining "
                      f"{el/k*(len(todo)-k)/60:.1f} min", flush=True)
    parts = sorted(part_dir.glob("part_*.parquet"))
    if len(parts) < len(chunks):
        print(f"[{pool}] incomplete - rerun to finish"); return
    lab = pd.concat([pd.read_parquet(p) for p in parts])
    out = defs.join(lab, how="left")
    out.to_parquet(PROC / f"aug_case14_{pool}{tag}.parquet")

    ok = out[out.converged]
    print(f"[{pool}] SUMMARY: {len(out)} samples, converged {len(ok)} ({100*len(ok)/len(out):.1f}%), "
          f"solve time median {out.time_s.median():.3f} s")
    print(f"  physics check max |S mismatch|: median {ok.max_mismatch_mva.median():.1e}, "
          f"max {ok.max_mismatch_mva.max():.1e} MVA")
    print(f"  total load: {ok.filter(like='P_load').sum(1).min():.0f} - "
          f"{ok.filter(like='P_load').sum(1).max():.0f} MW | cost {ok.cost.min():.0f} - {ok.cost.max():.0f}")
    bins = pd.cut(out.gscale, [0.7, 0.8, 0.9, 1.0, 1.1, 1.2], include_lowest=True)
    print("  convergence by global scale: " + "  ".join(
        f"{b}: {100*v:.0f}%" for b, v in out.groupby(bins, observed=True).converged.mean().items()))
    if (out.outage >= 0).any():
        fail_out = out[~out.converged & (out.outage >= 0)].outage.value_counts().head(5)
        print(f"  most frequent outages among failures: {dict(fail_out)}")
    ld = ok.filter(like="loading_br").max(axis=1)
    print(f"  samples with a branch >= 99% loaded: {(ld >= 99).sum()} ({100*(ld >= 99).mean():.1f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", nargs="+", default=["L", "LP", "LPT"], choices=list(POOLS))
    ap.add_argument("--variants", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for i, pool in enumerate(a.pools):
        run_pool(pool, a.variants, a.seed + i, a.workers, a.limit)
    RES.mkdir(exist_ok=True)
    pd.DataFrame([
        ("Global load scaling", "all pools", f"U{RANGES['gscale']}", "per sample"),
        ("Per-bus load noise", "all pools",
         f"N(0, {RANGES['noise_sigma']}) correlated rho={RANGES['noise_rho']}, clipped +-{RANGES['noise_clip']}",
         "per load, multiplicative; Q follows P"),
        ("Branch r, x", "LP, LPT", f"U{RANGES['rx']} independent", "per branch"),
        ("Generator cost c1, c2", "LP, LPT", f"U{RANGES['cost']}", "per generator"),
        ("N-1 outage", "LPT", "uniform over non-bridge branches (19 of 20)", "one per sample"),
        ("Variants per train hour", "each pool", str(a.variants), f"seed {a.seed}+pool index"),
    ], columns=["Perturbation", "Pools", "Distribution", "Applied"]).to_csv(
        RES / "augmentation_table.csv", index=False)
    print(f"\nAugmentation table saved to {RES / 'augmentation_table.csv'}")
