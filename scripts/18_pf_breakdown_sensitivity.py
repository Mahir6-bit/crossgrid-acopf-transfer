"""
18_pf_breakdown_sensitivity.py   (revision: Supplementary Tables S4 and S5)

Re-runs the AC power-flow correction of pf_postprocess.py hour by hour and reports
  S4  breakdown: for every model, the share of test hours that are infeasible and, among the
      infeasible hours, how often each limit type is violated (voltage, branch thermal,
      generator reactive, slack active, slack reactive, generator active), plus the worst
      branch loading and the worst voltage violation in those hours.
  S5  sensitivity: the feasibility rate under three correction variants
        single      the paper's procedure (slack generator absorbs the whole imbalance)
        distributed all dispatchable generators share the imbalance in proportion to their
                    capacity (pandapower distributed slack); generator P limits are then checked too
        zero_tol    single slack, but zero tolerances (0 p.u., 100.0 % loading, 0 MW/Mvar)
      The DC-OPF dispatch is processed by exactly the same three variants.

The "single" variant must reproduce results\\pf_postprocess_runs.csv (printed as a check).

Run from the project folder (after training and pf_postprocess.py):
    python scripts\\18_pf_breakdown_sensitivity.py                 # all default groups, 1464 test hours
    python scripts\\18_pf_breakdown_sensitivity.py --limit 200     # quick test
Options: --groups "Fine-tune 5 % (E3)" "From scratch 100 % (E4)"   --variants single distributed
         --no-dc   --no-models (labels and DC-OPF only)
Writes results\\pf_breakdown_hours.csv (every hour), results\\pf_breakdown_runs.csv (per run),
       results\\tables\\table_S4_breakdown.csv and results\\tables\\table_S5_sensitivity.csv
"""

import argparse
import copy
import time

import numpy as np
import pandas as pd
import pandapower as pp
import torch
from pandapower.toolbox import replace_ext_grid_by_gen

from pf_postprocess import DEFAULT_GROUPS, RES, TOL_LOAD, TOL_PQ, TOL_V, Grid, predict

VARIANTS = {"single": dict(dist=False, tol=(TOL_V, TOL_LOAD, TOL_PQ)),
            "distributed": dict(dist=True, tol=(TOL_V, TOL_LOAD, TOL_PQ)),
            "zero_tol": dict(dist=False, tol=(0.0, 100.0, 0.0))}
KINDS = ["vm", "branch", "genq", "slack_p", "slack_q", "gen_p"]
GROUPS = {k: v for k, v in DEFAULT_GROUPS.items() if v[0] == "case30" and "Zero-shot" not in k}


def dist_net(grid):
    """Copy of the network whose external grid is replaced by a slack generator with a scheduled P,
    so that a distributed slack shares only the active-power MISMATCH (not the whole slack output)."""
    if not hasattr(grid, "_dist"):
        net2 = copy.deepcopy(grid.net)
        n_old = len(net2.gen)
        replace_ext_grid_by_gen(net2, slack=True)
        idx = list(net2.gen.index)
        grid._dist = (net2, idx[:n_old], idx[n_old])
    return grid._dist


def check(grid, h, pg_mw, vm, dist, tol):
    tv, tl, tpq = tol
    g = grid.gens
    if dist:
        net, orig, sl = dist_net(grid)
        s_ = grid.sc.iloc[h]
        net.load["p_mw"] = [s_[f"P_bus{b}"] for b in grid.lbus]
        net.load["q_mvar"] = [s_[f"Q_bus{b}"] for b in grid.lbus]
        net.gen.loc[orig, "p_mw"] = pg_mw[1:]
        net.gen.loc[orig, "vm_pu"] = vm[grid.gen_bus]
        net.gen.loc[sl, "p_mw"] = pg_mw[0]
        net.gen.loc[sl, "vm_pu"] = vm[grid.slack_bus]
        rng = (g.p_max_mw - g.p_min_mw).clip(lower=0).to_numpy()   # participation ~ active-power range
        net.gen.loc[orig, "slack_weight"] = rng[1:]
        net.gen.loc[sl, "slack_weight"] = rng[0]
    else:
        net = grid.net
        grid.set_loads(h)
        net.gen["p_mw"] = pg_mw[1:]
        net.gen["vm_pu"] = vm[grid.gen_bus]
        net.ext_grid["vm_pu"] = vm[grid.slack_bus]
    try:
        pp.runpp(net, init="dc", enforce_q_lims=False, max_iteration=30, distributed_slack=dist)
        ok = bool(net.converged)
    except Exception:
        ok = False
    r = dict(converged=ok)
    if not ok:
        r["feasible"] = False
        return r
    vmr = net.res_bus.vm_pu.to_numpy()
    lo, hi = net.bus.min_vm_pu.to_numpy(), net.bus.max_vm_pu.to_numpy()
    load = net.res_line.loading_percent.to_numpy()
    if len(net.trafo):
        load = np.r_[load, net.res_trafo.loading_percent.to_numpy()]
    if dist:
        p = np.r_[net.res_gen.p_mw.loc[sl], net.res_gen.p_mw.loc[orig].to_numpy()]
        q = np.r_[net.res_gen.q_mvar.loc[sl], net.res_gen.q_mvar.loc[orig].to_numpy()]
    else:
        p = np.r_[net.res_ext_grid.p_mw.to_numpy(), net.res_gen.p_mw.to_numpy()]
        q = np.r_[net.res_ext_grid.q_mvar.to_numpy(), net.res_gen.q_mvar.to_numpy()]
    pmin, pmax = g.p_min_mw.to_numpy(), g.p_max_mw.to_numpy()
    qmin, qmax = g.q_min_mvar.to_numpy(), g.q_max_mvar.to_numpy()
    r.update(
        vm=bool(((vmr < lo - tv) | (vmr > hi + tv)).any()),
        branch=bool((load > tl).any()),
        genq=bool(((q[1:] < qmin[1:] - tpq) | (q[1:] > qmax[1:] + tpq)).any()),
        slack_p=bool((p[0] < pmin[0] - tpq) | (p[0] > pmax[0] + tpq)),
        slack_q=bool((q[0] < qmin[0] - tpq) | (q[0] > qmax[0] + tpq)),
        gen_p=bool(dist and ((p[1:] < pmin[1:] - tpq) | (p[1:] > pmax[1:] + tpq)).any()),
        max_loading_pct=float(load.max()),
        max_v_violation_pu=float(np.maximum(lo - vmr, vmr - hi).clip(min=0).max()),
        cost_pf=grid.cost(p), cost_opt=float(grid.lab.cost.iloc[h]))
    r["feasible"] = not any(r[k] for k in KINDS)
    return r


def dc_dispatch(grid):
    """pandapower DC-OPF per hour -> generator P (slack first) and case-default voltage set-points."""
    net = grid.net
    vm_set = np.ones(len(net.bus))
    vm_set[grid.gen_bus] = net.gen.vm_pu.to_numpy()
    vm_set[grid.slack_bus] = net.ext_grid.vm_pu.iloc[0]
    pg = []
    for h in range(len(grid.ts)):
        grid.set_loads(h)
        net.gen["p_mw"] = 0.0
        try:
            pp.rundcopp(net)
            ok = bool(net.OPF_converged)
        except Exception:
            ok = False
        pg.append(np.r_[net.res_ext_grid.p_mw.to_numpy(), net.res_gen.p_mw.to_numpy()] if ok else None)
    return pg, vm_set


def evaluate_set(grid, pg, vm, variants, label, run):
    rows = []
    for vname in variants:
        v = VARIANTS[vname]
        for h in range(len(grid.ts)):
            if pg[h] is None:
                rows.append(dict(group=label, run=run, variant=vname, hour=h, converged=False, feasible=False))
                continue
            vmh = vm[h] if vm.ndim == 2 else vm
            r = check(grid, h, pg[h], vmh, v["dist"], v["tol"])
            r.update(group=label, run=run, variant=vname, hour=h)
            rows.append(r)
    return rows


def per_run(hours):
    out = []
    for (grp, run, var), d in hours.groupby(["group", "run", "variant"], sort=False):
        inf = d[~d.feasible.astype(bool)]
        conv = d[d.converged.astype(bool)]
        row = dict(group=grp, run=run, variant=var, n_hours=len(d),
                   pf_converged_pct=100 * d.converged.mean(), feasible_pct=100 * d.feasible.mean(),
                   infeasible_pct=100 * (1 - d.feasible.mean()))
        for k in KINDS:
            row[f"{k}_in_infeasible_pct"] = 100 * inf[k].astype(float).mean() if len(inf) and k in inf else np.nan
        if len(inf) and "max_loading_pct" in inf:
            row["worst_loading_in_infeasible_pct"] = inf.max_loading_pct.max()
            row["worst_v_violation_in_infeasible_pu"] = inf.max_v_violation_pu.max()
        if len(conv):
            row["cost_gap_pf_abs_pct"] = ((conv.cost_pf - conv.cost_opt).abs() / conv.cost_opt * 100).mean()
        out.append(row)
    return pd.DataFrame(out)


def ms(x):
    x = pd.Series(x).dropna()
    if not len(x):
        return "–"
    return f"{x.mean():.2f} ± {x.std():.2f}" if len(x) > 1 else f"{x.mean():.2f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--groups", nargs="+", default=list(GROUPS))
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--no-dc", action="store_true")
    ap.add_argument("--no-models", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grid = Grid("case30", a.limit)
    rows = []
    t0 = time.time()
    # pipeline check: the AC-OPF labels
    ng, nb = len(grid.gens), len(grid.net.bus)
    pg_lab = [np.array([grid.lab.iloc[h][f"Pg_gen{k}"] for k in range(ng)]) for h in range(len(grid.ts))]
    vm_lab = np.array([[grid.lab.iloc[h][f"Vm_bus{b + 1}"] for b in range(nb)] for h in range(len(grid.ts))])
    rows += evaluate_set(grid, pg_lab, vm_lab, a.variants, "AC-OPF labels (check)", "oracle")
    if not a.no_models:
        for label in a.groups:
            case, pattern = DEFAULT_GROUPS[label]
            for ck in sorted(RES.glob(f"{pattern}/best.pt")):
                vm, pg = predict(ck, grid, dev)
                rows += evaluate_set(grid, list(pg), vm, a.variants, label, ck.parent.name)
                print(f"[{label}] {ck.parent.name} done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    if not a.no_dc:
        pg_dc, vm_dc = dc_dispatch(grid)
        rows += evaluate_set(grid, pg_dc, vm_dc, a.variants, "DC-OPF baseline", "dcopf")
        print(f"[DC-OPF] done ({(time.time() - t0) / 60:.1f} min)", flush=True)

    hours = pd.DataFrame(rows)
    hours.to_csv(RES / "pf_breakdown_hours.csv", index=False)
    runs = per_run(hours)
    runs.to_csv(RES / "pf_breakdown_runs.csv", index=False)

    # check against pf_postprocess (single slack must reproduce the paper)
    ref = RES / "pf_postprocess_runs.csv"
    if ref.exists() and "single" in a.variants and not a.limit:
        old = pd.read_csv(ref)
        old = old[old["case"] == "case30"].set_index("run").feasible_pct   # 'oracle'/'dcopf' exist for both grids
        new = runs[runs.variant == "single"].set_index("run").feasible_pct
        common = old.index.intersection(new.index)
        if len(common):
            print(f"Check vs pf_postprocess_runs.csv: max |feasible % difference| = "
                  f"{(old[common] - new[common]).abs().max():.3f} (expect 0)")

    (RES / "tables").mkdir(exist_ok=True)
    s = runs[runs.variant == "single"]
    s4 = []
    for grp, d in s.groupby("group", sort=False):
        s4.append({"Model": grp, "Seeds": len(d), "Infeasible hours (%)": ms(d.infeasible_pct),
                   **{f"{n} (% of infeasible hours)": ms(d[f"{k}_in_infeasible_pct"]) for k, n in
                      [("vm", "Voltage"), ("branch", "Branch thermal"), ("genq", "Generator Q"),
                       ("slack_p", "Slack P"), ("slack_q", "Slack Q")]},
                   "Worst loading in infeasible hours (%)": ms(d.get("worst_loading_in_infeasible_pct"))})
    pd.DataFrame(s4).to_csv(RES / "tables" / "table_S4_breakdown.csv", index=False)
    s5 = []
    for grp, d in runs.groupby("group", sort=False):
        row = {"Model": grp}
        for v in a.variants:
            dv = d[d.variant == v]
            row[f"Feasible (%), {v}"] = ms(dv.feasible_pct)
        if "distributed" in a.variants:
            row["|Cost gap| after PF (%), distributed"] = ms(d[d.variant == "distributed"].get("cost_gap_pf_abs_pct"))
        s5.append(row)
    pd.DataFrame(s5).to_csv(RES / "tables" / "table_S5_sensitivity.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print("\nS4 (single slack):\n", pd.DataFrame(s4).to_string(index=False))
        print("\nS5:\n", pd.DataFrame(s5).to_string(index=False))
    print(f"\nSaved pf_breakdown_hours.csv, pf_breakdown_runs.csv and tables\\table_S4_breakdown.csv, "
          f"table_S5_sensitivity.csv in {RES}")
