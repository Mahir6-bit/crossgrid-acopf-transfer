"""
pf_postprocess.py
E6 (power-flow post-processing) + DC-OPF baseline, research plan Steps 6-7.

For every test hour, the model's predictions are turned into an operating point that is
AC-feasible with respect to power balance:
    generator P set-points  = predicted Pg (non-slack generators)
    generator V set-points  = predicted Vm at generator buses and at the slack bus
    loads                   = the test-hour loads
then pandapower AC power flow (Newton-Raphson) is run; the slack bus balances the system.
After PF the operating point is checked against every operating limit:
    voltage magnitudes, branch loading (<= 100 %, pandapower current limit),
    generator reactive limits, slack active and reactive limits.
Reported per model (mean +- std over seeds):
    pf_converged_pct      share of hours where the power flow converged
    feasible_pct          share of hours with NO violation after PF (fully feasible)
    vm_viol_pct, branch_viol_pct, genq_viol_pct, slack_p_viol_pct, slack_q_viol_pct
                          share of hours with at least one violation of that type
    cost_gap_pf_abs_pct   |cost after PF - AC-OPF cost| / AC-OPF cost
    pf_ms                 power-flow time per hour (add GNN time from metrics.json)
Tolerances: 1e-3 pu voltage, 0.5 % branch loading, 0.1 MW / Mvar for generator limits.

Pipeline check: the AC-OPF labels themselves are pushed through the same routine first;
they must come out ~100 % feasible with ~0 % cost gap.

DC-OPF baseline (classical approximation): pandapower rundcopp per test hour -> Pg, Va
(Vm = generator set-points of the case). Its raw accuracy vs the AC-OPF labels and the same
PF post-processing are reported, plus its solve time.

Run from the project folder (after training):
    python scripts\\17_pf_postprocess.py
Options: --groups (see DEFAULT_GROUPS), --limit 200 (first N test hours, quick test), --no-dc
Writes results\\pf_postprocess_runs.csv and results\\pf_postprocess_summary.csv
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
RES = PROJECT_DIR / "results"
PROC = PROJECT_DIR / "data" / "processed"
TOL_V, TOL_LOAD, TOL_PQ = 1e-3, 100.5, 0.1

# label -> (evaluation grid, run-name glob)
DEFAULT_GROUPS = {
    "Source model on IEEE 14 (E1)": ("case14", "src14_s?"),
    "Zero-shot on IEEE 30 (E2)": ("case30", "src14_s?"),
    "Fine-tune 1 % (E3)": ("case30", "ft30_f0.01_s?"),
    "Fine-tune 5 % (E3)": ("case30", "ft30_f0.05_s?"),
    "Fine-tune 100 % (E3)": ("case30", "ft30_f1.0_s?"),
    "From scratch 5 % (E4)": ("case30", "scratch30_f0.05_s?"),
    "From scratch 100 % (E4)": ("case30", "scratch30_f1.0_s?"),
}


class Grid:
    """pandapower network + test-hour loads + AC-OPF reference costs for one case."""

    def __init__(self, case, limit):
        self.case = case
        self.ds = OPFGraphDataset(case, split="test")
        if limit:
            self.ds.idx = self.ds.idx[:limit]
        self.ts = [self.ds.t["timestamps"][i] for i in self.ds.idx]
        sc = pd.read_parquet(PROC / f"load_scenarios_{case}.parquet")
        sc.index = sc.index.astype(str)
        self.sc = sc.loc[self.ts]
        lab = pd.read_parquet(PROC / f"acopf_labels_{case}.parquet")
        lab.index = lab.index.astype(str)
        self.lab = lab.loc[self.ts]
        self.net = get_net(case)
        self.lbus = load_buses(self.net)
        g = pd.read_csv(PROC / f"network_{case}_gens.csv", index_col=0)
        self.gens = g
        self.slack_bus = int(self.net.ext_grid.bus.iloc[0])
        self.gen_bus = self.net.gen.bus.to_numpy()

    def set_loads(self, h):
        s = self.sc.iloc[h]
        self.net.load["p_mw"] = [s[f"P_bus{b}"] for b in self.lbus]
        self.net.load["q_mvar"] = [s[f"Q_bus{b}"] for b in self.lbus]

    def cost(self, p_mw):
        g = self.gens
        return float((g.c2 * p_mw ** 2 + g.c1 * p_mw + g.c0).sum())

    def check(self, h, pg_mw, vm):
        """Set point -> AC power flow -> violation flags."""
        net, g = self.net, self.gens
        self.set_loads(h)
        net.gen["p_mw"] = pg_mw[1:]
        net.gen["vm_pu"] = vm[self.gen_bus]
        net.ext_grid["vm_pu"] = vm[self.slack_bus]
        t0 = time.perf_counter()
        try:
            pp.runpp(net, init="dc", enforce_q_lims=False, max_iteration=30)
            ok = bool(net.converged)
        except Exception:
            ok = False
        r = dict(pf_ms=(time.perf_counter() - t0) * 1000, converged=ok)
        if not ok:
            return r
        vmr = net.res_bus.vm_pu.to_numpy()
        b = net.bus
        load = net.res_line.loading_percent.to_numpy()
        if len(net.trafo):
            load = np.r_[load, net.res_trafo.loading_percent.to_numpy()]
        p = np.r_[net.res_ext_grid.p_mw.to_numpy(), net.res_gen.p_mw.to_numpy()]
        q = np.r_[net.res_ext_grid.q_mvar.to_numpy(), net.res_gen.q_mvar.to_numpy()]
        r.update(
            vm=bool(((vmr < b.min_vm_pu - TOL_V) | (vmr > b.max_vm_pu + TOL_V)).any()),
            branch=bool((load > TOL_LOAD).any()),
            genq=bool(((q[1:] < g.q_min_mvar[1:] - TOL_PQ) | (q[1:] > g.q_max_mvar[1:] + TOL_PQ)).any()),
            slack_p=bool((p[0] < g.p_min_mw.iloc[0] - TOL_PQ) | (p[0] > g.p_max_mw.iloc[0] + TOL_PQ)),
            slack_q=bool((q[0] < g.q_min_mvar.iloc[0] - TOL_PQ) | (q[0] > g.q_max_mvar.iloc[0] + TOL_PQ)),
            cost_pf=self.cost(p), cost_opt=float(self.lab.cost.iloc[h]),
        )
        r["feasible"] = not any(r[k] for k in ["vm", "branch", "genq", "slack_p", "slack_q"])
        return r


def summarize_hours(rows):
    d = pd.DataFrame(rows)
    out = dict(n_hours=len(d), pf_converged_pct=100 * d.converged.mean(), pf_ms=d.pf_ms.mean())
    c = d[d.converged]
    out["feasible_pct"] = 100 * c.feasible.sum() / len(d) if len(c) else 0.0
    for k in ["vm", "branch", "genq", "slack_p", "slack_q"]:
        out[f"{k}_viol_pct"] = 100 * c[k].mean() if len(c) else np.nan
    if len(c):
        gap = (c.cost_pf - c.cost_opt) / c.cost_opt * 100
        out["cost_gap_pf_abs_pct"] = gap.abs().mean()
    return out


@torch.no_grad()
def predict(ckpt, grid, dev):
    model, _ = load_checkpoint(ckpt, dev)
    model.eval()
    gb = GraphBatcher(grid.ds, dev)
    vm, pg = [], []
    for b in gb.batches(512):
        p = model(b)
        n = b.num_graphs
        vm.append(p["vm"].view(n, -1).cpu()); pg.append(p["pg"].view(n, -1).cpu())
    return torch.cat(vm).numpy(), torch.cat(pg).numpy() * 100.0


def oracle(grid):
    """Pipeline check: the AC-OPF labels themselves must come out ~100 % feasible, ~0 % gap."""
    ng, nb = len(grid.gens), len(grid.net.bus)
    rows = []
    for h in range(len(grid.ts)):
        lab = grid.lab.iloc[h]
        pg = np.array([lab[f"Pg_gen{k}"] for k in range(ng)])
        vm = np.array([lab[f"Vm_bus{b + 1}"] for b in range(nb)])
        rows.append(grid.check(h, pg, vm))
    return summarize_hours(rows)


def dc_baseline(grid):
    net = grid.net
    rows, acc = [], []
    vm_set = np.ones(len(net.bus))
    vm_set[grid.gen_bus] = net.gen.vm_pu.to_numpy()
    vm_set[grid.slack_bus] = net.ext_grid.vm_pu.iloc[0]
    for h in range(len(grid.ts)):
        grid.set_loads(h)
        net.gen["p_mw"] = 0.0
        t0 = time.perf_counter()
        try:
            pp.rundcopp(net)
            ok = bool(net.OPF_converged)
        except Exception:
            ok = False
        dc_ms = (time.perf_counter() - t0) * 1000
        if not ok:
            rows.append(dict(converged=False, pf_ms=0.0)); continue
        pg = np.r_[net.res_ext_grid.p_mw.to_numpy(), net.res_gen.p_mw.to_numpy()]
        va = net.res_bus.va_degree.to_numpy()
        lab = grid.lab.iloc[h]
        disp = (grid.gens.p_max_mw > grid.gens.p_min_mw).to_numpy()
        pg_true = np.array([lab[f"Pg_gen{k}"] for k in range(len(pg))])
        va_true = np.array([lab[f"Va_bus{b + 1}"] for b in range(len(va))])
        acc.append(dict(dc_ms=dc_ms, pg_mae_mw=np.abs(pg - pg_true)[disp].mean(),
                        va_mae_deg=np.abs(va - va_true).mean(),
                        cost_gap_abs_pct=abs(grid.cost(pg) - lab.cost) / lab.cost * 100))
        rows.append(grid.check(h, pg, vm_set))
    out = summarize_hours(rows)
    a = pd.DataFrame(acc).mean()
    out.update({f"dc_{k}": v for k, v in a.items()})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only the first N test hours")
    ap.add_argument("--no-dc", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grids = {}
    runs = []
    for case in ["case14", "case30"]:
        grid = grids.setdefault(case, Grid(case, a.limit))
        r = oracle(grid)
        r.update(group=f"AC-OPF labels through PF ({case}) - check", run="oracle", case=case)
        runs.append(r)
        print(f"[pipeline check] {case}: AC-OPF labels -> feasible {r['feasible_pct']:.1f} %, "
              f"|gap| {r['cost_gap_pf_abs_pct']:.1e} % (expect ~100 % and ~0)", flush=True)
    for label, (case, pattern) in DEFAULT_GROUPS.items():
        ckpts = sorted(RES.glob(f"{pattern}/best.pt"))
        if not ckpts:
            print(f"[{label}] no runs matching {pattern} - skipped"); continue
        grid = grids.setdefault(case, Grid(case, a.limit))
        for ck in ckpts:
            t0 = time.time()
            vm, pg = predict(ck, grid, dev)
            rows = [grid.check(h, pg[h], vm[h]) for h in range(len(grid.ts))]
            r = summarize_hours(rows)
            r.update(group=label, run=ck.parent.name, case=case)
            runs.append(r)
            print(f"[{label}] {ck.parent.name}: feasible {r['feasible_pct']:.1f} %, PF converged "
                  f"{r['pf_converged_pct']:.1f} %, |gap| after PF {r.get('cost_gap_pf_abs_pct', np.nan):.3f} % "
                  f"({time.time() - t0:.0f} s)", flush=True)
    if not a.no_dc:
        for case in ["case14", "case30"]:
            grid = grids.setdefault(case, Grid(case, a.limit))
            print(f"[DC-OPF baseline] {case}: solving {len(grid.ts)} test hours ...", flush=True)
            r = dc_baseline(grid)
            r.update(group=f"DC-OPF baseline ({case})", run="dcopf", case=case)
            runs.append(r)
    df = pd.DataFrame(runs)
    df.to_csv(RES / "pf_postprocess_runs.csv", index=False)
    cols = ["pf_converged_pct", "feasible_pct", "vm_viol_pct", "branch_viol_pct", "genq_viol_pct",
            "slack_p_viol_pct", "slack_q_viol_pct", "cost_gap_pf_abs_pct", "pf_ms"]
    cols += [c for c in df.columns if c.startswith("dc_")]
    g = df.groupby("group", sort=False)
    mean, std = g[cols].mean(), g[cols].std()
    table = mean.copy().astype(object)
    for c in cols:
        table[c] = [f"{m:.3g} ± {s:.2g}" if pd.notna(s) else (f"{m:.3g}" if pd.notna(m) else "")
                    for m, s in zip(mean[c], std[c])]
    table.insert(0, "n_models", g.size())
    table.to_csv(RES / "pf_postprocess_summary.csv")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print("\nAFTER POWER-FLOW POST-PROCESSING (test hours; % of hours)\n")
        print(table.T.to_string())
    print(f"\nSaved {RES / 'pf_postprocess_runs.csv'} and {RES / 'pf_postprocess_summary.csv'}")
