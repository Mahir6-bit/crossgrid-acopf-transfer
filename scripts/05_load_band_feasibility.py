"""
05_load_band_feasibility.py
Uniformly scales all loads (P and Q, constant power factor) of pandapower case14
and case30 from 0.10x to 2.00x nominal and solves AC-OPF at each level.
Shows at which load level each network stops being AC-OPF feasible, which tells
us how to map the dataset's 0-2x load factors onto solvable operating points.

Run from the project folder with the venv active:
    python scripts\\05_load_band_feasibility.py
Saves results\\opf_feasibility_sweep.csv
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn

warnings.filterwarnings("ignore")
import logging
logging.getLogger("pandapower").setLevel(logging.ERROR)

PROJECT_DIR = Path(__file__).resolve().parents[1]
RES_DIR = PROJECT_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)

SCALES = np.round(np.arange(0.10, 2.0001, 0.05), 2)
rows = []

for name, builder in [("case14", pn.case14), ("case30", pn.case30)]:
    base = builder()
    p0, q0 = base.load.p_mw.copy(), base.load.q_mvar.copy()
    pmax = base.gen.max_p_mw.sum() + base.ext_grid.max_p_mw.sum()
    print(f"\n{name}: nominal load {p0.sum():.1f} MW, total Pmax {pmax:.1f} MW")
    print(f"{'scale':>6} {'load_MW':>8} {'conv':>5} {'cost':>10} {'loss_MW':>8} "
          f"{'maxLine%':>8} {'Vmin':>6} {'Vmax':>6} {'t_s':>6}")
    for s in SCALES:
        net = builder()
        net.load.p_mw = p0 * s
        net.load.q_mvar = q0 * s
        t0 = time.perf_counter()
        try:
            pp.runopp(net)
            ok = bool(net.OPF_converged)
        except Exception:
            ok = False
        dt = time.perf_counter() - t0
        r = dict(case=name, scale=s, load_mw=p0.sum() * s, converged=ok, time_s=dt)
        if ok:
            loss = net.res_line.pl_mw.sum() + (net.res_trafo.pl_mw.sum() if len(net.trafo) else 0)
            r.update(cost=net.res_cost, loss_mw=loss,
                     max_line_loading=net.res_line.loading_percent.max(),
                     vm_min=net.res_bus.vm_pu.min(), vm_max=net.res_bus.vm_pu.max())
            print(f"{s:6.2f} {r['load_mw']:8.1f} {'yes':>5} {r['cost']:10.1f} {loss:8.2f} "
                  f"{r['max_line_loading']:8.1f} {r['vm_min']:6.3f} {r['vm_max']:6.3f} {dt:6.2f}")
        else:
            print(f"{s:6.2f} {r['load_mw']:8.1f} {'NO':>5}")
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(RES_DIR / "opf_feasibility_sweep.csv", index=False)
print("\nHighest converged uniform scale:")
print(df[df.converged].groupby("case").scale.max().to_string())
print("Lowest converged uniform scale:")
print(df[df.converged].groupby("case").scale.min().to_string())
print(f"\nSaved {RES_DIR / 'opf_feasibility_sweep.csv'}")
