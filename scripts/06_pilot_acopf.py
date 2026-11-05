"""
06_pilot_acopf.py
Solves AC-OPF on a sample of hours for both networks to check that the
0.6-1.0x load band is feasible BEFORE generating the full dataset.

Sample: every 29th hour (covers every hour of day and every month, ~300 hours)
        + the 20 highest- and 20 lowest-load hours (stress test).

Run from the project folder with the venv active:
    python scripts\\06_pilot_acopf.py
Writes results\\pilot_acopf_case14.csv and results\\pilot_acopf_case30.csv
"""

import time

import numpy as np
import pandas as pd
import pandapower as pp

from networks import PROJECT_DIR, get_net, load_buses

RES_DIR = PROJECT_DIR / "results"
RES_DIR.mkdir(parents=True, exist_ok=True)


def run_case(case):
    sc = pd.read_parquet(PROJECT_DIR / "data" / "processed" / f"load_scenarios_{case}.parquet")
    total = sc.filter(like="P_bus").sum(axis=1)
    pos = set(range(0, len(sc), 29)) | set(np.argsort(total.to_numpy())[:20]) \
        | set(np.argsort(total.to_numpy())[-20:])
    pos = sorted(pos)

    net = get_net(case)
    buses = load_buses(net)
    rows = []
    print(f"\n{case}: solving {len(pos)} hours ...")
    for n, i in enumerate(pos, 1):
        s = sc.iloc[i]
        net.load["p_mw"] = [s[f"P_bus{b}"] for b in buses]
        net.load["q_mvar"] = [s[f"Q_bus{b}"] for b in buses]
        t0 = time.perf_counter()
        try:
            pp.runopp(net)
            ok = bool(net.OPF_converged)
        except Exception:
            ok = False
        r = dict(timestamp=sc.index[i], load_mw=total.iloc[i],
                 lambda_mapped=s["lambda_mapped_mean"], converged=ok,
                 time_s=time.perf_counter() - t0)
        if ok:
            r.update(cost=net.res_cost,
                     max_line_loading=net.res_line.loading_percent.max(),
                     max_trafo_loading=(net.res_trafo.loading_percent.max() if len(net.trafo) else np.nan),
                     vm_min=net.res_bus.vm_pu.min(), vm_max=net.res_bus.vm_pu.max(),
                     losses_mw=net.res_line.pl_mw.sum() + (net.res_trafo.pl_mw.sum() if len(net.trafo) else 0))
        rows.append(r)
        if n % 50 == 0:
            print(f"  {n}/{len(pos)} done")

    df = pd.DataFrame(rows)
    df.to_csv(RES_DIR / f"pilot_acopf_{case}.csv", index=False)

    ok = df[df.converged]
    print(f"  converged: {len(ok)}/{len(df)} ({100*len(ok)/len(df):.1f}%)")
    print(f"  solve time: median {df.time_s.median():.3f} s, max {df.time_s.max():.3f} s")
    if len(ok):
        print(f"  cost: {ok.cost.min():.1f} - {ok.cost.max():.1f}")
        print(f"  max line loading: median {ok.max_line_loading.median():.1f}%, "
              f"max {ok.max_line_loading.max():.1f}% | hours with a line >= 99%: "
              f"{(ok.max_line_loading >= 99).sum()}")
        print(f"  Vm: {ok.vm_min.min():.3f} - {ok.vm_max.max():.3f} pu | losses "
              f"{ok.losses_mw.min():.2f} - {ok.losses_mw.max():.2f} MW")
    df["band"] = pd.cut(df.lambda_mapped, [0.6, 0.7, 0.8, 0.9, 0.95, 1.0], include_lowest=True)
    print("  convergence by mapped load factor:")
    print(df.groupby("band", observed=True).converged.agg(["count", "mean"])
          .rename(columns={"mean": "conv_rate"}).round(3).to_string())
    bad = df[~df.converged]
    if len(bad):
        print("  failed hours (first 10):")
        print(bad[["timestamp", "load_mw", "lambda_mapped"]].head(10).to_string(index=False))


if __name__ == "__main__":
    for c in ["case14", "case30"]:
        run_case(c)
