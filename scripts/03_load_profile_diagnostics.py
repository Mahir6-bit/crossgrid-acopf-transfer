"""
03_load_profile_diagnostics.py
Characterises the hourly LOAD profiles (the only part of the dataset we will reuse)
and explains the 30-bus rows that fail nodal balance.

Run from the project folder with the venv active:
    python scripts\\03_load_profile_diagnostics.py
Output is printed and saved to logs\\load_profile_diagnostics.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CASES = {
    "case14": dict(
        file="IEEE_14_Bus_Time_Series_Hourly.csv", n_bus=14,
        branches=[(1,2),(1,5),(2,3),(2,4),(2,5),(3,4),(4,5),(4,7),(4,9),(5,6),
                  (6,11),(6,12),(6,13),(7,8),(7,9),(9,10),(9,14),(10,11),(12,13),(13,14)],
        Pd=[0,21.7,94.2,47.8,7.6,11.2,0,0,29.5,9,3.5,6.1,13.5,14.9],
        gen_buses=[1,2,3,6,8], Pmax_total=332.4+140+100+100+100,
    ),
    "case30": dict(
        file="IEEE_30_Bus_Time_Series_Hourly.csv", n_bus=30,
        branches=[(1,2),(1,3),(2,4),(3,4),(2,5),(2,6),(4,6),(5,7),(6,7),(6,8),(6,9),(6,10),
                  (9,11),(9,10),(4,12),(12,13),(12,14),(12,15),(12,16),(14,15),(16,17),(15,18),
                  (18,19),(19,20),(10,20),(10,17),(10,21),(10,22),(21,22),(15,23),(22,24),(23,24),
                  (24,25),(25,26),(25,27),(28,27),(27,29),(27,30),(29,30),(8,28),(6,28)],
        Pd=[0,21.7,2.4,7.6,0,0,22.8,30,0,5.8,0,11.2,0,6.2,8.2,3.5,9,3.2,9.5,2.2,
            17.5,0,3.2,8.7,0,3.5,0,0,2.4,10.6],
        gen_buses=[1,2,13,22,23,27], Pmax_total=80+80+40+50+30+55,
    ),
}


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, x): [t.write(x) for t in self.s]
    def flush(self): [t.flush() for t in self.s]


def load(cfg):
    df = pd.read_csv(DATA_DIR / cfg["file"], index_col=0)
    df.index = pd.to_datetime(df.index)
    nb, br = cfg["n_bus"], cfg["branches"]
    nl = len(br)
    F = df[[f"line_{k+1}" for k in range(nl)]].to_numpy(float)
    C = df[[f"Cons_{i+1}" for i in range(nb)]].to_numpy(float)
    A = np.zeros((nl, nb))
    for k, (i, j) in enumerate(br):
        A[k, i-1], A[k, j-1] = -1, 1
    resid = np.abs(C - F @ A).max(axis=1)
    Pd = np.array(cfg["Pd"], float)
    gen = set(np.array(cfg["gen_buses"]) - 1)
    pure = [i for i in range(nb) if Pd[i] > 0 and i not in gen]
    scale = C[:, pure] / Pd[pure]                      # per-bus load factor
    mean_scale = scale.mean(axis=1)
    # estimated total load: exact at pure-load buses, mean factor at gen buses with load
    gen_load = [i for i in gen if Pd[i] > 0]
    total = C[:, pure].sum(axis=1) + mean_scale * Pd[gen_load].sum()
    return df, F, C, resid, scale, mean_scale, total


def pct(x, q=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    return "  ".join(f"p{p}={v:.3f}" for p, v in zip(q, np.percentile(x, q)))


def main():
    out = {}
    with open(LOG_DIR / "load_profile_diagnostics.txt", "w", encoding="utf-8") as f:
        sys.stdout = Tee(sys.__stdout__, f)
        for name, cfg in CASES.items():
            print("\n" + "=" * 80 + f"\n{name}\n" + "=" * 80)
            df, F, C, resid, scale, ms, total = load(cfg)
            out[name] = (df.index, ms)
            n = len(df)

            print("[A] Per-bus load factor (Cons / nominal Pd), pure-load buses:")
            print("    " + pct(scale.ravel()))
            print(f"    bus-hours exactly 0: {(scale == 0).mean()*100:.2f}%  |  "
                  f"exactly 2.0: {(np.abs(scale-2) < 1e-9).mean()*100:.2f}%  (clipping?)")
            print("[B] Hourly mean load factor across buses:")
            print("    " + pct(ms))
            print(f"    hours with mean factor < 0.2: {(ms < 0.2).sum()}  | > 1.5: {(ms > 1.5).sum()}")
            print(f"[C] Estimated total load (MW): " + pct(total, (0, 50, 95, 100)))
            print(f"    total generator Pmax = {cfg['Pmax_total']:.1f} MW -> hours with load "
                  f"> 100% Pmax: {(total > cfg['Pmax_total']).sum()}, "
                  f"> 90%: {(total > 0.9*cfg['Pmax_total']).sum()}")

            m = pd.Series(ms, index=df.index)
            print("[D] Mean load factor by month:")
            print("    " + "  ".join(f"{k}:{v:.2f}" for k, v in m.groupby(m.index.month).mean().items()))
            print("    by hour of day:")
            print("    " + "  ".join(f"{k}:{v:.2f}" for k, v in m.groupby(m.index.hour).mean().items()))

            bad = resid > 1e-3
            print(f"[E] Hours failing nodal balance: {bad.sum()} of {n} ({bad.mean()*100:.1f}%)")
            if bad.any():
                same_prev = np.r_[False, np.all(np.abs(F[1:] - F[:-1]) < 1e-9, axis=1)]
                print(f"    bad hours whose line flows are identical to the previous hour: "
                      f"{(bad & same_prev).sum()}")
                dec = pd.qcut(total, 10, labels=False, duplicates="drop")
                tab = pd.DataFrame({"decile": dec, "bad": bad, "load": total}) \
                        .groupby("decile").agg(load_min=("load", "min"), load_max=("load", "max"),
                                               bad_pct=("bad", lambda x: 100 * x.mean()))
                print("    share of bad hours by total-load decile:")
                print(tab.round(1).to_string())
                bh = pd.Series(bad, index=df.index)
                print("    bad share by month (%): " +
                      "  ".join(f"{k}:{v*100:.0f}" for k, v in bh.groupby(bh.index.month).mean().items()))

        if "case14" in out and "case30" in out:
            a, b = out["case14"][1], out["case30"][1]
            print("\n" + "=" * 80)
            print(f"[F] Correlation of hourly mean load factor, case14 vs case30: "
                  f"{np.corrcoef(a, b)[0,1]:.3f}  (near 1 => same underlying demand profile)")
        sys.stdout = sys.__stdout__
    print(f"\nReport saved to {LOG_DIR / 'load_profile_diagnostics.txt'}")


if __name__ == "__main__":
    main()
