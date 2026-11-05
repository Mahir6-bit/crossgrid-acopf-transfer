"""
04_verify_dataset_structure.py
Tests the hypothesis that the hourly files contain lossless DC power-flow / DC-OPF
results on MATPOWER case14 and case30:
    line_1..N      = from-side branch active-power flow (MW), MATPOWER branch order
    line_N+1..2N   = to-side flow (exact negative -> lossless DC)
    Cons_i         = net active consumption at bus i (load - generation, MW)

Run from the project folder with the venv active:
    python scripts\\04_verify_dataset_structure.py
Output is printed and saved to logs\\verify_structure_report.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "raw"
LOG_DIR = PROJECT_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- MATPOWER reference data ----------------
CASES = {
    "case14": dict(
        file="IEEE_14_Bus_Time_Series_Hourly.csv",
        n_bus=14,
        branches=[(1,2),(1,5),(2,3),(2,4),(2,5),(3,4),(4,5),(4,7),(4,9),(5,6),
                  (6,11),(6,12),(6,13),(7,8),(7,9),(9,10),(9,14),(10,11),(12,13),(13,14)],
        Pd=[0,21.7,94.2,47.8,7.6,11.2,0,0,29.5,9,3.5,6.1,13.5,14.9],
        gen_buses=[1,2,3,6,8],
        rateA=None,  # MATPOWER case14 has no line ratings
    ),
    "case30": dict(
        file="IEEE_30_Bus_Time_Series_Hourly.csv",
        n_bus=30,
        branches=[(1,2),(1,3),(2,4),(3,4),(2,5),(2,6),(4,6),(5,7),(6,7),(6,8),(6,9),(6,10),
                  (9,11),(9,10),(4,12),(12,13),(12,14),(12,15),(12,16),(14,15),(16,17),(15,18),
                  (18,19),(19,20),(10,20),(10,17),(10,21),(10,22),(21,22),(15,23),(22,24),(23,24),
                  (24,25),(25,26),(25,27),(28,27),(27,29),(27,30),(29,30),(8,28),(6,28)],
        Pd=[0,21.7,2.4,7.6,0,0,22.8,30,0,5.8,0,11.2,0,6.2,8.2,3.5,9,3.2,9.5,2.2,
            17.5,0,3.2,8.7,0,3.5,0,0,2.4,10.6],
        gen_buses=[1,2,13,22,23,27],
        rateA=[130,130,65,130,130,65,90,70,130,32,65,32,65,65,65,65,32,32,32,16,16,
               16,16,32,32,32,32,32,32,16,16,16,16,16,16,65,16,16,16,32,32],
    ),
}


class Tee:
    def __init__(self, *s): self.s = s
    def write(self, x): [t.write(x) for t in self.s]
    def flush(self): [t.flush() for t in self.s]


def section(t): print("\n" + "=" * 80 + "\n" + t + "\n" + "=" * 80)


def verify(name, cfg):
    section(f"{name}: {cfg['file']}")
    df = pd.read_csv(DATA_DIR / cfg["file"], index_col=0)
    nb, br = cfg["n_bus"], cfg["branches"]
    nl = len(br)

    # ---- 1. Timestamps ----
    ts = pd.to_datetime(df.index, errors="coerce")
    print(f"[1] Timestamps: {ts.min()} -> {ts.max()} | unparsable: {ts.isna().sum()} | "
          f"duplicates: {ts.duplicated().sum()} | sorted: {ts.is_monotonic_increasing}")
    steps = pd.Series(ts).diff().dropna().value_counts()
    print(f"    Time steps found: {dict(steps)}")

    line_cols = [c for c in df.columns if c.startswith("line_")]
    cons_cols = [c for c in df.columns if c.startswith("Cons_")]
    print(f"[2] line columns: {len(line_cols)} (expected {2*nl}) | "
          f"Cons columns: {len(cons_cols)} (expected {nb})")
    if len(line_cols) != 2 * nl or len(cons_cols) != nb:
        print("    !! Column counts do not match MATPOWER case -- stopping checks for this file.")
        return

    F = df[[f"line_{k+1}" for k in range(2 * nl)]].to_numpy(float)
    C = df[[f"Cons_{i+1}" for i in range(nb)]].to_numpy(float)
    f_from, f_to = F[:, :nl], F[:, nl:]

    # ---- 3. Lossless check ----
    anti = np.abs(f_from + f_to).max(axis=1)
    print(f"[3] max |from + to| over all rows: {anti.max():.2e} "
          f"(~0 => lossless DC flow, no line losses)")

    # ---- 4. System balance ----
    s = C.sum(axis=1)
    print(f"[4] sum(Cons) per hour: max |.| = {np.abs(s).max():.4f} MW, "
          f"hours with |sum| > 0.01 MW: {(np.abs(s) > 0.01).sum()}")

    # ---- 5. Nodal balance against MATPOWER branch order ----
    A = np.zeros((nl, nb))
    for k, (i, j) in enumerate(br):
        A[k, i-1], A[k, j-1] = -1, 1
    inflow = f_from @ A            # net power flowing into each bus
    resid = np.abs(C - inflow).max(axis=1)
    bad = np.where(resid > 1e-3)[0]
    print(f"[5] Nodal balance residual: median {np.median(resid):.2e}, max {resid.max():.4f} MW")
    print(f"    Hours violating nodal balance (>1e-3 MW): {len(bad)}")
    if len(bad):
        print(f"    First few bad hours: {list(df.index[bad[:10]])}")

    # ---- 6. Bus roles ----
    Pd = np.array(cfg["Pd"], float)
    gen = np.array(cfg["gen_buses"]) - 1
    pure_load = [i for i in range(nb) if Pd[i] > 0 and i not in gen]
    zero_bus = [i for i in range(nb) if Pd[i] == 0 and i not in gen]
    print(f"[6] Zero-load, non-generator buses {[i+1 for i in zero_bus]}: "
          f"max |Cons| = {np.abs(C[:, zero_bus]).max():.2e} (expect 0)")
    print(f"    Negative Cons at pure-load buses (unexpected): {(C[:, pure_load] < -1e-6).sum()} values")
    print("    Generator-bus net consumption (negative = net generation):")
    print(pd.DataFrame(C[:, gen], columns=[f"bus{g+1}" for g in gen])
          .describe().loc[["mean", "min", "max"]].round(2).to_string())

    # ---- 7. Load scaling relative to MATPOWER nominal Pd ----
    scale = C[:, pure_load] / Pd[pure_load]
    print(f"[7] Load scale factor Cons/Pd_nominal at pure-load buses:")
    print(f"    overall min {scale.min():.3f}, mean {scale.mean():.3f}, max {scale.max():.3f}")
    spread = scale.max(axis=1) - scale.min(axis=1)
    print(f"    spread across buses within an hour: median {np.median(spread):.3f}, "
          f"max {spread.max():.3f}  (0 => one system-wide profile)")
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(scale.T)
    print(f"    mean pairwise correlation between bus profiles: "
          f"{corr[np.triu_indices_from(corr, 1)].mean():.3f}")

    # ---- 8. Binding line limits (sign of DC-OPF) ----
    absf = np.abs(f_from)
    print("[8] Lines that sit exactly at their maximum |flow| in many hours "
          "(suggests a binding limit / OPF):")
    found = False
    for k in range(nl):
        m = absf[:, k].max()
        n_at = int((np.abs(absf[:, k] - m) < 1e-4).sum())
        rate = cfg["rateA"][k] if cfg["rateA"] else None
        if m > 1e-6 and 5 < n_at < len(absf):
            found = True
            print(f"    line_{k+1} ({br[k][0]}-{br[k][1]}): max|f| = {m:.4f} MW, "
                  f"hours at that value: {n_at}" + (f", MATPOWER rateA = {rate}" if rate else ""))
    if not found:
        print("    none")
    if cfg["rateA"]:
        r = np.array(cfg["rateA"], float)
        at_lim = (absf >= r - 1e-4)
        over = (absf > r + 1e-3)
        print(f"    Hours with at least one line at its MATPOWER rateA: {at_lim.any(axis=1).sum()} "
              f"| line-hours ABOVE rateA: {over.sum()}")

    # ---- 9. Total load / generation envelope ----
    load_total = C.clip(min=0).sum(axis=1)
    print(f"[9] Total positive consumption per hour: min {load_total.min():.1f}, "
          f"mean {load_total.mean():.1f}, max {load_total.max():.1f} MW "
          f"(MATPOWER nominal total Pd = {Pd.sum():.1f} MW)")


def main():
    with open(LOG_DIR / "verify_structure_report.txt", "w", encoding="utf-8") as f:
        sys.stdout = Tee(sys.__stdout__, f)
        for name, cfg in CASES.items():
            try:
                verify(name, cfg)
            except Exception as e:
                print(f"!! ERROR in {name}: {type(e).__name__}: {e}")
        sys.stdout = sys.__stdout__
    print(f"\nReport saved to {LOG_DIR / 'verify_structure_report.txt'}")


if __name__ == "__main__":
    main()
