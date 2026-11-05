"""
07_build_load_scenarios.py
Turns the hourly dataset into per-bus AC-OPF load scenarios.

  1. lambda_i(t) = Cons_i(t) / Pd_nominal_i   at pure-load buses (exact loads in the data)
  2. At generator buses that also carry load (Cons mixes load and generation), lambda_i(t)
     = mean of the pure-load-bus factors in that hour.
  3. Map to the solvable band, same rule for both grids:   lambda' = 0.6 + 0.2 * lambda
     (dataset range 0-2x  ->  0.6-1.0x nominal; daily/seasonal shape and per-bus diversity kept)
  4. P_d = lambda' * P_nominal(network),  Q_d = lambda' * Q_nominal(network)  (constant power factor)

The dataset's line flows and generator values are NOT used.

Run from the project folder with the venv active:
    python scripts\\07_build_load_scenarios.py
Writes data\\processed\\load_scenarios_case14.parquet and ..._case30.parquet
"""

import numpy as np
import pandas as pd

from networks import DATASET, PROJECT_DIR, get_net, load_buses

LAMBDA_MIN, LAMBDA_SPAN = 0.6, 0.2       # lambda' = 0.6 + 0.2 * lambda
DATA_DIR = PROJECT_DIR / "data" / "raw"
OUT_DIR = PROJECT_DIR / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build(case):
    meta = DATASET[case]
    df = pd.read_csv(DATA_DIR / meta["file"], index_col=0)
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"

    Pd = np.array(meta["Pd"], float)
    gen = set(meta["gen_buses"])
    pure = [b for b in range(1, meta["n_bus"] + 1) if Pd[b-1] > 0 and b not in gen]

    lam = {b: df[f"Cons_{b}"].to_numpy() / Pd[b-1] for b in pure}
    hourly_mean = np.mean(np.column_stack(list(lam.values())), axis=1)

    net = get_net(case)
    out = {}
    imputed = []
    for row, b in zip(net.load.itertuples(), load_buses(net)):
        if b in lam:
            raw = lam[b]
        else:                                   # generator bus with load, or no profile in data
            raw = hourly_mean
            imputed.append(b)
        mapped = LAMBDA_MIN + LAMBDA_SPAN * raw
        out[f"P_bus{b}"] = mapped * row.p_mw
        out[f"Q_bus{b}"] = mapped * row.q_mvar
    res = pd.DataFrame(out, index=df.index)
    res["lambda_raw_mean"] = hourly_mean
    res["lambda_mapped_mean"] = LAMBDA_MIN + LAMBDA_SPAN * hourly_mean

    P = res.filter(like="P_bus").sum(axis=1)
    Q = res.filter(like="Q_bus").sum(axis=1)
    print(f"\n{case}: {len(res)} hours, {len(net.load)} loads at buses {load_buses(net)}")
    print(f"  buses using the hourly-mean factor (generator buses with load): {imputed}")
    print(f"  nominal network load: P = {net.load.p_mw.sum():.1f} MW, Q = {net.load.q_mvar.sum():.1f} Mvar")
    print(f"  scenario total P (MW): min {P.min():.1f} | median {P.median():.1f} | max {P.max():.1f}")
    print(f"  scenario total Q (Mvar): min {Q.min():.1f} | median {Q.median():.1f} | max {Q.max():.1f}")
    lm = res["lambda_mapped_mean"]
    print(f"  mapped mean factor: min {lm.min():.3f} | median {lm.median():.3f} | max {lm.max():.3f}")
    path = OUT_DIR / f"load_scenarios_{case}.parquet"
    res.to_parquet(path)
    print(f"  saved {path}")


if __name__ == "__main__":
    for c in ["case14", "case30"]:
        build(c)
