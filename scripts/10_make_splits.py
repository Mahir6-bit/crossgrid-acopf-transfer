"""
10_make_splits.py   (revision version - drop-in replacement)
Defines ONE train/validation/test split of the 8760 hours, shared by all grids,
plus the nested fine-tuning subsets of the training set of each target grid.

Why shared: case14 and case30 are driven by the same demand profile (corr 0.994), and the case57
scenarios are built from the case30 profiles. If target test hours were source training hours,
zero-shot results would be optimistic.

Scheme (default "chronological"):
    Jan-Aug -> train (8 months), Sep-Oct -> validation (2 months), Nov-Dec -> test (2 months)
Alternative "--scheme weekly": 7-day blocks, block index mod 10: 0-6 train, 7 val, 8-9 test.

Fine-tuning subsets: random, NESTED (1% in 5% in 10% in 25% in 100%) samples of the train hours,
one set per seed. Matching validation subsets are drawn the same way (min 24 hours).
The subsets of seed s depend only on s, so seeds 0-4 are IDENTICAL to the original run when more
seeds are added (checked below against an existing file, if present).

Run from the project folder with the venv active:
    python scripts\\10_make_splits.py                                   # case30, seeds 0-4 (original)
    python scripts\\10_make_splits.py --n-seeds 10                      # seeds 0-9 for the revision
    python scripts\\10_make_splits.py --n-seeds 10 --cases case14 case30 case57
Writes data\\processed\\splits.csv and data\\processed\\finetune_subsets_<case>.json
"""

import argparse
import json

import numpy as np
import pandas as pd

from networks import PROJECT_DIR

PROC = PROJECT_DIR / "data" / "processed"
FRACS = [0.01, 0.05, 0.10, 0.25, 1.00]


def assign(index, scheme):
    if scheme == "chronological":
        m = index.month
        return np.where(m <= 8, "train", np.where(m <= 10, "val", "test"))
    block = ((index - index[0]).days // 7) % 10
    return np.select([block <= 6, block == 7], ["train", "val"], "test")


def nested_subsets(hours, rng, min_n=1):
    perm = rng.permutation(hours)
    return {str(f): sorted(perm[:max(min_n, int(round(f * len(hours))))].tolist()) for f in FRACS}


def main(scheme, cases, n_seeds):
    check = list(dict.fromkeys(["case14"] + list(cases)))          # always include the source grid
    lab = {c: pd.read_parquet(PROC / f"acopf_labels_{c}.parquet") for c in check
           if (PROC / f"acopf_labels_{c}.parquet").exists()}
    missing = [c for c in cases if c not in lab]
    assert not missing, f"labels missing for {missing} - run generate_acopf_labels.py first"
    ref = next(iter(lab.values())).index
    for c in lab:
        assert lab[c].index.equals(ref), f"{c} label timestamps differ"
        bad = int((~lab[c].converged).sum())
        assert bad == 0, f"{c} has {bad} non-converged hours - fix the scenarios before splitting"
    print(f"Checks passed: {len(ref)} hours, identical timestamps in the labels of {list(lab)}")

    split = pd.Series(assign(ref, scheme), index=ref, name="split")
    old = PROC / "splits.csv"
    if old.exists():
        prev = pd.read_csv(old, index_col=0, parse_dates=True).split
        if not prev.reindex(ref).equals(split):
            raise SystemExit("splits.csv exists and differs from the requested scheme - not overwritten.")
    split.to_frame().to_csv(old)
    print(f"Scheme {scheme}: " + ", ".join(f"{s} {int((split == s).sum())} h" for s in ["train", "val", "test"]))

    fmt = lambda ts: [t.strftime("%Y-%m-%d %H:%M:%S") for t in ts]
    train_h, val_h = ref[split == "train"], ref[split == "val"]
    for case in cases:
        subsets = {"scheme": scheme, "fractions": FRACS, "seeds": {}}
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            tr = nested_subsets(np.arange(len(train_h)), rng)
            va = nested_subsets(np.arange(len(val_h)), rng, min_n=24)
            subsets["seeds"][str(seed)] = {"train": {f: fmt(train_h[i]) for f, i in tr.items()},
                                           "val": {f: fmt(val_h[i]) for f, i in va.items()}}
        for d in subsets["seeds"].values():                         # nesting check
            for a, b in zip(map(str, FRACS[:-1]), map(str, FRACS[1:])):
                assert set(d["train"][a]) <= set(d["train"][b]), "subsets not nested"
        path = PROC / f"finetune_subsets_{case}.json"
        if path.exists():                                           # seeds already used must not change
            prev = json.load(open(path))["seeds"]
            for s, d in prev.items():
                if s in subsets["seeds"] and subsets["seeds"][s] != d:
                    raise SystemExit(f"{path.name}: seed {s} would change - not overwritten.")
        json.dump(subsets, open(path, "w"))
        s0 = subsets["seeds"]["0"]
        print(f"{path.name}: seeds 0-{n_seeds - 1}; sizes (seed 0) " +
              ", ".join(f"{float(f) * 100:g}%: {len(s0['train'][f])}/{len(s0['val'][f])} h" for f in map(str, FRACS)))
    print("Done (train/val hours per fraction shown as train/val).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["weekly", "chronological"], default="chronological")
    ap.add_argument("--cases", nargs="+", default=["case30"],
                    help="grids that need fine-tuning subsets (case14 for the reverse transfer, case57 for the extension)")
    ap.add_argument("--n-seeds", type=int, default=5)
    a = ap.parse_args()
    main(a.scheme, a.cases, a.n_seeds)
