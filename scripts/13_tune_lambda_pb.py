"""
13_tune_lambda_pb.py
Chooses the physics-loss weight lambda_pb on the VALIDATION set (research plan, Step 4).

Tuning uses only the SOURCE grid (IEEE 14 train + LPT augmentation, IEEE 14 Sep-Oct validation),
so the IEEE 30 target is never looked at. Each candidate is trained with a reduced budget
(--steps, default 10k) and evaluated on IEEE 14 validation in physical units.

Selection rule (fixed before looking at results, state it in the paper):
  among candidates whose validation Pg MAE is within 10 % of the best candidate,
  pick the one with the lowest validation mean |dP| power-balance mismatch.
lambda_br (thermal + angle hinge) stays 1.0: these limits are almost never active on IEEE 14.

Run from the project folder:
    python scripts\\13_tune_lambda_pb.py
Options: --values 0.1 1 10  --steps 10000  --parallel 3
Writes results\\lambda_tuning.csv
"""

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import torch

from evaluate import evaluate, load_checkpoint

PROJECT = Path(__file__).resolve().parents[1]
TRAIN = PROJECT / "scripts" / "train.py"


def train(job):
    name, lam, steps = job
    d = PROJECT / "results" / name
    if (d / "best.pt").exists() and (d / "history.csv").exists():
        return name, "skipped (done)"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "train.log", "w", encoding="utf-8") as log:
        r = subprocess.run([sys.executable, str(TRAIN), "--name", name, "--train-case", "case14",
                            "--aug", "LPT", "--seed", "0", "--lambda-pb", str(lam), "--steps", str(steps),
                            "--eval-cases", "case14"], stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT)
    return name, "ok" if r.returncode == 0 else f"FAILED (see {d / 'train.log'})"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--values", nargs="+", type=float, default=[0.1, 1.0, 10.0])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--parallel", type=int, default=3)
    a = ap.parse_args()

    jobs = [(f"tune_pb{v:g}", v, a.steps) for v in a.values]
    print(f"Training {len(jobs)} candidates ({a.steps} steps each, {a.parallel} at a time) ...")
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for name, status in ex.map(train, jobs):
            print(f"  {name:14s} {status}", flush=True)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name, lam, _ in jobs:
        model, ck = load_checkpoint(PROJECT / "results" / name / "best.pt", dev)
        m = evaluate(model, "case14", "val", dev)
        rows.append(dict(lambda_pb=lam, best_step=ck["step"], val_vm_mae_pu=m["vm_mae_pu"],
                         val_va_mae_deg=m["va_mae_deg"], val_pg_mae_mw=m["pg_mae_mw"],
                         val_pb_p_mean_mw=m["pb_p_mean_mw"], val_pb_p_max_mw=m["pb_p_max_mw"],
                         val_pb_q_mean_mvar=m["pb_q_mean_mvar"], val_cost_gap_abs_pct=m["cost_gap_abs_pct"]))
    df = pd.DataFrame(rows).set_index("lambda_pb")
    ok = df[df.val_pg_mae_mw <= 1.10 * df.val_pg_mae_mw.min()]
    choice = ok.val_pb_p_mean_mw.idxmin()
    df["selected"] = df.index == choice
    df.to_csv(PROJECT / "results" / "lambda_tuning.csv")
    with pd.option_context("display.width", 200, "display.float_format", "{:.4g}".format):
        print("\nIEEE 14 VALIDATION (Sep-Oct) metrics per lambda_pb:\n")
        print(df.to_string())
    print(f"\nSelected lambda_pb = {choice:g}  (Pg MAE within 10 % of best, lowest mean |dP|)")
    print(f"Saved {PROJECT / 'results' / 'lambda_tuning.csv'}")
