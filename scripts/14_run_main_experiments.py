"""
14_run_main_experiments.py
Runs the experiment grid with train.py, skipping runs that already finished
(results\\<name>\\metrics.json exists) - safe to stop and restart.

Stages
  source : E1 (+E2 zero-shot) - IEEE 14 source models, one per seed      -> src14_s{seed}
  target : E3/E4 - IEEE 30 fine-tune vs from scratch                      -> ft30_/scratch30_f{frac}_s{seed}

Protocol (research plan, Step 5-6)
  * Source models are trained on the original IEEE 14 train hours + augmentation pool --aug
    (default LPT = load + parameter + N-1 topology augmentation; main model).
  * 5 seeds for every experiment, including the source model.
  * Fine-tune seed s starts from source model src14_s{s} (so the spread includes source variability).
  * Fine-tune and from scratch use IDENTICAL hyper-parameters (lr 3e-4, physics warm-up 2000 steps);
    the only difference is the initial weights.
  * Training data = nested subset of the case30 TRAIN split (Jan-Aug); validation = matching small
    subset of Sep-Oct (--small-val); test = full Nov-Dec of both grids.

Usage (from the project folder):
  python scripts\\14_run_main_experiments.py --stage source            (5 source models)
  python scripts\\14_run_main_experiments.py --stage target            (5 fractions x 5 seeds x 2 methods)
  options: --fractions 0.05 0.1 --seeds 0 1 --methods ft scratch --parallel 3 --lr 3e-4
           variants (ablations): --tag _nophys --extra "--lambda-pb 0"
           augmentation ablation (E5): --stage source --aug none --tag _augnone  (also L, LP)
                                       then --stage target --methods ft --tag _augnone
"""

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TRAIN = PROJECT / "scripts" / "train.py"
DEFAULT_FRACS = ["0.01", "0.05", "0.1", "0.25", "1.0"]


def source_jobs(seeds, lr, tag, extra, aug):
    return [(f"src14_s{s}{tag}", ["--train-case", "case14", "--seed", str(s), "--lr", str(lr),
                                  "--aug", aug] + extra) for s in seeds]


def target_jobs(fracs, seeds, methods, lr, tag, extra, source_tag=None):
    out = []
    for f in fracs:
        for s in seeds:
            base = ["--train-case", "case30", "--subset", f, "--seed", str(s), "--small-val",
                    "--lr", str(lr)] + extra
            if "ft" in methods:
                stag = tag if source_tag is None else source_tag
                src = PROJECT / "results" / f"src14_s{s}{stag}" / "best.pt"
                out.append((f"ft30_f{f}_s{s}{tag}", base + ["--init", str(src)]))
            if "scratch" in methods:
                out.append((f"scratch30_f{f}_s{s}{tag}", base))
    return out


def run(job):
    name, args = job
    d = PROJECT / "results" / name
    if (d / "metrics.json").exists():
        return name, "skipped (done)", 0.0
    if "--init" in args and not Path(args[args.index("--init") + 1]).exists():
        return name, "SKIPPED: source model missing", 0.0
    d.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(d / "train.log", "w", encoding="utf-8") as log:
        r = subprocess.run([sys.executable, str(TRAIN), "--name", name] + args,
                           stdout=log, stderr=subprocess.STDOUT, cwd=PROJECT)
    status = "ok" if r.returncode == 0 else f"FAILED (code {r.returncode}, see {d / 'train.log'})"
    return name, status, (time.time() - t0) / 60


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["source", "target"], required=True)
    ap.add_argument("--fractions", nargs="+", default=DEFAULT_FRACS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=["ft", "scratch"], choices=["ft", "scratch"])
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--tag", default="", help="suffix for run names of a variant, e.g. _nophys")
    ap.add_argument("--aug", choices=["none", "L", "LP", "LPT"], default="LPT",
                    help="source-model augmentation pool (main model: LPT)")
    ap.add_argument("--source-tag", default=None,
                    help="tag of the source models to fine-tune from (default: same as --tag)")
    ap.add_argument("--extra", default="", help='extra train.py options, e.g. "--lambda-pb 0"')
    a = ap.parse_args()

    extra = a.extra.split()
    todo = (source_jobs(a.seeds, a.lr, a.tag, extra, a.aug) if a.stage == "source"
            else target_jobs(a.fractions, a.seeds, a.methods, a.lr, a.tag, extra, a.source_tag))
    print(f"{len(todo)} runs, {a.parallel} at a time. Logs: results\\<run>\\train.log")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.parallel) as ex:
        for k, (name, status, mins) in enumerate(ex.map(run, todo), 1):
            print(f"[{k:3d}/{len(todo)}] {name:24s} {status:16s} {mins:6.1f} min | "
                  f"total {(time.time() - t0) / 60:.1f} min", flush=True)
    print("Done. Next: python scripts\\25_summarize_results.py")
