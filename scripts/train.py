"""
train.py
Trains the heterogeneous GNN on one grid, optionally starting from a checkpoint
(transfer / fine-tuning), then evaluates on the test split of the chosen grids.

Loss = supervised MSE (Vm, Va, Pg, Qg; each scaled by its std on the training set)
     + lambda_pb * mean(dP^2 + dQ^2)               power-balance (physics) term, ramped in
     + lambda_br * [mean(relu(|I| - I_max)^2) + mean(relu(|theta_i - theta_j| - 30 deg)^2)]
                                                    branch hinge: thermal (current) and angle-difference limits

Training is step-based (same budget for 62 or 6240 training graphs); validation every
--eval-every steps, best model kept, early stopping after --patience evaluations without improvement.

Examples (run from the project folder):
  Source model on IEEE 14 with full augmentation (also reports zero-shot on IEEE 30):
    python scripts\\train.py --train-case case14 --aug LPT --seed 0 --name src14_s0
  (--aug none | L | LP | LPT selects the augmentation pool; none = original hours only)
  Fine-tune on 5% of IEEE 30, seed 0:
    python scripts\\train.py --train-case case30 --subset 0.05 --seed 0 --init results\\src14_s0\\best.pt --name ft30_f0.05_s0
  Same subset from scratch:
    python scripts\\train.py --train-case case30 --subset 0.05 --seed 0 --name scratch30_f0.05_s0
Outputs: results\\<name>\\best.pt, history.csv, metrics.json
"""

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch

from evaluate import angle_terms, evaluate, physics_terms, print_metrics
from model import BUS_IN, HeteroOPFNet, HomoGNN, MLPBaseline, build_model, feature_stats
from networks import PROJECT_DIR
from opf_dataset import GraphBatcher, OPFGraphDataset, n1_holdout_branches

PROC = PROJECT_DIR / "data" / "processed"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def norm_stats(dev):
    """Feature mean/std from the IEEE 14 TRAIN split only (cached). Used for every model,
    including IEEE 30 from-scratch runs, so all models see identically scaled inputs."""
    path = PROC / f"norm_stats_case14_train_bus{BUS_IN}.pt"
    if not path.exists():
        gb = GraphBatcher(OPFGraphDataset("case14", split="train"), dev)
        torch.save({k: v.cpu() for k, v in feature_stats(gb.batches(512)).items()}, path)
        print(f"Computed normalisation statistics -> {path.name}")
    return {k: v.to(dev) for k, v in torch.load(path, weights_only=False).items()}


def target_scales(ds):
    t, i = ds.t, ds.idx
    gx = t["gen_x"][0] if t["gen_x"].dim() == 3 else t["gen_x"]
    disp = gx[:, 1] > gx[:, 0]
    s = dict(vm=t["vm"][i].std(), va=t["va"][i].std(),
             pg=t["pg"][i][:, disp].std(), qg=t["qg"][i].std())
    return {k: float(v.clamp_min(1e-4)) for k, v in s.items()}


def losses(batch, pred, sc, ang_max):
    by, gy = batch["bus"].y, batch["gen"].y
    sup = (((pred["vm"] - by[:, 0]) / sc["vm"]) ** 2).mean() \
        + (((pred["va"] - by[:, 1]) / sc["va"]) ** 2).mean() \
        + (((pred["pg"] - gy[:, 0]) / sc["pg"]) ** 2).mean() \
        + (((pred["qg"] - gy[:, 1]) / sc["qg"]) ** 2).mean()
    dp, dq, _, over = physics_terms(batch, pred)
    ang_over = torch.relu(angle_terms(batch, pred).abs() - ang_max)
    return sup, (dp ** 2 + dq ** 2).mean(), (over ** 2).mean() + (ang_over ** 2).mean()


def datasets(a):
    if a.subset is None:
        pools = None if a.aug == "none" else [a.aug]
        if pools:
            assert a.train_case == "case14", "--aug is only for the IEEE 14 source grid"
        return (OPFGraphDataset(a.train_case, split="train", aug_pools=pools, seed=a.seed),
                OPFGraphDataset(a.train_case, split="val"))
    sub = json.load(open(PROC / f"finetune_subsets_{a.train_case}.json"))["seeds"][str(a.seed)]
    key = str(a.subset)
    val_key = key if a.small_val else "1.0"
    n1 = lambda part: f"graph_tensors_{a.train_case}_n1{part}.pt" if a.n1 else None
    excl = n1_holdout_branches(a.n1_holdout, a.n1_holdout_seed) if a.n1 else []
    if excl:
        print(f"Unseen-outage experiment: N-1 samples of branches {excl} are held out of training/validation")
    return (OPFGraphDataset(a.train_case, timestamps=sub["train"][key], n1_file=n1("train"), n1_exclude=excl),
            OPFGraphDataset(a.train_case, timestamps=sub["val"][val_key], n1_file=n1("val"), n1_exclude=excl))


@torch.no_grad()
def val_loss(model, batcher, sc, a, dev):
    model.eval()
    tot, n = 0.0, 0
    for b in batcher.batches(256):
        sup, pb, br = losses(b, model(b), sc, a.ang_max)
        tot += (sup + a.lambda_pb * pb + a.lambda_br * br).item() * b.num_graphs
        n += b.num_graphs
    return tot / n


def main(a):
    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = PROJECT_DIR / "results" / a.name
    out.mkdir(parents=True, exist_ok=True)

    tr, va = datasets(a)
    # loss scales from the ORIGINAL training hours, so all augmentation arms use the same loss
    sc = target_scales(OPFGraphDataset(a.train_case, split="train") if (tr.n_aug and a.aug != "none") else tr)
    tr_b, va_loader = GraphBatcher(tr, dev), GraphBatcher(va, dev)
    print(f"Run '{a.name}' on {dev} | train {a.train_case}: {len(tr)} graphs"
          + (f" ({len(tr) - tr.n_aug} original + {tr.n_aug} " + ("N-1" if a.n1 else f"augmented, pool {a.aug}") + ")"
             if tr.n_aug else "")
          + f", val {len(va)} graphs")
    print(f"Target std (pu/rad): " + ", ".join(f"{k}={v:.4f}" for k, v in sc.items()))

    if a.init:
        ck = torch.load(a.init, map_location=dev, weights_only=False)
        model = build_model(ck["model_cfg"]).to(dev)
        model.load_state_dict(ck["state_dict"])
        print(f"Initialised from {a.init}")
    else:
        if a.arch == "hetero":
            model = HeteroOPFNet(a.hidden, a.layers, a.heads, use_pe=not a.no_pe,
                                 use_attn=not a.no_attn).to(dev)
        elif a.arch in ("gcn", "gat"):
            model = HomoGNN(a.arch, a.hidden, a.layers, a.heads).to(dev)
        else:
            model = MLPBaseline(tr.t["n_bus"], tr.t["n_gen"], hidden=a.hidden,
                                n_branch=tr.t["n_branch"] if a.arch == "mlp_topo" else 0).to(dev)
        model.set_normalization(norm_stats(dev))
        print("Input normalisation: IEEE 14 training-split statistics")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.01)

    best, bad, hist, step, t0 = float("inf"), 0, [], 0, time.time()
    run_sup = run_pb = run_br = 0.0
    while step < a.steps and bad < a.patience:
        for b in tr_b.batches(min(a.batch, tr_b.n), shuffle=True):
            model.train()
            sup, pb, br = losses(b, model(b), sc, a.ang_max)
            w_pb = a.lambda_pb * min(1.0, step / max(1, a.pb_warmup))
            loss = sup + w_pb * pb + a.lambda_br * br
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            run_sup += sup.item(); run_pb += pb.item(); run_br += br.item()
            if step % a.eval_every == 0 or step == a.steps:
                v = val_loss(model, va_loader, sc, a, dev)
                k = a.eval_every
                hist.append(dict(step=step, train_sup=run_sup / k, train_pb=run_pb / k,
                                 train_br=run_br / k, val=v, lr=sched.get_last_lr()[0]))
                run_sup = run_pb = run_br = 0.0
                flag = ""
                if v < best:
                    best, bad, flag = v, 0, " *"
                    torch.save(dict(state_dict=model.state_dict(), model_cfg=model.cfg,
                                    args=vars(a), step=step, val=v, scales=sc), out / "best.pt")
                else:
                    bad += 1
                h = hist[-1]
                print(f"step {step:6d} | sup {h['train_sup']:.4f} pb {h['train_pb']:.2e} "
                      f"br {h['train_br']:.2e} | val {v:.4f}{flag} | {(time.time() - t0) / 60:.1f} min")
            if step >= a.steps or bad >= a.patience:
                break
    pd.DataFrame(hist).to_csv(out / "history.csv", index=False)
    print(f"Stopped at step {step} (best val {best:.4f}); training time {(time.time() - t0) / 60:.1f} min")

    ck = torch.load(out / "best.pt", map_location=dev, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    cases = [a.train_case] if model.cfg.get("arch", "").startswith("mlp") else a.eval_cases  # MLP: one grid
    rows = [evaluate(model, c, "test", dev) for c in cases]
    n1_test = PROC / "graph_tensors_case30_n1test.pt"
    if n1_test.exists() and a.n1:                       # N-1 adaptation runs: also the N-1 test set
        r = evaluate(model, "case30", device=dev, timing=False,
                     ds=OPFGraphDataset("case30", tensor_file=n1_test.name))
        r["case"] = "case30_n1"
        rows.append(r)
    for r in rows:
        r.update(run=a.name, best_step=ck["step"], train_graphs=len(tr))
    json.dump(rows, open(out / "metrics.json", "w"), indent=2)
    print("\nTEST METRICS")
    print_metrics(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-case", choices=["case14", "case30", "case57"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--subset", type=float, default=None, help="fraction from finetune_subsets json")
    ap.add_argument("--aug", choices=["none", "L", "LP", "LPT"], default="none",
                    help="IEEE 14 augmentation pool added to the training set")
    ap.add_argument("--n1", action="store_true",
                    help="N-1 adaptation: add each selected hour's N-1 sample (case30 n1train/n1val sets)")
    ap.add_argument("--n1-holdout", type=int, default=0,
                    help="unseen-outage experiment: hold this many branches' outages out of N-1 training data")
    ap.add_argument("--n1-holdout-seed", type=int, default=7)
    ap.add_argument("--small-val", action="store_true", help="use the matching small val subset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="checkpoint to start from (transfer)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)   # research plan range 1e-4 .. 5e-4
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lambda-pb", type=float, default=0.1,
                    help="physics (power-balance) weight; 0.1 selected on IEEE 14 validation (tune_lambda.py)")
    ap.add_argument("--lambda-br", type=float, default=1.0,
                    help="weight of the branch hinge (thermal current limit + angle-difference limit)")
    ap.add_argument("--ang-max-deg", type=float, default=30.0, help="branch angle-difference limit (PGLib)")
    ap.add_argument("--arch", choices=["hetero", "gcn", "gat", "mlp", "mlp_topo"], default="hetero",
                    help="model: hetero (proposed) or baselines gcn / gat (homogeneous) / mlp (fixed size)")
    ap.add_argument("--no-pe", action="store_true", help="ablation: without ER positional encoding")
    ap.add_argument("--no-attn", action="store_true", help="ablation: without global ER-biased attention")
    ap.add_argument("--pb-warmup", type=int, default=2000)
    ap.add_argument("--eval-cases", nargs="+", default=["case14", "case30"])
    args = ap.parse_args()
    args.ang_max = float(torch.deg2rad(torch.tensor(args.ang_max_deg)))
    main(args)
