"""
opf_dataset.py  (helper module; run directly for a self-test)

Serves heterogeneous AC-OPF graphs from data\\processed\\graph_tensors_<case>.pt and,
for the IEEE 14 source grid, the augmented pools graph_tensors_case14_aug_<pool>.pt.

Usage:
    ds = OPFGraphDataset("case14", split="train")                       # original hours
    ds = OPFGraphDataset("case14", split="train", aug_pools=["LPT"])    # + augmented samples
    ds = OPFGraphDataset("case30", timestamps=[...])                    # fine-tuning subset
    gb = GraphBatcher(ds, device); for batch in gb.batches(64, shuffle=True): ...

Augmented samples carry per-sample branch parameters, costs and an N-1 edge mask; the
batcher removes out-of-service edges, so message passing and physics never use them.
With equalize=True (default) every augmentation pool is subsampled to the size of the
smallest pool, so ablation arms (L / LP / LPT) have the same number of samples.

Self-test:  python scripts\\opf_dataset.py
"""

import numpy as np
import torch
from torch_geometric.data import HeteroData

from networks import PROJECT_DIR

PROC = PROJECT_DIR / "data" / "processed"
PER_SAMPLE = ["pd", "qd", "vm", "va", "pg", "qg", "flow_p", "flow_q", "cost"]
ALL_POOLS = ["L", "LP", "LPT"]


def _load(name):
    return torch.load(PROC / name, weights_only=False)


def n1_holdout_branches(k=8, seed=7):
    """Branches whose outages are HELD OUT of all IEEE 30 N-1 training/validation data
    (unseen-outage experiment). Drawn from the outages present in the N-1 test set."""
    if k <= 0:
        return []
    cand = sorted(set(_load("graph_tensors_case30_n1test.pt")["outage"].tolist()))
    return sorted(int(x) for x in np.random.default_rng(seed).choice(cand, size=k, replace=False))


def effective_resistance(edge_attr, edge_index, n_bus, edge_mask=None):
    """
    Effective-resistance positional encoding (research plan, Step 3).
    Susceptance-weighted Laplacian L (w = x / (r^2 + x^2) per in-service branch), pseudo-inverse L+,
    Omega_ij = L+_ii + L+_jj - 2 L+_ij. Omega is divided by its mean off-diagonal value per graph
    (scale-free, so grids with different impedance levels are comparable).
    Returns omega (N, nb, nb) and PE (N, nb, 5) = [min, max, mean, std, median] of each bus row
    (off-diagonal). edge_attr: (E, 8) shared or (N, E, 8) per sample.
    """
    ea = edge_attr if edge_attr.dim() == 3 else edge_attr[None]
    N = ea.shape[0] if edge_mask is None else max(ea.shape[0], edge_mask.shape[0])
    fwd = torch.nonzero(ea[0, :, 7] > 0.5).flatten()
    r, x = ea[:, fwd, 0].double(), ea[:, fwd, 1].double()
    w = (x / (r * r + x * x)).expand(N, -1).clone()
    if edge_mask is not None:
        w = w * edge_mask[:, fwd].double()
    f, t = edge_index[0, fwd], edge_index[1, fwd]
    W = torch.zeros(N, n_bus * n_bus, dtype=torch.float64)
    W.index_add_(1, f * n_bus + t, w)
    W = W.view(N, n_bus, n_bus)
    W = W + W.transpose(1, 2)
    L = torch.diag_embed(W.sum(-1)) - W
    Lp = torch.linalg.pinv(L)
    d = torch.diagonal(Lp, dim1=1, dim2=2)
    om = d[:, :, None] + d[:, None, :] - 2 * Lp
    off = ~torch.eye(n_bus, dtype=torch.bool)
    om = om / om[:, off].mean(-1)[:, None, None]
    rows = om[:, off].view(N, n_bus, n_bus - 1)
    pe = torch.stack([rows.min(-1).values, rows.max(-1).values, rows.mean(-1), rows.std(-1),
                      rows.median(-1).values], -1)
    return om.float(), pe.float()


def attach_structure(t):
    """Adds omega / er_pe (static, or per sample for augmented data) to a tensor dict."""
    per = t["edge_attr"].dim() == 3
    om, pe = effective_resistance(t["edge_attr"], t["edge_index"], t["n_bus"],
                                  t.get("edge_mask") if per else None)
    t["omega"], t["er_pe"] = (om, pe) if per else (om[0], pe[0])


class OPFGraphDataset(torch.utils.data.Dataset):
    def __init__(self, case, split=None, timestamps=None, aug_pools=None, equalize=True, seed=0,
                 tensor_file=None, n1_file=None, n1_exclude=()):
        """n1_file: add, for every selected hour, its N-1 sample(s) from that tensor file
        (e.g. graph_tensors_case30_n1train.pt) - used for the N-1 adaptation experiment."""
        self.t = _load(tensor_file or f"graph_tensors_{case}.pt")   # tensor_file: e.g. N-1 test set
        ts = self.t["timestamps"]
        if timestamps is not None:
            pos = {s: i for i, s in enumerate(ts)}
            self.idx = [pos[str(s)] for s in timestamps]
        elif split is not None:
            self.idx = [i for i, s in enumerate(self.t["split"]) if s == split]
        else:
            self.idx = list(range(len(ts)))
        self.case = case
        self.n_aug = 0
        if "outage" not in self.t:
            self.t["outage"] = torch.full((len(ts),), -1, dtype=torch.long)
        if aug_pools:
            assert case == "case14" and split == "train", "augmentation only for the case14 train split"
            self._merge_aug(aug_pools, equalize, seed)
        if n1_file:
            n1 = _load(n1_file)
            hours = {self.t["timestamps"][i] for i in self.idx}
            excl = set(int(b) for b in n1_exclude)
            sel = torch.as_tensor([j for j, h in enumerate(n1["base_hour"])
                                   if h in hours and int(n1["outage"][j]) not in excl], dtype=torch.long)
            self._merge([(n1, sel)])
            self.n_n1 = len(sel)
        attach_structure(self.t)

    # ---- merge original train hours with augmented pools into per-sample arrays ----
    def _merge_aug(self, pools, equalize, seed):
        base, idx = self.t, torch.as_tensor(self.idx)
        augs = [_load(f"graph_tensors_case14_aug_{p}.pt") for p in pools]
        cap = None
        if equalize:
            sizes = [len(_load(f"graph_tensors_case14_aug_{p}.pt")["cost"])
                     for p in ALL_POOLS if (PROC / f"graph_tensors_case14_aug_{p}.pt").exists()]
            cap = min(sizes)
        g = torch.Generator().manual_seed(seed)
        sel = []
        for a in augs:
            n = len(a["cost"])
            sel.append(torch.randperm(n, generator=g)[:cap].sort().values if cap and n > cap
                       else torch.arange(n))
        self._merge(list(zip(augs, sel)))

    def _merge(self, extras):
        """Concatenate the selected base samples with extra per-sample tensor dicts."""
        base, idx = self.t, torch.as_tensor(self.idx, dtype=torch.long)
        nb0, E = len(idx), base["edge_index"].shape[1]

        def per(t, key, sel):
            x = t[key]
            return x[sel] if x.dim() == 3 else x.expand(len(sel), -1, -1)

        parts = [(base, idx)] + list(extras)
        m = {k: torch.cat([t[k][s] for t, s in parts]) for k in PER_SAMPLE}
        for k in ["edge_attr", "y_phys", "gen_x"]:
            m[k] = torch.cat([per(t, k, s) for t, s in parts])
        m["edge_mask"] = torch.cat([t["edge_mask"][s] if "edge_mask" in t else
                                    torch.ones(len(s), E, dtype=torch.bool) for t, s in parts])
        m["outage"] = torch.cat([t["outage"][s] if "outage" in t else
                                 torch.full((len(s),), -1, dtype=torch.long) for t, s in parts])
        m["timestamps"] = [t["timestamps"][i] for t, s in parts for i in s.tolist()]
        m["split"] = [base["split"][i] for i in idx.tolist()] + \
            [x for t, s in extras for x in [t["split"][i] for i in s.tolist()]]
        for k in ["case", "baseMVA", "n_bus", "n_gen", "n_branch", "edge_index", "gen_bus", "bus_static"]:
            m[k] = base[k]
        self.t = m
        self.n_aug = len(m["timestamps"]) - nb0
        self.idx = list(range(len(m["timestamps"])))

    def __len__(self):
        return len(self.idx)

    def _s(self, key, i):
        x = self.t[key]
        return x[i] if x.dim() == 3 or (key == "edge_mask") else x

    def __getitem__(self, k):
        i, t = self.idx[k], self.t
        keep = t["edge_mask"][i] if "edge_mask" in t else torch.ones(t["edge_index"].shape[1], dtype=torch.bool)
        d = HeteroData()
        d["bus"].x = torch.cat([t["pd"][i, :, None], t["qd"][i, :, None], t["bus_static"]], 1)
        d["bus"].y = torch.stack([t["vm"][i], t["va"][i]], 1)
        d["bus"].pe = t["er_pe"][i] if t["er_pe"].dim() == 3 else t["er_pe"]
        d["gen"].x = self._s("gen_x", i)
        d["gen"].y = torch.stack([t["pg"][i], t["qg"][i]], 1)
        e = d["bus", "branch", "bus"]
        e.edge_index = t["edge_index"][:, keep]
        e.edge_attr = self._s("edge_attr", i)[keep]
        e.y_phys = self._s("y_phys", i)[keep]
        e.y = torch.stack([t["flow_p"][i], t["flow_q"][i]], 1)[keep]
        g2b = torch.stack([torch.arange(t["n_gen"]), t["gen_bus"]])
        d["gen", "gen_to_bus", "bus"].edge_index = g2b
        d["bus", "bus_to_gen", "gen"].edge_index = g2b.flip(0)
        d.cost = t["cost"][i].view(1)
        return d


class GraphBatcher:
    """
    Fast replacement for the PyG DataLoader: builds a batch with a few tensor operations
    directly on the GPU. Supports shared (static) or per-sample branch parameters / costs
    and an optional per-sample edge mask (N-1 outages).
    """

    def __init__(self, ds, device):
        t, idx = ds.t, torch.as_tensor(ds.idx, dtype=torch.long)
        to = lambda x: x.to(device)
        self.dev, self.n = device, len(ds.idx)
        self.nb, self.ng = t["n_bus"], t["n_gen"]
        self.pd, self.qd = to(t["pd"][idx]), to(t["qd"][idx])
        self.vm, self.va = to(t["vm"][idx]), to(t["va"][idx])
        self.pg, self.qg = to(t["pg"][idx]), to(t["qg"][idx])
        self.fp, self.fq = to(t["flow_p"][idx]), to(t["flow_q"][idx])
        self.cost = to(t["cost"][idx])
        self.bus_static, self.gen_bus = to(t["bus_static"]), to(t["gen_bus"])
        self.ei = to(t["edge_index"])
        sel = lambda x: to(x[idx]) if x.dim() == 3 else to(x)
        self.ea, self.yp, self.gen_x = sel(t["edge_attr"]), sel(t["y_phys"]), sel(t["gen_x"])
        self.mask = to(t["edge_mask"][idx]) if "edge_mask" in t else None
        self.pe = to(t["er_pe"][idx]) if t["er_pe"].dim() == 3 else to(t["er_pe"])
        self.om = to(t["omega"][idx]) if t["omega"].dim() == 3 else to(t["omega"])
        self.outage = to(t["outage"][idx]) if "outage" in t else torch.full((self.n,), -1, device=device)
        self.E = self.ei.shape[1]
        self.hours = idx

    def _per(self, x, rows, B):
        return x[rows].reshape(B * x.shape[1], -1) if x.dim() == 3 else x.repeat(B, 1)

    def make(self, rows):
        B, nb, ng, E = len(rows), self.nb, self.ng, self.E
        ar = torch.arange(B, device=self.dev)
        d = HeteroData()
        d["bus"].x = torch.cat([self.pd[rows, :, None], self.qd[rows, :, None],
                                self.bus_static.expand(B, nb, -1)], -1).reshape(B * nb, -1)
        d["bus"].y = torch.stack([self.vm[rows], self.va[rows]], -1).reshape(B * nb, 2)
        d["bus"].batch = ar.repeat_interleave(nb)
        d["bus"].pe = (self.pe[rows] if self.pe.dim() == 3 else self.pe.expand(B, -1, -1)).reshape(B * nb, -1)
        d.omega = self.om[rows] if self.om.dim() == 3 else self.om.expand(B, -1, -1)
        d.n_bus = nb
        d.outage = self.outage[rows]                 # outaged branch index per graph (-1 = intact)
        d["gen"].x = self._per(self.gen_x, rows, B)
        d["gen"].y = torch.stack([self.pg[rows], self.qg[rows]], -1).reshape(B * ng, 2)
        d["gen"].batch = ar.repeat_interleave(ng)
        ei = self.ei.repeat(1, B) + (ar * nb).repeat_interleave(E)
        ea, yp = self._per(self.ea, rows, B), self._per(self.yp, rows, B)
        ey = torch.stack([self.fp[rows], self.fq[rows]], -1).reshape(B * E, 2)
        if self.mask is not None:
            keep = self.mask[rows].reshape(-1)
            ei, ea, yp, ey = ei[:, keep], ea[keep], yp[keep], ey[keep]
        e = d["bus", "branch", "bus"]
        e.edge_index, e.edge_attr, e.y_phys, e.y = ei, ea, yp, ey
        g2b = torch.stack([torch.arange(B * ng, device=self.dev),
                           self.gen_bus.repeat(B) + (ar * nb).repeat_interleave(ng)])
        d["gen", "gen_to_bus", "bus"].edge_index = g2b
        d["bus", "bus_to_gen", "gen"].edge_index = g2b.flip(0)
        d.cost = self.cost[rows]
        d.num_graphs = B
        return d

    def batches(self, batch_size, shuffle=False):
        order = torch.randperm(self.n, device=self.dev) if shuffle else torch.arange(self.n, device=self.dev)
        for s in range(0, self.n, batch_size):
            yield self.make(order[s:s + batch_size])


if __name__ == "__main__":
    from physics import power_balance_residual, scatter_to_bus

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def residual(batch):
        b, e = batch["bus"], batch["bus", "branch", "bus"]
        g2b = batch["gen", "gen_to_bus", "bus"].edge_index
        nb = b.num_nodes
        dp, dq = power_balance_residual(b.y[:, 0], b.y[:, 1], b.x[:, 0], b.x[:, 1], b.x[:, 2], b.x[:, 3],
                                        scatter_to_bus(batch["gen"].y[:, 0], g2b[1], nb),
                                        scatter_to_bus(batch["gen"].y[:, 1], g2b[1], nb),
                                        e.edge_index, e.y_phys)
        return dp.abs().max().item() * 100, dq.abs().max().item() * 100, nb, e.edge_index.shape[1]

    for case in ["case14", "case30"]:
        print(f"\n[{case}]")
        for s in ["train", "val", "test"]:
            print(f"  {s:5s}: {len(OPFGraphDataset(case, split=s))} graphs")
        ds = OPFGraphDataset(case, split="train")
        g = ds[0]
        print(f"  one graph: bus x {tuple(g['bus'].x.shape)}, gen x {tuple(g['gen'].x.shape)}, "
              f"branch edges {tuple(g['bus', 'branch', 'bus'].edge_index.shape)}, "
              f"edge_attr {tuple(g['bus', 'branch', 'bus'].edge_attr.shape)}")
        dp, dq, nb, ne = residual(GraphBatcher(ds, dev).make(torch.arange(64, device=dev)))
        full = OPFGraphDataset(case)
        dth = (full.t["va"][:, full.t["edge_index"][0]] - full.t["va"][:, full.t["edge_index"][1]]).abs()
        pe = full.t["er_pe"]
        print(f"  ER-PE (bus 1): min {pe[0, 0]:.3f}, max {pe[0, 1]:.3f}, mean {pe[0, 2]:.3f}, std {pe[0, 3]:.3f}, "
              f"median {pe[0, 4]:.3f} | max branch angle difference in labels: {torch.rad2deg(dth.max()):.1f} deg")
        print(f"  batch of 64 on {dev}: {nb} buses, {ne} edges | physics residual on labels: "
              f"max |dP| {dp:.2e} MW, max |dQ| {dq:.2e} Mvar")

    for pool in ALL_POOLS:
        if not (PROC / f"graph_tensors_case14_aug_{pool}.pt").exists():
            continue
        ds = OPFGraphDataset("case14", split="train", aug_pools=[pool])
        gb = GraphBatcher(ds, dev)
        worst = [0.0, 0.0]
        n_edges = []
        for b in gb.batches(512):
            dp, dq, _, ne = residual(b)
            worst = [max(worst[0], dp), max(worst[1], dq)]
            n_edges.append(ne / b.num_graphs)
        print(f"\n[case14 train + aug {pool}] {len(ds)} graphs ({len(ds) - ds.n_aug} original + {ds.n_aug} "
              f"augmented) | mean directed edges/graph {sum(n_edges) / len(n_edges):.2f} | physics residual on "
              f"labels: max |dP| {worst[0]:.2e} MW, max |dQ| {worst[1]:.2e} Mvar")
    print("\nDataset self-test passed.")
