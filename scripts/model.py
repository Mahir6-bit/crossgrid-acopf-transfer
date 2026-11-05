"""
model.py  (helper module)

Physics-informed heterogeneous GNN for AC-OPF, following the research plan (Step 4):

  Encoder   : separate 2-layer MLP per node / edge type (hidden 128 by default)
  Processor : K layers (default 4), each with
                1. local heterogeneous message passing
                   - branch edge update  e <- LN(e + MLP[h_src, h_dst, e])
                   - node update via GATv2 (bus-branch-bus, uses edge embeddings) and
                     SAGE (gen <-> bus), residual + LayerNorm
                2. global self-attention over the bus nodes of each graph, biased by the
                   effective resistance:  score_ij - softplus(alpha_head) * Omega_ij
                   (+ feed-forward block, residual + LayerNorm)
  Decoder   : Vm = Vmin + sigmoid * (Vmax - Vmin); Pg, Qg likewise within their limits;
              Va predicted directly and fixed to 0 at the slack bus.

  Inputs: bus [Pd, Qd, Gs, Bs, Vmin, Vmax, PQ, PV, slack] + ER-PE (5);
          gen [Pmin, Pmax, Qmin, Qmax, relative c1, relative 2*c2*Pmax, slack];
          branch [r, x, b, log(1+rate), tap, shift, is_trafo, is_forward].
  Normalisation statistics come ONLY from the IEEE 14 training split (stored as buffers).
  No bus IDs or fixed-size layers -> the same weights run on any grid.

  Ablation switches (E5): use_pe=False zeroes the ER-PE input, use_attn=False removes the
  global attention blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, SAGEConv
from torch_geometric.utils import scatter

BUS_IN, GEN_IN, EDGE_IN = 9 + 5, 7, 8
BRANCH = ("bus", "branch", "bus")
G2B = ("gen", "gen_to_bus", "bus")
B2G = ("bus", "bus_to_gen", "gen")


def mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


def gen_features(gx, gbatch):
    """raw gen.x [Pmin,Pmax,Qmin,Qmax,c2,c1,c0,slack] -> scale-free features (7)."""
    pmin, pmax, qmin, qmax, c2, c1, _c0, slack = gx.unbind(-1)
    mc_max = c1 + 2 * c2 * pmax                       # marginal cost at Pmax
    ref = scatter(mc_max, gbatch, reduce="max")[gbatch].clamp_min(1e-6)
    return torch.stack([pmin, pmax, qmin, qmax, c1 / ref, 2 * c2 * pmax / ref, slack], -1)


def edge_features(ea):
    return torch.cat([ea[:, :3], torch.log1p(ea[:, 3:4]), ea[:, 4:]], -1)


def raw_features(batch):
    """Model input features before normalisation: bus (14), gen (7), branch edges (8)."""
    return (torch.cat([batch["bus"].x, batch["bus"].pe], -1),
            gen_features(batch["gen"].x, batch["gen"].batch),
            edge_features(batch[BRANCH].edge_attr))


@torch.no_grad()
def feature_stats(batches):
    """Mean/std of every input feature over an iterable of batches (IEEE 14 train split)."""
    acc = {"bus": [], "gen": [], "edge": []}
    for b in batches:
        xb, xg, xe = raw_features(b)
        acc["bus"].append(xb); acc["gen"].append(xg); acc["edge"].append(xe)
    out = {}
    for k, v in acc.items():
        x = torch.cat(v)
        std = x.std(0)
        out[f"{k}_mean"] = x.mean(0)
        out[f"{k}_std"] = torch.where(std < 1e-6, torch.ones_like(std), std)
    return out


class ERBiasedAttention(nn.Module):
    """Global self-attention over the buses of each graph, biased by effective resistance."""

    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.mha = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.alpha = nn.Parameter(torch.zeros(heads))
        self.n1, self.n2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.ff = mlp(hidden, 2 * hidden, hidden)

    def forward(self, h, omega, n_graphs):
        nb = h.shape[0] // n_graphs
        x = h.view(n_graphs, nb, -1)
        bias = -(F.softplus(self.alpha)[None, :, None, None] * omega[:, None])
        a, _ = self.mha(x, x, x, attn_mask=bias.reshape(n_graphs * self.heads, nb, nb),
                        need_weights=False)
        x = self.n1(x + a)
        x = self.n2(x + self.ff(x))
        return x.reshape(n_graphs * nb, -1)


class HeteroOPFNet(nn.Module):
    def __init__(self, hidden=128, layers=4, heads=4, use_pe=True, use_attn=True):
        super().__init__()
        self.cfg = dict(arch="hetero", hidden=hidden, layers=layers, heads=heads, use_pe=use_pe, use_attn=use_attn)
        for k, n in [("bus", BUS_IN), ("gen", GEN_IN), ("edge", EDGE_IN)]:
            self.register_buffer(f"{k}_mean", torch.zeros(n))
            self.register_buffer(f"{k}_std", torch.ones(n))
        self.bus_enc = mlp(BUS_IN, hidden, hidden)
        self.gen_enc = mlp(GEN_IN, hidden, hidden)
        self.edge_enc = mlp(EDGE_IN, hidden, hidden)
        self.convs, self.norms = nn.ModuleList(), nn.ModuleList()
        self.edge_upd, self.edge_norm = nn.ModuleList(), nn.ModuleList()
        self.attn = nn.ModuleList()
        for _ in range(layers):
            self.edge_upd.append(mlp(3 * hidden, hidden, hidden))
            self.edge_norm.append(nn.LayerNorm(hidden))
            self.convs.append(HeteroConv({
                BRANCH: GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=hidden,
                                  add_self_loops=False),
                G2B: SAGEConv((hidden, hidden), hidden),
                B2G: SAGEConv((hidden, hidden), hidden),
            }, aggr="sum"))
            self.norms.append(nn.ModuleDict({"bus": nn.LayerNorm(hidden), "gen": nn.LayerNorm(hidden)}))
            if use_attn:
                self.attn.append(ERBiasedAttention(hidden, heads))
        self.bus_head = mlp(hidden, hidden, 2)
        self.gen_head = mlp(hidden, hidden, 2)

    def set_normalization(self, stats):
        for k, v in stats.items():
            getattr(self, k).copy_(v)

    def forward(self, batch):
        xb, xg, xe = raw_features(batch)
        xb = (xb - self.bus_mean) / self.bus_std
        if not self.cfg["use_pe"]:
            xb = torch.cat([xb[:, :9], torch.zeros_like(xb[:, 9:])], -1)
        h = {"bus": self.bus_enc(xb), "gen": self.gen_enc((xg - self.gen_mean) / self.gen_std)}
        e = self.edge_enc((xe - self.edge_mean) / self.edge_std)
        src, dst = batch[BRANCH].edge_index
        B = batch.num_graphs
        for k, conv in enumerate(self.convs):
            e = self.edge_norm[k](e + self.edge_upd[k](torch.cat([h["bus"][src], h["bus"][dst], e], -1)))
            out = conv(h, batch.edge_index_dict, edge_attr_dict={BRANCH: e})
            h = {t: self.norms[k][t](h[t] + F.gelu(out[t])) for t in h}
            if self.cfg["use_attn"]:
                h["bus"] = self.attn[k](h["bus"], batch.omega, B)

        ob, og = self.bus_head(h["bus"]), self.gen_head(h["gen"])
        bx, gx = batch["bus"].x, batch["gen"].x            # raw values for the limit decoder
        vmin, vmax, slack = bx[:, 4], bx[:, 5], bx[:, 8]
        pmin, pmax, qmin, qmax = gx[:, 0], gx[:, 1], gx[:, 2], gx[:, 3]
        return dict(
            vm=vmin + (vmax - vmin) * torch.sigmoid(ob[:, 0]),
            va=ob[:, 1] * (1 - slack),
            pg=pmin + (pmax - pmin) * torch.sigmoid(og[:, 0]),
            qg=qmin + (qmax - qmin) * torch.sigmoid(og[:, 1]),
        )


# =====================================================================================
# Baselines (research plan, Step 6): homogeneous GCN / GAT and a fixed-size MLP.
# Same inputs (without ER-PE), same normalisation, same limit-respecting decoder and
# the same loss, so differences come from the architecture only.
# =====================================================================================
from torch_geometric.nn import GATConv, GCNConv  # noqa: E402

GEN_AGG = 4   # per bus: sum of Pmin, Pmax, Qmin, Qmax of the generators at that bus


def _decode(batch, ob, og):
    bx, gx = batch["bus"].x, batch["gen"].x
    vmin, vmax, slack = bx[:, 4], bx[:, 5], bx[:, 8]
    return dict(vm=vmin + (vmax - vmin) * torch.sigmoid(ob[:, 0]), va=ob[:, 1] * (1 - slack),
                pg=gx[:, 0] + (gx[:, 1] - gx[:, 0]) * torch.sigmoid(og[:, 0]),
                qg=gx[:, 2] + (gx[:, 3] - gx[:, 2]) * torch.sigmoid(og[:, 1]))


class _Norm(nn.Module):
    """Holds the IEEE 14 normalisation buffers (same layout as HeteroOPFNet)."""

    def __init__(self):
        super().__init__()
        for k, n in [("bus", BUS_IN), ("gen", GEN_IN), ("edge", EDGE_IN)]:
            self.register_buffer(f"{k}_mean", torch.zeros(n))
            self.register_buffer(f"{k}_std", torch.ones(n))

    def set_normalization(self, stats):
        for k, v in stats.items():
            getattr(self, k).copy_(v)

    def inputs(self, batch):
        xb, xg, xe = raw_features(batch)
        xb = ((xb - self.bus_mean) / self.bus_std)[:, :9]          # without ER-PE
        xg = (xg - self.gen_mean) / self.gen_std
        xe = (xe - self.edge_mean) / self.edge_std
        g2b = batch[G2B].edge_index[1]
        gx = batch["gen"].x[:, :4]
        agg = scatter(gx, g2b, dim=0, dim_size=xb.shape[0], reduce="sum") / self.bus_std[:2].mean()
        return xb, xg, xe, agg


class HomoGNN(_Norm):
    """Homogeneous bus graph: GCN (no edge features) or GAT (with edge features)."""

    def __init__(self, kind="gcn", hidden=128, layers=4, heads=4):
        super().__init__()
        self.cfg = dict(arch=kind, hidden=hidden, layers=layers, heads=heads)
        self.enc = mlp(9 + GEN_AGG, hidden, hidden)
        self.edge_enc = mlp(EDGE_IN, hidden, hidden) if kind == "gat" else None
        self.convs = nn.ModuleList(
            GCNConv(hidden, hidden) if kind == "gcn" else
            GATConv(hidden, hidden // heads, heads=heads, edge_dim=hidden, add_self_loops=False)
            for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.bus_head = mlp(hidden, hidden, 2)
        self.gen_head = mlp(hidden + GEN_IN, hidden, 2)

    def forward(self, batch):
        xb, xg, xe, agg = self.inputs(batch)
        h = self.enc(torch.cat([xb, agg], -1))
        ei = batch[BRANCH].edge_index
        ea = self.edge_enc(xe) if self.edge_enc is not None else None
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, ei) if ea is None else conv(h, ei, edge_attr=ea)
            h = norm(h + F.gelu(out))
        g2b = batch[G2B].edge_index[1]
        og = self.gen_head(torch.cat([h[g2b], xg], -1))
        return _decode(batch, self.bus_head(h), og)


class MLPBaseline(_Norm):
    """Fixed-size feed-forward network for ONE grid (cannot be transferred to another grid).
    n_branch > 0 adds a one-hot vector of the outaged branch to the input (all zeros = intact):
    a grid-specific way to tell the MLP about N-1 topology changes ("mlp_topo")."""

    def __init__(self, n_bus, n_gen, hidden=512, layers=3, n_branch=0):
        super().__init__()
        self.cfg = dict(arch="mlp_topo" if n_branch else "mlp", n_bus=n_bus, n_gen=n_gen,
                        hidden=hidden, layers=layers, n_branch=n_branch)
        dims = [2 * n_bus + n_branch] + [hidden] * layers
        net = []
        for i, o in zip(dims[:-1], dims[1:]):
            net += [nn.Linear(i, o), nn.GELU()]
        self.body = nn.Sequential(*net)
        self.out = nn.Linear(hidden, 2 * n_bus + 2 * n_gen)

    def forward(self, batch):
        xb, _, _, _ = self.inputs(batch)
        B, nb, ng = batch.num_graphs, self.cfg["n_bus"], self.cfg["n_gen"]
        x = xb[:, :2].reshape(B, 2 * nb)                          # loads Pd, Qd of all buses
        if self.cfg["n_branch"]:
            o = batch.outage
            onehot = torch.zeros(B, self.cfg["n_branch"] + 1, device=x.device)
            onehot[torch.arange(B, device=x.device), o.clamp(min=-1) + 1] = 1.0
            x = torch.cat([x, onehot[:, 1:]], -1)                # column 0 = intact, dropped
        z = self.out(self.body(x))
        ob = z[:, :2 * nb].reshape(B * nb, 2)
        og = z[:, 2 * nb:].reshape(B * ng, 2)
        return _decode(batch, ob, og)


def build_model(cfg):
    """Re-create any model from the cfg stored in a checkpoint."""
    cfg = dict(cfg)
    arch = cfg.pop("arch", "hetero")
    if arch == "hetero":
        return HeteroOPFNet(**cfg)
    if arch in ("gcn", "gat"):
        return HomoGNN(kind=arch, **cfg)
    if arch in ("mlp", "mlp_topo"):
        return MLPBaseline(**cfg)
    raise ValueError(arch)
