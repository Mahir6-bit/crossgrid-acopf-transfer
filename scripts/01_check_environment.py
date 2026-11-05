"""
01_check_environment.py
Verifies PyTorch + CUDA + PyTorch Geometric, including a tiny heterogeneous GNN
forward/backward pass on the GPU (the same building blocks the AC-OPF model will use).

Run from the project folder with the venv active:
    python scripts\\01_check_environment.py
"""

import sys
import time

import torch

print(f"Python        : {sys.version.split()[0]}")
print(f"PyTorch       : {torch.__version__}")
print(f"CUDA build    : {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dev.type == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"GPU           : {p.name} | {p.total_memory / 1024**3:.1f} GB | "
          f"compute capability {p.major}.{p.minor}")
    a = torch.randn(2048, 2048, device=dev)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    print(f"GPU matmul    : OK ({(time.perf_counter() - t0) * 1000 / 20:.2f} ms per 2048x2048)")
else:
    print("!! Running on CPU - check the CUDA install if you expected the GPU.")

import torch_geometric
from torch_geometric.data import HeteroData
from torch_geometric.nn import GraphConv, HeteroConv

print(f"PyG           : {torch_geometric.__version__}")
opt = []
for m in ["pyg_lib", "torch_scatter", "torch_sparse"]:
    try:
        __import__(m); opt.append(f"{m}=yes")
    except Exception:
        opt.append(f"{m}=no")
print(f"PyG extensions: {', '.join(opt)} (optional, not required)")

# Tiny bus/generator heterogeneous graph like the AC-OPF model will use
d = HeteroData()
d["bus"].x = torch.randn(14, 4)
d["gen"].x = torch.randn(5, 6)
ei = torch.tensor([[0, 1, 1, 2, 3], [1, 2, 4, 3, 4]])
d["bus", "line", "bus"].edge_index = torch.cat([ei, ei.flip(0)], 1)
d["gen", "at", "bus"].edge_index = torch.tensor([[0, 1, 2, 3, 4], [0, 1, 2, 5, 7]])
d["bus", "has", "gen"].edge_index = d["gen", "at", "bus"].edge_index.flip(0)
d = d.to(dev)

conv = HeteroConv({
    ("bus", "line", "bus"): GraphConv((-1, -1), 32),
    ("gen", "at", "bus"): GraphConv((-1, -1), 32),
    ("bus", "has", "gen"): GraphConv((-1, -1), 32),
}, aggr="sum").to(dev)
out = conv(d.x_dict, d.edge_index_dict)
loss = sum(o.pow(2).mean() for o in out.values())
loss.backward()
print(f"HeteroConv    : OK on {dev} | bus out {tuple(out['bus'].shape)}, gen out {tuple(out['gen'].shape)}")
print("\nAll checks passed.")
