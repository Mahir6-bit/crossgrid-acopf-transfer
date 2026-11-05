"""
physics.py  (helper module - imported by other scripts)

AC power-flow physics in PyTorch, written for DIRECTED branch edges so it works for any
grid and for batched PyG graphs.

Each branch k (from f, to t) appears as two directed edges:
    forward  f->t : Y_self = Yff, Y_mut = Yft     (flow leaving f)
    reverse  t->f : Y_self = Ytt, Y_mut = Ytf     (flow leaving t)
y_phys[e] = [Re Y_self, Im Y_self, Re Y_mut, Im Y_mut] (per-unit, MATPOWER pi-model incl. taps/shifts).

Complex power leaving source bus s through edge e towards d:
    S = V_s * conj(Y_self V_s + Y_mut V_d)
All quantities in per-unit on baseMVA = 100. Tensors may have leading batch dims (..., n).
"""

import torch


def edge_flows(vm, va, edge_index, y_phys):
    """P, Q (per-unit) leaving the source bus of every directed edge. Shapes (..., E)."""
    s, d = edge_index[0], edge_index[1]
    vs, vd = vm[..., s], vm[..., d]
    th = va[..., s] - va[..., d]
    g_ss, b_ss, g_sd, b_sd = y_phys.unbind(-1)
    c, sn = torch.cos(th), torch.sin(th)
    p = vs * vs * g_ss + vs * vd * (g_sd * c + b_sd * sn)
    q = -vs * vs * b_ss + vs * vd * (g_sd * sn - b_sd * c)
    return p, q


def scatter_to_bus(values, index, n_bus):
    """Sum edge/generator values onto buses along the last dimension."""
    out = values.new_zeros(values.shape[:-1] + (n_bus,))
    return out.index_add_(-1, index, values)


def power_balance_residual(vm, va, pd, qd, gs, bs, pg_bus, qg_bus, edge_index, y_phys):
    """
    Nodal mismatch (per-unit):  dP = Pg - Pd - P_shunt - sum(P leaving)
                                dQ = Qg - Qd - Q_shunt - sum(Q leaving)
    gs, bs: bus shunt conductance/susceptance in pu (consumption = |V|^2 (gs - j bs)).
    """
    n_bus = vm.shape[-1]
    p, q = edge_flows(vm, va, edge_index, y_phys)
    p_out = scatter_to_bus(p, edge_index[0], n_bus)
    q_out = scatter_to_bus(q, edge_index[0], n_bus)
    v2 = vm * vm
    dp = pg_bus - pd - v2 * gs - p_out
    dq = qg_bus - qd + v2 * bs - q_out
    return dp, dq


def generation_cost(pg, c2, c1, c0):
    """Quadratic cost with pu coefficients (c2*base^2, c1*base, c0) and pg in pu."""
    return c2 * pg * pg + c1 * pg + c0
