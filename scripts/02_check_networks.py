"""
02_check_networks.py
Confirms the pandapower network models we will use for AC-OPF label generation
match the dataset (MATPOWER case14 / case30) and solve at nominal load.

Run from the project folder with the venv active:
    python scripts\\02_check_networks.py
"""

import warnings

import pandapower as pp
import pandapower.networks as pn

warnings.filterwarnings("ignore")

EXPECTED = {  # from the dataset analysis (1-based MATPOWER bus numbers)
    "case14": dict(n_bus=14, n_branch=20, gen_buses={1, 2, 3, 6, 8}),
    "case30": dict(n_bus=30, n_branch=41, gen_buses={1, 2, 13, 22, 23, 27}),
}

for name, builder in [("case14", pn.case14), ("case30", pn.case30)]:
    net = builder()
    exp = EXPECTED[name]
    print("\n" + "=" * 70 + f"\n{name}  (pandapower {pp.__version__})\n" + "=" * 70)

    n_branch = len(net.line) + len(net.trafo)
    gen_buses = set((net.gen.bus + 1).tolist()) | set((net.ext_grid.bus + 1).tolist())
    print(f"buses: {len(net.bus)} (expect {exp['n_bus']}) | lines: {len(net.line)} + "
          f"trafos: {len(net.trafo)} = {n_branch} (expect {exp['n_branch']})")
    print(f"generator buses (1-based): {sorted(gen_buses)} "
          f"-> match dataset: {gen_buses == exp['gen_buses']}")
    print(f"nominal load: P = {net.load.p_mw.sum():.1f} MW, Q = {net.load.q_mvar.sum():.1f} Mvar")
    pmax = net.gen.max_p_mw.sum() + net.ext_grid.max_p_mw.sum()
    print(f"total Pmax = {pmax:.1f} MW")
    print(f"voltage limits: {net.bus.min_vm_pu.min()} - {net.bus.max_vm_pu.max()} pu")
    print(f"cost functions defined: {len(net.poly_cost)} (poly_cost rows)")

    pp.runopp(net)
    print(f"AC-OPF at nominal load: converged = {net.OPF_converged}, cost = {net.res_cost:.2f}")
    print(f"  max line loading = {net.res_line.loading_percent.max():.1f}%"
          + (f", max trafo loading = {net.res_trafo.loading_percent.max():.1f}%"
             if len(net.trafo) else ""))
    print(f"  Vm range = {net.res_bus.vm_pu.min():.3f} - {net.res_bus.vm_pu.max():.3f} pu")
    print(f"  losses = {net.res_line.pl_mw.sum() + (net.res_trafo.pl_mw.sum() if len(net.trafo) else 0):.2f} MW")
