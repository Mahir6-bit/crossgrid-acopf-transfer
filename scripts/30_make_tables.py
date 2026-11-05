"""
30_make_tables.py
Builds all paper tables from the result files and writes them into ONE Word document
(three-line tables, Times New Roman, caption above), plus one CSV per table.

  Numbering of the revised manuscript (Tables 1, 2, 6, 14 and 15 are written in the text):
  Main text : Table 3 hyper-parameters   Table 4 test systems     Table 5 augmentation
              Table 7 accuracy E1-E4     Table 8 baselines        Table 9 GNN vs MLP
              Table 10 ablation          Table 11 constraints/PF  Table 12 runtime
              Table 13 N-1 and unseen outages
  Supplement: S1 lambda_pb tuning        S2 significance tests (with 95 % CI and d_z if stats_ci.csv exists)
              S3 reproducibility         S4 violated limits after PF   S5 slack/tolerance sensitivity

Needs:  python -m pip install python-docx
Run from the project folder (after all experiments):
    python scripts\\30_make_tables.py
Writes results\\tables\\paper_tables.docx and results\\tables\\table_*.csv
Missing result files only skip the affected table.
"""

import importlib.metadata as md
import json
import platform
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

ROOT = Path(__file__).resolve().parents[1]
RES, PROC = ROOT / "results", ROOT / "data" / "processed"
OUT = RES / "tables"
OUT.mkdir(parents=True, exist_ok=True)

FRACS = [0.01, 0.05, 0.1, 0.25, 1.0]
DEC = dict(vm_mae_pu=4, va_mae_deg=3, pg_mae_mw=2, qg_mae_mvar=2, pb_p_mean_mw=2, pb_p_max_mw=1,
           pb_q_mean_mvar=2, branch_viol_pct=2, samples_with_branch_viol_pct=1, cost_gap_abs_pct=2)
COLS = dict(vm_mae_pu="Vm MAE (10\u207b\u00b3 p.u.)", va_mae_deg="Va MAE (\u00b0)", pg_mae_mw="Pg MAE (MW)",
            qg_mae_mvar="Qg MAE (Mvar)", pb_p_mean_mw="Mean |\u0394P| (MW)", pb_p_max_mw="Max |\u0394P| (MW)",
            branch_viol_pct="Overloaded branches (%)", samples_with_branch_viol_pct="Hours with overload (%)",
            cost_gap_abs_pct="|Cost gap| (%)")


# ------------------------------------------------------------------ helpers: numbers
def ms(values, dec=2):
    """mean ± std; decimals follow the std (2 significant digits), 'dec' only if std is 0 or n = 1."""
    v = pd.Series(values, dtype=float).dropna()
    if not len(v):
        return "\u2013"
    sd = v.std() if len(v) > 1 else 0.0
    mean_dec = int(np.clip(3 - np.floor(np.log10(abs(v.mean()))), 0, 6)) if abs(v.mean()) > 0 else dec
    if sd > 0:
        dec = int(np.clip(1 - np.floor(np.log10(sd)), 0, 6))
        dec = min(dec, mean_dec)                     # at most 4 significant digits of the mean
    else:
        dec = max(0, min(mean_dec, 6) - 1)
    if len(v) == 1:
        return f"{v.iloc[0]:.{dec}f}"
    s = f"{sd:.{dec}f}"
    if sd > 0 and float(s) == 0:                                 # std below the last shown digit
        s = f"<{10 ** -dec:.{dec}f}"
    return f"{v.mean():.{dec}f} \u00b1 {s}"


def n_train_hours():
    try:
        s = pd.read_csv(PROC / "splits.csv", index_col=0).split
        return int((s == "train").sum())
    except FileNotFoundError:
        return 5832


def hours(f):
    return max(1, int(round(f * n_train_hours())))


def pct(f):
    return f"{f * 100:g} %"


def runs():
    return pd.read_csv(RES / "all_runs.csv")


def stats_df():
    p = RES / "stats_tests.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame(
        columns=["family", "A", "B", "metric", "mean_A", "mean_B", "p_holm"])


def significant(st, family, a, b, metric):
    """True if A is significantly better (lower) than B after Holm correction."""
    d = st[(st.family == family) & (st.A == a) & (st.B == b) & (st.metric == metric)]
    return bool(len(d) and d.p_holm.iloc[0] < 0.05 and d.mean_A.iloc[0] < d.mean_B.iloc[0])


def sigmark(st, family, a, b, metric, raw=False):
    """'\u2020' A significantly better than B, '\u2021' B significantly better than A (Holm, p < 0.05);
    with raw=True '\u00b0' marks p < 0.05 before correction (not significant after Holm)."""
    d = st[(st.family == family) & (st.A == a) & (st.B == b) & (st.metric == metric)]
    if not len(d):
        return ""
    x = d.iloc[0]
    if x.p_holm < 0.05:
        return "\u2020" if x.mean_A < x.mean_B else "\u2021"
    if raw and x.p_value < 0.05:
        return "\u00b0"
    return ""


def n_params(run):
    import torch
    from evaluate import load_checkpoint
    p = RES / run / "best.pt"
    if not p.exists():
        return "\u2013"
    model, _ = load_checkpoint(p, torch.device("cpu"))
    return f"{sum(x.numel() for x in model.parameters()) / 1e6:.2f} M"


# ------------------------------------------------------------------ helpers: Word
def set_cell_border(cell, **edges):
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge, (sz, val) in edges.items():
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), val)
        el.set(qn("w:sz"), str(sz))
        el.set(qn("w:space"), "0")
        el.set(qn("w:color"), "000000")
        borders.append(el)


def no_table_borders(table):
    tbl_pr = table._tbl.tblPr
    b = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        el = OxmlElement(f"w:{edge}")
        el.set(qn("w:val"), "nil")
        b.append(el)
    tbl_pr.append(b)


def run_fmt(run, size, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    run.font.size = Pt(size)
    run.bold, run.italic = bold, italic


def para(doc, text, size=10, bold=False, italic=False, align=None, space_after=4):
    p = doc.add_paragraph()
    r = p.add_run(text)
    run_fmt(r, size, bold, italic)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.space_before = Pt(0)
    if align:
        p.alignment = align
    return p


def usable_width(doc):
    s = doc.sections[-1]
    return s.page_width - s.left_margin - s.right_margin


def is_text_col(series):
    """Left-align columns that hold words rather than numbers."""
    v = [x for x in series if x not in ("", "\u2013")]
    return bool(v) and np.mean([sum(ch.isalpha() for ch in x) > 0.5 * len(x) for x in v]) > 0.5


def add_table(doc, number, caption, df, notes=None, size=9, first_col_left=True, group_rows=None):
    """Three-line table. df: all cells as strings; columns become the header row.
    group_rows: row indices that are section labels (italic, spanning, thin rule above)."""
    df = df.fillna("\u2013").astype(str)
    group_rows = set(group_rows or [])
    body = df.drop(index=[df.index[i] for i in group_rows]) if group_rows else df
    left = [j == 0 and first_col_left or is_text_col(body.iloc[:, j]) for j in range(len(df.columns))]
    weights = []
    for j, c in enumerate(df.columns):
        longest = max([len(x) for x in body.iloc[:, j]] + [1])
        head = max(len(w) for w in str(c).split()) if str(c).split() else 1
        weights.append(float(np.clip(max(longest, head + 2, 0.45 * len(str(c))), 7, 42)))
    total = usable_width(doc)
    widths = [int(total * w / sum(weights)) for w in weights]
    df.to_csv(OUT / f"table_{number}.csv", index=False)
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    label = f"Supplementary Table {number}. " if str(number).startswith("S") else f"Table {number}. "
    run_fmt(p.add_run(label), 10, bold=True)
    run_fmt(p.add_run(caption), 10)

    t = doc.add_table(rows=len(df) + 1, cols=len(df.columns))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    no_table_borders(t)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    t._tbl.tblPr.append(layout)
    for j, w in enumerate(widths):
        t.columns[j].width = w
        for cell in t.columns[j].cells:
            cell.width = w
    hdr = t.rows[0]._tr.get_or_add_trPr()
    el = OxmlElement("w:tblHeader")
    el.set(qn("w:val"), "true")
    hdr.append(el)
    for j, c in enumerate(df.columns):
        cell = t.cell(0, j)
        cell.text = ""
        r = cell.paragraphs[0].add_run(str(c))
        run_fmt(r, size, bold=True)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if left[j] else WD_ALIGN_PARAGRAPH.CENTER
        set_cell_border(cell, top=(12, "single"), bottom=(6, "single"))
    for i, row in enumerate(df.itertuples(index=False), start=1):
        if (i - 1) in group_rows:
            merged = t.cell(i, 0).merge(t.cell(i, len(df.columns) - 1))
            merged.text = ""
            r = merged.paragraphs[0].add_run(str(row[0]))
            run_fmt(r, size, italic=True)
            if i > 1:
                set_cell_border(merged, top=(4, "single"))
            continue
        for j, v in enumerate(row):
            cell = t.cell(i, j)
            cell.text = ""
            r = cell.paragraphs[0].add_run(v if v else "\u00a0")   # empty cell keeps the small font height
            run_fmt(r, size)
            cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if left[j] else WD_ALIGN_PARAGRAPH.CENTER
    for j in range(len(df.columns)):
        set_cell_border(t.cell(len(df), j), bottom=(12, "single"))
    for row in t.rows:
        tr = row._tr.get_or_add_trPr()
        cs = OxmlElement("w:cantSplit")
        cs.set(qn("w:val"), "true")
        tr.append(cs)
        for cell in row.cells:
            for pp in cell.paragraphs:
                pp.paragraph_format.space_after = Pt(1)
                pp.paragraph_format.space_before = Pt(1)
    if notes:
        para(doc, notes, size=8, italic=False, space_after=8)
    print(f"  Table {number}: {len(df)} rows")


def landscape(doc):
    sec = doc.add_section()
    sec.orientation = WD_ORIENT.LANDSCAPE
    sec.page_width, sec.page_height = sec.page_height, sec.page_width
    return sec


def portrait(doc):
    sec = doc.add_section()
    sec.orientation = WD_ORIENT.PORTRAIT
    if sec.page_width > sec.page_height:
        sec.page_width, sec.page_height = sec.page_height, sec.page_width
    return sec


# ------------------------------------------------------------------ Table 2 test systems
def table02(doc):
    rows = {}
    for case, role in [("case14", "Source (pre-training)"), ("case30", "Target (adaptation)")]:
        br = pd.read_csv(PROC / f"network_{case}_branches.csv")
        gens = pd.read_csv(PROC / f"network_{case}_gens.csv")
        buses = pd.read_csv(PROC / f"network_{case}_buses.csv")
        bus = np.load(PROC / f"network_{case}_ppc.npz", allow_pickle=True)["bus"]
        disp = gens[gens.p_max_mw > 0]
        sync = gens[(gens.type == "gen") & (gens.p_max_mw <= 0)]
        pq = buses[buses.bus != gens[gens.type == "ext_grid"].bus.iloc[0]]
        rows[case] = {
            "Role in this study": role,
            "Data source": "PGLib-OPF v23.07 (pglib_opf_case14_ieee)" if case == "case14" else "MATPOWER case30",
            "Buses": f"{len(buses)}",
            "Branches (lines / transformers)": f"{len(br)} ({int((br.type == 'line').sum())} / "
                                               f"{int((br.type != 'line').sum())})",
            "Dispatchable generators (incl. slack)": f"{len(disp)}",
            "Synchronous condensers": f"{len(sync)}",
            "Load buses": f"{int((bus[:, 2] != 0).sum())}",
            "Nominal load, P / Q": f"{bus[:, 2].sum():.1f} MW / {bus[:, 3].sum():.1f} Mvar",
            "Generation capacity (\u03a3 Pmax)": f"{disp.p_max_mw.sum():.0f} MW",
            "Voltage limits (non-slack buses)": f"{pq.vm_min_pu.min():.2f}\u2013{pq.vm_max_pu.max():.2f} p.u.",
            "Branch thermal ratings": f"{br.rate_mva.min():.0f}\u2013{br.rate_mva.max():.0f} MVA",
            "Cost function": "quadratic" if (gens.c2 > 0).any() else "linear",
        }
    s = pd.read_csv(PROC / "splits.csv", index_col=0).split
    n = s.value_counts()
    common = {"Hourly operating points": f"{len(s)} (year 2007)",
              "Split (chronological)": f"train Jan\u2013Aug {n.get('train', 0)} h, val Sep\u2013Oct "
                                       f"{n.get('val', 0)} h, test Nov\u2013Dec {n.get('test', 0)} h"}
    df = pd.DataFrame({"Property": list(rows["case14"]) + list(common),
                       "IEEE 14-bus": list(rows["case14"].values()) + list(common.values()),
                       "IEEE 30-bus": list(rows["case30"].values()) + list(common.values())})
    add_table(doc, "2", "Properties of the source (IEEE 14-bus) and target (IEEE 30-bus) test systems and of the data used in this study.", df,
              "Load scenarios: per-bus factors \u03bb from the hourly time series mapped to "
              "\u03bb\u2032 = 0.6 + 0.2\u03bb (0.6\u20131.0 \u00d7 nominal), reactive load at constant power factor. "
              "Both grids use the same chronological split.")


# ------------------------------------------------------------------ Table 3 augmentation
def table03(doc):
    desc = {"L": "load scaling + correlated per-bus noise",
            "LP": "L + branch r, x and generator cost perturbation",
            "LPT": "LP + single-branch outage (N-1, non-bridge branches)"}
    rows, used = [], []
    for pool in ["L", "LP", "LPT"]:
        p = PROC / f"aug_case14_{pool}.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p, columns=["converged"])
        rows.append([pool, desc[pool], f"{len(d):,}", f"{int(d.converged.sum()):,} "
                     f"({d.converged.mean() * 100:.1f} %)"])
        used.append(int(d.converged.sum()))
    cap = min(used) if used else 0
    for r in rows:
        r.append(f"{cap:,}")
    df = pd.DataFrame(rows, columns=["Pool", "Perturbation", "Requested", "Converged AC-OPF",
                                     "Used (equalized)"])
    ranges = ("Ranges: global load scale U(0.7, 1.2); per-bus noise \u03c3 = 0.10, correlation \u03c1 = 0.5, "
              "clipped to \u00b10.20; branch r and x factors U(0.8, 1.2); cost factors U(0.6, 1.4); "
              "4 variants per IEEE 14 training hour; RNG seed 2026 + pool index. ")
    n0 = n_train_hours()
    total = f"Each arm of the augmentation ablation is capped to the same {cap:,} samples, so the default " \
            f"source training set (LPT) has {n0:,} original + {cap:,} augmented = {n0 + cap:,} graphs."
    add_table(doc, "3", "Source-grid (IEEE 14-bus) data augmentation pools.", df, ranges + total)
    p = RES / "augmentation_table.csv"
    if p.exists():
        pd.read_csv(p).to_csv(OUT / "table_3_ranges_from_augment_case14.csv", index=False)


# ------------------------------------------------------------------ Table 4 hyper-parameters
def table04(doc):
    rows = [("Architecture", "Hidden size / message-passing layers / heads", "128 / 4 / 4"),
            ("", "Bus\u2013bus convolution", "GATv2 with edge features + edge update"),
            ("", "Bus\u2013generator convolution", "GraphSAGE (both directions)"),
            ("", "Global attention", "multi-head, biased by effective resistance \u03a9"),
            ("", "Positional encoding", "effective resistance (\u03a9 / mean off-diagonal \u03a9)"),
            ("", "Output decoder", "sigmoid scaled to Vm, Pg, Qg limits; Va linear"),
            ("", "Trainable parameters", n_params("src14_s0")),
            ("Loss", "Supervised term", "MSE on scaled Vm, Va, Pg, Qg"),
            ("", "Power-balance weight \u03bb_pb", "0.1 (tuned on IEEE 14 validation, Supplementary Table S1)"),
            ("", "Limit weight \u03bb_br", "1.0 (thermal current limit + 30\u00b0 angle-difference hinge)"),
            ("", "Power-balance ramp-in", "2,000 steps"),
            ("Optimization", "Optimizer", "AdamW (weight decay 10\u207b\u2075), lr 3\u00d710\u207b\u2074 with cosine decay to 3\u00d710\u207b\u2076, gradient clipping at 1.0"),
            ("", "Batch size / max. steps", "64 / 20,000"),
            ("", "Early stopping", "validation loss every 250 steps, patience 20"),
            ("", "Fine-tuning", "all weights, same optimizer settings as from scratch"),
            ("Data", "Feature normalization", "statistics of IEEE 14 training set only"),
            ("", "Target data fractions", "1, 5, 10, 25, 100 % of 5,832 h (nested subsets)"),
            ("", "Seeds", "0\u20134 (main results), 0\u20132 (ablations and baselines)")]
    add_table(doc, "5", "Model and training hyper-parameters of the proposed heterogeneous GNN (identical for all runs unless stated).",
              pd.DataFrame(rows, columns=["Group", "Setting", "Value"]))


# ------------------------------------------------------------------ Table 5 main accuracy
def table05(doc):
    df = runs()
    st = stats_df()
    keys = ["vm_mae_pu", "va_mae_deg", "pg_mae_mw", "qg_mae_mvar", "pb_p_mean_mw", "cost_gap_abs_pct"]
    rows, groups = [], []

    def add(label, h, d, mark=None):
        cells = [label, h]
        for k in keys:
            c = ms(d[k] * (1000 if k == "vm_mae_pu" else 1), DEC[k])
            if mark and mark(k):
                c += "*"
            cells.append(c)
        rows.append(cells)

    groups.append(len(rows)); rows.append(["Source model (no IEEE 30 data)"] + [""] * (len(keys) + 1))
    d = df[(df.case == "case14") & (df.method == "source (E1)")]
    add("E1: tested on IEEE 14", "0", d)
    d = df[(df.case == "case30") & (df.method == "zero-shot (E2)")]
    add("E2: zero-shot on IEEE 30", "0", d)
    for method, lab, e in [("fine-tune", "Fine-tuned from IEEE 14 (E3)", "E3"), ("scratch", "Trained from scratch (E4)", "E4")]:
        groups.append(len(rows)); rows.append([lab] + [""] * (len(keys) + 1))
        for f in FRACS:
            d = df[(df.case == "case30") & (df.method == method) & np.isclose(df.fraction, f)]
            if not len(d):
                continue
            mark = (lambda k, f=f: significant(st, "E3 vs E4", f"fine-tune {f:.0%}", f"scratch {f:.0%}", k)) \
                if method == "fine-tune" else None
            add(f"{pct(f)} of IEEE 30 data", f"{hours(f):,}", d, mark)
    n_seeds = int(df[df.method == "fine-tune"].seed.nunique())
    cols = ["Model", "Target training hours"] + [COLS[k] for k in keys]
    add_table(doc, "7", "Prediction accuracy on the IEEE 30 test set (Nov\u2013Dec; E1 on the IEEE 14 test set), "
              f"mean \u00b1 std over {n_seeds} seeds.", pd.DataFrame(rows, columns=cols),
              "* fine-tuning significantly better than training from scratch with the same data "
              "(paired t-test over seeds, Holm-corrected within the family, p < 0.05). "
              "Mean |\u0394P| = mean absolute active-power mismatch at a bus; |Cost gap| = absolute relative "
              "difference between the cost of the predicted dispatch and the AC-OPF cost.",
              size=7.5, group_rows=groups)


# ------------------------------------------------------------------ Table 6 constraints / PF
def table06(doc):
    df = runs()
    df = df[df.case == "case30"]
    pf = pd.read_csv(RES / "pf_postprocess_runs.csv")
    items = [("Zero-shot (E2)", "zero-shot (E2)", 0.0, "Zero-shot on IEEE 30 (E2)"),
             ("Fine-tune 1 %", "fine-tune", 0.01, "Fine-tune 1 % (E3)"),
             ("Fine-tune 5 %", "fine-tune", 0.05, "Fine-tune 5 % (E3)"),
             ("Fine-tune 100 %", "fine-tune", 1.0, "Fine-tune 100 % (E3)"),
             ("Scratch 5 %", "scratch", 0.05, "From scratch 5 % (E4)"),
             ("Scratch 100 %", "scratch", 1.0, "From scratch 100 % (E4)"),
             ("DC-OPF baseline", None, None, "DC-OPF baseline (case30)")]
    rows = []
    for lab, m, f, grp in items:
        d = df[(df.method == m) & np.isclose(df.fraction, f)] if m else df.iloc[:0]
        p = pf[pf.group == grp]
        if not len(d) and not len(p):
            continue
        cost_before = ms(d.cost_gap_abs_pct, 2) if len(d) else (
            ms(p.dc_cost_gap_abs_pct, 2) if "dc_cost_gap_abs_pct" in p else "\u2013")
        rows.append([lab,
                     ms(d.branch_viol_pct, 2) if len(d) else "\u2013",
                     ms(d.samples_with_branch_viol_pct, 1) if len(d) else "\u2013",
                     ms(d.pb_p_max_mw, 1) if len(d) else "\u2013",
                     cost_before,
                     ms(p.pf_converged_pct, 1) if len(p) else "\u2013",
                     ms(p.feasible_pct, 1) if len(p) else "\u2013",
                     ms(p.cost_gap_pf_abs_pct, 2) if len(p) else "\u2013"])
    cols = ["Model", "Overloaded branches (%)", "Hours with overload (%)", "Max |\u0394P| (MW)",
            "|Cost gap| (%)", "PF converged (%)", "Feasible after PF (%)", "|Cost gap| after PF (%)"]
    n_h = int(pf[pf.run == "oracle"].n_hours.iloc[0]) if "n_hours" in pf and (pf.run == "oracle").any() else None
    add_table(doc, "11", "Constraint satisfaction and optimality on the IEEE 30 test set, before and after "
              "power-flow (PF) post-processing.", pd.DataFrame(rows, columns=cols),
              "Columns 2\u20135: raw model output. Columns 6\u20138: predicted Pg (non-slack) and generator voltage "
              "set-points passed to an AC power flow; feasible = no voltage, thermal, generator-Q or slack limit "
              "violated. DC-OPF: pandapower rundcopp dispatch passed through the same AC power flow."
              + (f" PF evaluation on {n_h} test hours." if n_h else ""), size=8)


# ------------------------------------------------------------------ Table 7 ablation
def table07(doc):
    df = runs()
    df = df[(df.case == "case30") & df.seed.isin([0, 1, 2])]
    st = stats_df()
    variants = [("", "Full model"), ("nope", "without ER positional encoding"),
                ("noattn", "without global attention"), ("nophys", "without power-balance loss (\u03bb_pb = 0)"),
                ("augnone", "source data without augmentation"), ("augL", "augmentation L only"),
                ("augLP", "augmentation L + P")]
    settings = [("zero-shot (E2)", 0.0, "zero-shot"), ("fine-tune", 0.01, "FT 1%"), ("fine-tune", 0.05, "FT 5%")]
    rows = []
    for tag, lab in variants:
        cells = [lab]
        for m, f, short in settings:
            method = m + (f" [{tag}]" if tag else "")
            d = df[(df.method == method) & np.isclose(df.fraction, f)]
            for k in ["pg_mae_mw", "pb_p_mean_mw"]:
                c = ms(d[k], DEC[k])
                if tag and len(d):
                    c += sigmark(st, "E5 ablation", f"full ({short})", f"{tag} ({short})", k, raw=True)
                cells.append(c)
        if any(c != "\u2013" for c in cells[1:]):
            rows.append(cells)
    cols = ["Variant", "Zero-shot Pg MAE (MW)", "Zero-shot mean |\u0394P| (MW)", "FT 1 % Pg MAE (MW)",
            "FT 1 % mean |\u0394P| (MW)", "FT 5 % Pg MAE (MW)", "FT 5 % mean |\u0394P| (MW)"]
    add_table(doc, "10", "Ablation study (E5) on the IEEE 30 test set, seeds 0\u20132, mean \u00b1 std.",
              pd.DataFrame(rows, columns=cols),
              "Paired t-test against the full model (same seeds and data subsets): \u2020 variant significantly worse, "
              "\u2021 significantly better (Holm-corrected within the ablation family, p < 0.05); \u00b0 p < 0.05 "
              "before correction only. Each variant is pre-trained on IEEE 14 and fine-tuned like the full model. "
              "With three seeds the tests have low power.", size=8)


# ------------------------------------------------------------------ Table 8 runtime
def table08(doc):
    rt = pd.read_csv(RES / "runtime_benchmark.csv", index_col=0)
    pf = pd.read_csv(RES / "pf_postprocess_runs.csv") if (RES / "pf_postprocess_runs.csv").exists() else None

    def g(case, key):
        return float(rt.loc[key, case]) if key in rt.index and case in rt.columns else np.nan

    rows = []
    defs = [("AC-OPF solver (pandapower PIPS), median", "solver_ms_median"),
            ("AC-OPF solver, worst case", "solver_ms_worst"),
            ("GNN, single sample, CPU", "gnn_cpu_single_ms_median"),
            ("GNN, single sample, GPU", "gnn_gpu_single_ms_median"),
            ("GNN, batched on GPU (per sample)", "gnn_gpu_batched_ms_per_sample")]
    for lab, key in defs:
        r = [lab]
        for case in ["case14", "case30"]:
            v, base = g(case, key), g(case, "solver_ms_median")
            r.append("\u2013" if not np.isfinite(v) else
                     (f"{v:.3g}" if key.startswith("solver") else f"{v:.3g} ({base / v:,.0f}\u00d7)"))
        rows.append(r)
    if pf is not None:
        for lab, sel in [("DC-OPF (pandapower)", "dc"), ("GNN (CPU) + AC power flow", "pf")]:
            r = [lab]
            for case in ["case14", "case30"]:
                base = g(case, "solver_ms_median")
                if sel == "dc":
                    d = pf[(pf.case == case) & (pf.run == "dcopf")]
                    v = d.dc_dc_ms.mean() if len(d) and "dc_dc_ms" in d else np.nan
                else:
                    d = pf[(pf.case == case) & ~pf.run.isin(["oracle", "dcopf"])]
                    v = g(case, "gnn_cpu_single_ms_median") + d.pf_ms.mean() if len(d) else np.nan
                r.append(f"{v:.3g} ({base / v:,.0f}\u00d7)" if np.isfinite(v) else "\u2013")
            rows.append(r)
    add_table(doc, "12", "Computation time per operating point in ms (speed-up relative to the median AC-OPF "
              "solver time).", pd.DataFrame(rows, columns=["Method", "IEEE 14", "IEEE 30"]),
              "Hardware: see Supplementary Table S3. Model times are forward passes with the input graph already prepared "
              "in memory (medians over 200 test hours); for a single sample the GPU is slower than the CPU because "
              "kernel-launch overhead dominates, and batching removes this overhead.")


# ------------------------------------------------------------------ Table 10 baselines
def table10(doc):
    df = runs()
    df = df[(df.case == "case30") & df.seed.isin([0, 1, 2])]
    st = stats_df()
    models = [("", "Proposed heterogeneous GNN", "src14_s0"), ("gcn", "Homogeneous GCN", "src14_s0_gcn"),
              ("gat", "Homogeneous GAT", "src14_s0_gat"), ("mlp", "MLP (fixed topology)", "scratch30_f1.0_s0_mlp")]
    params = {tag: n_params(run) for tag, _, run in models}
    settings = [("zero-shot (E2)", 0.0, "Zero-shot (no IEEE 30 data)"), ("fine-tune", 0.05, "Fine-tune, 5 %"),
                ("fine-tune", 1.0, "Fine-tune, 100 %"), ("scratch", 0.05, "From scratch, 5 %"),
                ("scratch", 1.0, "From scratch, 100 %")]
    keys = ["pg_mae_mw", "va_mae_deg", "pb_p_mean_mw", "cost_gap_abs_pct"]
    rows, groups = [], []
    for m, f, slab in settings:
        block = []
        for tag, mlab, _ in models:
            method = m + (f" [{tag}]" if tag else "")
            d = df[(df.method == method) & np.isclose(df.fraction, f)]
            if not len(d):
                continue
            cells = [mlab, params[tag]]
            for k in keys:
                c = ms(d[k], DEC[k])
                if tag:
                    c += sigmark(st, "baselines", f"proposed ({m} {f:.0%})", f"{tag} ({m} {f:.0%})", k)
                cells.append(c)
            block.append(cells)
        if block:
            groups.append(len(rows)); rows.append([slab] + [""] * (len(keys) + 1))
            rows += block
    cols = ["Model", "Parameters"] + [COLS[k] for k in keys]
    add_table(doc, "8", "Comparison with baseline architectures on the IEEE 30 test set (seeds 0\u20132).",
              pd.DataFrame(rows, columns=cols),
              "Baseline widths were chosen to match the parameter count of the proposed model. "
              "\u2020 proposed model significantly better than this baseline in the same setting, \u2021 baseline "
              "significantly better (paired t-test, Holm-corrected, p < 0.05). The MLP maps the load vector of one fixed grid to the solution and "
              "cannot be transferred between grids, so it only appears in the from-scratch settings.",
              size=8, group_rows=groups)


def table10b(doc):
    df = runs()
    df = df[df.case == "case30"]
    st = stats_df()
    rows = []
    for f in FRACS:
        a = df[(df.method == "fine-tune") & np.isclose(df.fraction, f) & df.seed.isin([0, 1, 2])]
        b = df[(df.method == "scratch [mlp]") & np.isclose(df.fraction, f)]
        if not len(b):
            continue
        r = [pct(f), f"{hours(f):,}"]
        for k in ["pg_mae_mw", "pb_p_mean_mw", "cost_gap_abs_pct"]:
            r += [ms(a[k], DEC[k]), ms(b[k], DEC[k])]
            d = st[(st.family == "FT vs MLP") & (st.A == f"proposed fine-tune {f:.0%}") & (st.metric == k)]
            if len(d) and d.p_holm.iloc[0] < 0.05:
                r[-2 if d.mean_A.iloc[0] < d.mean_B.iloc[0] else -1] += "*"
        rows.append(r)
    cols = ["IEEE 30 data", "Hours", "Pg MAE, GNN FT", "Pg MAE, MLP", "Mean |\u0394P|, GNN FT",
            "Mean |\u0394P|, MLP", "|Cost gap|, GNN FT (%)", "|Cost gap|, MLP (%)"]
    add_table(doc, "9", "Transferred GNN (fine-tuned) versus a grid-specific MLP trained from scratch on the "
              "same IEEE 30 data (intact topology, seeds 0\u20132).", pd.DataFrame(rows, columns=cols),
              "* the better of the two is significantly better (paired t-test, Holm-corrected, p < 0.05). "
              "MW for Pg MAE and mean |\u0394P|.", size=8)


# ------------------------------------------------------------------ Table 11 N-1
def table11(doc):
    keys = ["pg_mae_mw", "va_mae_deg", "pb_p_mean_mw", "cost_gap_abs_pct"]
    recs = []
    n1r = pd.read_csv(RES / "n1_runs.csv")
    n1r = n1r[np.isclose(n1r.fraction, 0.25)]
    for model, method, lab in [("proposed", "fine-tune", "GNN fine-tune"), ("proposed", "scratch", "GNN from scratch"),
                               ("MLP", "scratch", "MLP, topology-blind")]:
        for _, x in n1r[(n1r.model == model) & (n1r.method == method)].iterrows():
            recs.append(dict(scen="(a) Trained on intact topology only; tested on all N-1 test hours", model=lab,
                             **{k: x[f"n1_{k}"] for k in keys}))
    for tag, kind, lab in [("n1", "ft30", "GNN fine-tune"), ("n1", "scratch30", "GNN from scratch"),
                           ("n1mlp", "scratch30", "MLP, topology-blind"), ("n1mlptopo", "scratch30", "MLP + outage one-hot")]:
        for f in sorted(RES.glob(f"{kind}_f0.25_s?_{tag}/metrics.json")):
            for m in json.load(open(f)):
                if m["case"] == "case30_n1":
                    recs.append(dict(scen="(b) Trained with N-1 data; tested on all N-1 test hours", model=lab,
                                     **{k: m[k] for k in keys}))
    p = RES / "unseen_runs.csv"
    if p.exists():
        un = pd.read_csv(p)
        for sub, scen in [("seen", "(c) Trained with N-1 data, 8 branches held out; tested on seen outages"),
                          ("unseen", "(d) Same models; tested on the 8 held-out (unseen) outages")]:
            for src, lab in [("GNN fine-tune, outages held out", "GNN fine-tune"),
                             ("GNN from scratch, outages held out", "GNN from scratch"),
                             ("MLP one-hot, outages held out (raw)", "MLP + outage one-hot"),
                             ("MLP one-hot, outages held out (unseen -> intact)", "MLP + one-hot, unseen fed as intact"),
                             ("REF MLP topology-blind, no N-1 data (floor)", "MLP, topology-blind, no N-1 data")]:
                if "intact" in src and sub == "seen":
                    continue                       # identical to the raw MLP on seen outages
                for _, x in un[(un.subset == sub) & (un.model == src)].iterrows():
                    recs.append(dict(scen=scen, model=lab, **{k: x[k] for k in keys}))
    d = pd.DataFrame(recs)
    rows, groups = [], []
    for scen in d.scen.unique():
        groups.append(len(rows)); rows.append([scen] + [""] * (len(keys) + 1))
        s = d[d.scen == scen]
        for lab in s.model.unique():
            x = s[s.model == lab]
            rows.append([lab, f"{len(x)}"] + [ms(x[k], DEC[k]) for k in keys])
    cols = ["Model", "Seeds"] + [COLS[k] for k in keys]
    add_table(doc, "13", "Robustness to single-branch outages (IEEE 30 N-1 test hours, 25 % of the target "
              "training hours).", pd.DataFrame(rows, columns=cols),
              "The topology-blind MLP ignores the outage and is the reference floor; a model that uses the "
              "topology is useful only if it beats this floor.", size=8, group_rows=groups)


# ------------------------------------------------------------------ Appendix tables
def tableA1(doc):
    lt = pd.read_csv(RES / "lambda_tuning.csv")
    rows = [[f"{x.lambda_pb:g}", f"{x.val_pg_mae_mw:.2f}", f"{x.val_va_mae_deg:.3f}", f"{x.val_pb_p_mean_mw:.2f}",
             f"{x.val_pb_p_max_mw:.1f}", f"{x.val_cost_gap_abs_pct:.2f}", "\u2713" if x.selected else ""]
            for x in lt.itertuples()]
    add_table(doc, "S1", "Tuning of the power-balance weight \u03bb_pb on the IEEE 14 validation set (Sep\u2013Oct).",
              pd.DataFrame(rows, columns=["\u03bb_pb", "Pg MAE (MW)", "Va MAE (\u00b0)", "Mean |\u0394P| (MW)",
                                          "Max |\u0394P| (MW)", "|Cost gap| (%)", "Selected"]),
              "Selection rule: lowest mean |\u0394P| among the values whose Pg MAE is within 10 % of the best.")


def tableA2(doc):
    ci = RES / "tables" / "table_S2_stats_ci.csv"            # written by stats_ci.py (revision)
    if ci.exists():
        add_table(doc, "S2", "Paired significance tests on the IEEE 30 test set with 95 % confidence intervals and "
                  "effect sizes.", pd.read_csv(ci, dtype=str).fillna(""),
                  "Two-sided paired t-test over seeds (same seed = same data subset and source model); Holm\u2013"
                  "Bonferroni correction within each family over all six tested metrics (three shown). \u0394 = mean A / "
                  "mean B \u2212 1 (negative = A better); 95 % CI of the paired difference relative to mean B (Student t); "
                  "d_z = mean paired difference / its standard deviation. \u2018uncorr. only\u2019 = p < 0.05 before "
                  "correction only.", size=6.5)
        return
    st = pd.read_csv(RES / "stats_tests.csv")
    st = st[st.metric.isin(["pg_mae_mw", "pb_p_mean_mw", "cost_gap_abs_pct"])]
    name = {"pg_mae_mw": "Pg MAE", "pb_p_mean_mw": "mean |\u0394P|", "cost_gap_abs_pct": "|cost gap|"}
    rows = [[x.family, x.A, x.B, name[x.metric], f"{x.n_seeds}", f"{x.mean_A:.3g}", f"{x.mean_B:.3g}",
             f"{x.rel_diff_pct:+.1f}", x.A_better_in_seeds, f"{x.p_value:.3g}", f"{x.p_holm:.3g}",
             "yes" if x.p_holm < 0.05 else ""] for x in st.itertuples()]
    add_table(doc, "S2", "Paired significance tests (IEEE 30 test set).",
              pd.DataFrame(rows, columns=["Family", "A", "B", "Metric", "n", "Mean A", "Mean B", "\u0394 (%)",
                                          "A better", "p", "p (Holm)", "Sig."]),
              "Two-sided paired t-test over seeds (same seed = same data subset and source model); Holm\u2013Bonferroni "
              "correction within each family over all six tested metrics (only three shown). \u0394 = mean A / mean B "
              "\u2212 1; negative = A better. With 3\u20135 seeds the tests have low power.", size=7)


def tableA3(doc):
    def v(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return "\u2013"
    gpu, cuda = "\u2013", "\u2013"
    try:
        import torch
        cuda = torch.version.cuda or "\u2013"
        if torch.cuda.is_available():
            gpu = f"{torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GB)"
    except Exception:
        pass
    cpu = platform.processor() or "\u2013"
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        pass
    rows = [("Software", "Python", platform.python_version()), ("", "PyTorch / CUDA", f"{v('torch')} / {cuda}"),
            ("", "PyTorch Geometric", v("torch_geometric")), ("", "pandapower (AC-OPF: PIPS)", v("pandapower")),
            ("", "NumPy / pandas / SciPy", f"{v('numpy')} / {v('pandas')} / {v('scipy')}"),
            ("", "Operating system", f"{platform.system()} {platform.release()}"),
            ("Hardware", "GPU", gpu), ("", "CPU", cpu),
            ("Seeds", "Model training and fine-tuning subsets", "0\u20134"),
            ("", "IEEE 14 augmentation (L, LP, LPT)", "2026, 2027, 2028"),
            ("", "N-1 test / train / validation sets", "2027 / 2028 / 2029"),
            ("", "Held-out outages (unseen experiment)", "8 branches, seed 7"),
            ("Data", "Time series", "IEEE 14 / IEEE 30 hourly bus loads, year 2007"),
            ("", "Grids", "PGLib-OPF v23.07 case14; MATPOWER case30")]
    add_table(doc, "S3", "Reproducibility information.", pd.DataFrame(rows, columns=["Item", "Detail", "Value"]),
              "Times in Table 12 of the main text were measured on this hardware.")


def tableS4(doc):
    """Violated limit types in the infeasible hours, with the tolerances of Section 5.9 and with zero tolerances.
    Built from results\\pf_breakdown_runs.csv (falls back to tables\\table_S4_breakdown.csv)."""
    src = RES / "pf_breakdown_runs.csv"
    notes = ("Written by pf_breakdown_sensitivity.py (single slack). Infeasible hours in % of the 1,464 test hours; limit "
             "types in % of the infeasible hours (an hour can violate several types, so the shares can add up to more "
             "than 100 %). Worst loading = highest branch loading in the infeasible hours, averaged over seeds. "
             "Generator reactive limits and the active limit of the slack generator were never violated. Hours that "
             "are infeasible only without tolerances violate a limit by no more than 10\u207b\u00b3 p.u., 0.5 % loading "
             "or 0.1 Mvar.")
    if not src.exists():
        df = pd.read_csv(RES / "tables" / "table_S4_breakdown.csv", dtype=str).fillna("\u2013")
        add_table(doc, "S4", "Operating limits violated in the infeasible test hours after AC power-flow correction "
                  "(IEEE 30 test set, single slack; mean \u00b1 std over seeds).", df, notes, size=7)
        return
    runs = pd.read_csv(src)
    runs = runs[~runs.group.str.startswith("AC-OPF labels")]
    kinds = [("vm", "Voltage"), ("branch", "Branch thermal"), ("slack_q", "Slack Q")]
    cols = ["Model", "Seeds", "Infeasible hours (%)"] + [f"{n} (% of infeasible)" for _, n in kinds] + \
           ["Worst loading (%)"]
    rows, groups = [], []
    for var, lab in [("single", "With the tolerances of Section 5.9 (as in the main text)"),
                     ("zero_tol", "Without tolerances")]:
        d = runs[runs.variant == var]
        if not len(d):
            continue
        groups.append(len(rows)); rows.append([lab] + [""] * (len(cols) - 1))
        for grp, g in d.groupby("group", sort=False):
            one = lambda x: f"{float(x.iloc[0]):.1f}" if len(g) == 1 else ms(x)   # DC-OPF: single run
            rows.append([grp, str(len(g)), one(g.infeasible_pct)] +
                        [one(g[f"{k}_in_infeasible_pct"]) for k, _ in kinds] +
                        [one(g.worst_loading_in_infeasible_pct)])
    add_table(doc, "S4", "Operating limits violated in the infeasible test hours after AC power-flow correction, "
              "with and without tolerances (IEEE 30 test set, single slack; mean \u00b1 std over seeds).",
              pd.DataFrame(rows, columns=cols), notes, size=7, group_rows=groups)


def tableS5(doc):
    add_table(doc, "S5", "Sensitivity of the feasibility rate after AC power-flow correction to the slack treatment "
              "and the tolerances (IEEE 30 test set; mean \u00b1 std over seeds).",
              pd.read_csv(RES / "tables" / "table_S5_sensitivity.csv", dtype=str).fillna("\u2013"),
              "single = slack generator absorbs the imbalance (paper); distributed = all dispatchable generators share "
              "the imbalance in proportion to their active-power range, with generator P limits checked; zero_tol = "
              "single slack with zero tolerances. single and zero_tol evaluate the same power-flow solution, so an hour that is "
              "feasible under single but infeasible under zero_tol violates a limit only within the tolerances of Section 5.9 "
              "(10\u207b\u00b3 p.u., 0.5 % loading, 0.1 MW/Mvar). The DC-OPF dispatch is processed identically.", size=7.5)


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for s in [sec]:
        s.left_margin = s.right_margin = Cm(1.8)
        s.top_margin = s.bottom_margin = Cm(2.0)
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
    style.font.size = Pt(10)
    para(doc, "Main-text tables \u2013 Cross-grid transfer learning for AC-OPF", size=14, bold=True)
    para(doc, "Generated by scripts/30_make_tables.py. Tables 2, 3, 5 and 7\u201313 are the main-text tables of the "
         "revised manuscript; Supplementary Tables S1\u2013S5 go into the Supplementary Information.",
         size=9, italic=True, space_after=10)
    extra = [f for f, csv in [(tableS4, "table_S4_breakdown.csv"), (tableS5, "table_S5_sensitivity.csv")]
             if (OUT / csv).exists()]
    for f in [table02, table03, table04, table05, table10, table10b, table07, table06, table08, table11,
              tableA1, tableA2, tableA3] + extra:
        if f is tableA1:
            para(doc, "Supplementary Information \u2013 tables", size=14, bold=True)
        if f is tableA2:
            landscape(doc)
        if f is tableA3:
            portrait(doc)
        try:
            f(doc)
        except Exception as e:
            print(f"  {f.__name__}: SKIPPED ({type(e).__name__}: {e})")
            traceback.print_exc(limit=1)
    out = OUT / "paper_tables.docx"
    doc.save(out)
    print(f"\nSaved {out}\n      and one CSV per table in {OUT}")
