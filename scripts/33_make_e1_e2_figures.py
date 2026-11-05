"""
33_make_e1_e2_figures.py
Figures 6 (E1) and 7 (E2) of the paper from the saved test-set predictions of the source model
(seed 0), written by 29_predict_test.py:
  Fig. 6  predicted versus AC-OPF values on the IEEE 14-bus test months (V, theta, Pg, Qg)
  Fig. 7  zero-shot transfer to the IEEE 30-bus system: total generation over one test week and
          mean output of each generator over the test months
Output: results\\figures\\paper\\Fig6*.png, Fig7*.png (600 dpi)

    python scripts\\33_make_e1_e2_figures.py
"""

import os
from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, MultipleLocator

from figure_style import plt, new_axes, framed_legend, callout, bold_ticks, MODEL, PAL, LW, MS, FS

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / 'results' / 'predictions'
OUT = str(ROOT / 'results' / 'figures' / 'paper') + os.sep
os.makedirs(OUT, exist_ok=True)
BASE = 100.0
ZS = PAL['teal']                       # zero-shot source model (new colour, not used by other models)


def load(c):
    return np.load(PRED / f'src14_s0_{c}.npz')


def headroom(ax, leg, top_data):
    ax.figure.canvas.draw()
    lb = leg.get_window_extent().transformed(ax.transAxes.inverted()).y0 - 0.03
    lo, hi = ax.get_ylim()
    if (top_data - lo) / (hi - lo) > lb:
        ax.set_ylim(lo, lo + (top_data - lo) / lb)


# ------------------------------------------------------------------ Fig. 6: E1 parity, IEEE 14-bus test months
z = load('case14')
disp = z['gen_pmax'] > z['gen_pmin']
PANELS = [
    ('a', 'vm', '(a) Voltage magnitude', 'p.u.', 1.0, None, '%.1e p.u.'),
    ('b', 'va', '(b) Voltage angle', 'deg', 180 / np.pi, None, '%.4f°'),
    ('c', 'pg', '(c) Active output, dispatchable generators', 'MW', BASE, disp, '%.3f MW'),
    ('d', 'qg', '(d) Reactive output', 'Mvar', BASE, None, '%.3f Mvar'),
]
for letter, k, title, unit, scale, mask, fmt in PANELS:
    t, p = z[f'true_{k}'] * scale, z[f'pred_{k}'] * scale
    if mask is not None:
        t, p = t[:, mask], p[:, mask]
    mae = np.abs(p - t).mean()                      # same definition as the MAE in the text (all entries)
    keep = t.std(axis=0) > 1e-9                     # constant columns (slack-bus angle) are not plotted
    t, p = t[:, keep].ravel(), p[:, keep].ravel()
    lo, hi = min(t.min(), p.min()), max(t.max(), p.max())
    pad = 0.05 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    fig, ax = new_axes()
    ax.plot([lo, hi], [lo, hi], ls='--', color=MODEL['ref'], lw=2.2, zorder=2)
    ax.plot(t, p, 'o', ms=3.5, color=MODEL['ft'], alpha=0.35, mew=0, zorder=3, rasterized=True)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.xaxis.set_major_locator(MaxNLocator(5)); ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_xlabel(f'AC-OPF value ({unit})'); ax.set_ylabel(f'Predicted value ({unit})')
    ax.set_title(title, loc='left')
    handles = [Line2D([], [], ls='none', marker='o', ms=MS, color=MODEL['ft'], mec='black', mew=1.2),
               Line2D([], [], color=MODEL['ref'], lw=2.2, ls='--')]
    framed_legend(ax, handles=handles, labels=['Source model (seed 0)', 'Perfect agreement'],
                  loc='upper left', bbox_to_anchor=(0.015, 0.985))
    callout(ax, 0.97, 0.05, 'MAE ' + (fmt % mae), ha='right', va='bottom')
    bold_ticks(ax)
    name = f'Fig6{letter}_e1_parity_{k}.png'
    fig.savefig(OUT + name); plt.close(fig)
    print(name, 'MAE', mae, 'n', t.size)

# ------------------------------------------------------------------ Fig. 7: E2 zero-shot dispatch, IEEE 30-bus
z = load('case30')
ts = list(z['timestamps'])
i0 = ts.index('2007-12-03 00:00:00')
week = slice(i0, i0 + 168)
tot_t = z['true_pg'].sum(1) * BASE
tot_p = z['pred_pg'].sum(1) * BASE
h = np.arange(168)
DAYS = ['Mon 03', 'Tue 04', 'Wed 05', 'Thu 06', 'Fri 07', 'Sat 08', 'Sun 09', 'Mon 10']

fig, ax = new_axes()
ax.plot(h, tot_t[week], '-', color=MODEL['ref'], lw=LW + 1.5, label='AC-OPF', zorder=3, solid_capstyle='butt')
ax.plot(h, tot_p[week], color=ZS, lw=LW, ls=(0, (3, 2)), label='Zero-shot prediction', zorder=4)
ax.set_xlim(-8.5, 176.5)
ax.set_xticks(np.arange(0, 169, 24)); ax.set_xticklabels(DAYS)
ax.set_xlabel('Date (December 2007)')
ax.set_ylabel('Total active generation (MW)')
ax.set_title('(a) Total active generation over one test week', loc='left')
ax.set_ylim(0, None)
ax.yaxis.set_major_locator(MultipleLocator(50))
leg = framed_legend(ax, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.985), columnspacing=1.2, handletextpad=0.5)
headroom(ax, leg, max(tot_t[week].max(), tot_p[week].max()) + 4)
bold_ticks(ax)
fig.savefig(OUT + 'Fig7a_e2_total_generation_week.png'); plt.close(fig)
print('week mean total: AC-OPF %.1f, zero-shot %.1f MW' % (tot_t[week].mean(), tot_p[week].mean()))
print('test months mean total: AC-OPF %.1f, zero-shot %.1f MW' % (tot_t.mean(), tot_p.mean()))

buses = z['gen_bus'] + 1
order = np.argsort(buses)
mt, st = (z['true_pg'] * BASE).mean(0)[order], (z['true_pg'] * BASE).std(0)[order]
mp, sp = (z['pred_pg'] * BASE).mean(0)[order], (z['pred_pg'] * BASE).std(0)[order]
x = np.arange(len(order)); w = 0.38
fig, ax = new_axes()
ax.grid(False, axis='x')
ax.bar(x - w / 2, mt, w, color=MODEL['ref'], edgecolor='black', linewidth=1.6, label='AC-OPF', zorder=2)
ax.bar(x + w / 2, mp, w, color=ZS, edgecolor='black', linewidth=1.6, label='Zero-shot prediction', zorder=2)
ax.errorbar(x - w / 2, mt, yerr=st, fmt='none', ecolor='black', elinewidth=2.2, capsize=5, capthick=2.2, zorder=3)
ax.errorbar(x + w / 2, mp, yerr=sp, fmt='none', ecolor='black', elinewidth=2.2, capsize=5, capthick=2.2, zorder=3)
ax.set_xticks(x); ax.set_xticklabels([f'Bus {b}' for b in buses[order]])
ax.set_xlim(-0.6, len(order) - 0.4)
ax.set_xlabel('Generator')
ax.set_ylabel('Mean active output (MW)')
ax.set_title('(b) Mean output of each generator, test months', loc='left')
ax.set_ylim(0, None)
leg = framed_legend(ax, ncol=2, loc='upper center', bbox_to_anchor=(0.5, 0.985), columnspacing=1.2, handletextpad=0.5)
headroom(ax, leg, max((mt + st).max(), (mp + sp).max()) + 2)
bold_ticks(ax)
fig.savefig(OUT + 'Fig7b_e2_generator_output.png'); plt.close(fig)
print('per generator (bus: AC-OPF / zero-shot MW):', [(int(b), round(float(a), 1), round(float(c), 1)) for b, a, c in zip(buses[order], mt, mp)])
