#!/usr/bin/env python3
"""EX2 figure: share of recordings each open-set score accepts, at the threshold keeping 95% of development clips."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

CJK_FONT = '/System/Library/Fonts/STHeiti Medium.ttc'
FOLDER = Path(__file__).resolve().parents[1] / 'docs/results/2026-11-04_open_routing'
GROUPS = (('dev', 'EV9V\n开发集', True), ('out_of_scope', 'PMPALA', False), ('dynamic', 'Dynamic', True), ('lvh', 'LVH', True), ('camus_4ch', 'CAMUS\n4CH', True), ('camus_2ch', 'CAMUS\n2CH', False))
METHODS = (('msp', '最大 softmax'), ('energy', '能量'), ('knn', 'kNN 特征距离'), ('mahalanobis', '马氏距离'))
INSIDE, OUTSIDE, PRIMARY, SECONDARY, GRID, SURFACE = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
MARKER_POINTS = 49
OFFSETS = {0: -.13, 1: 0., 2: .13}
MEAN_HALF_WIDTH = .28


def value(row, key):
    if key == 'dev':
        return .95
    return row['accept'][key]


def draw(axis, rows):
    for x, (key, _, inside) in enumerate(GROUPS):
        colour = INSIDE if inside else OUTSIDE
        values = [value(r, key) for r in rows]
        axis.scatter([x + OFFSETS[r['seed']] for r in rows], values, s=MARKER_POINTS, color=colour, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        mean = sum(values) / len(values)
        axis.plot([x - MEAN_HALF_WIDTH, x + MEAN_HALF_WIDTH], [mean, mean], color=PRIMARY, linewidth=2, solid_capstyle='round', zorder=2)
    axis.set_facecolor(SURFACE)
    axis.grid(axis='y', color=GRID, linewidth=1)
    axis.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        axis.spines[side].set_visible(False)
    axis.spines['bottom'].set_color(SECONDARY)
    axis.tick_params(axis='y', length=0, colors=SECONDARY)
    axis.tick_params(axis='x', length=0, colors=PRIMARY, labelsize=10)
    axis.set_ylim(-.04, 1.06)
    axis.set_yticks([0, .25, .5, .75, 1.])
    axis.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    axis.set_xticks(range(len(GROUPS)))
    axis.set_xticklabels([g[1] for g in GROUPS])


def main():
    font_manager.fontManager.addfont(CJK_FONT)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=CJK_FONT).get_name()
    rows = json.loads((FOLDER / 'x3_open_set.json').read_text())['rows']
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), sharey=True, facecolor=SURFACE)
    for axis, (method, name) in zip(axes.ravel(), METHODS):
        draw(axis, [r for r in rows if r['method'] == method])
        axis.set_title(name, fontsize=12, color=PRIMARY, loc='left')
    axes[0][0].set_ylabel('被放行的录像比例', fontsize=11, color=SECONDARY)
    axes[1][0].set_ylabel('被放行的录像比例', fontsize=11, color=SECONDARY)
    handles = [Line2D([], [], marker='o', linestyle='', color=INSIDE, markersize=8, label='属于五类，应该放行'),
               Line2D([], [], marker='o', linestyle='', color=OUTSIDE, markersize=8, label='不属于五类，应该拒绝'),
               Line2D([], [], color=PRIMARY, linewidth=2, label='三个 seed 的均值')]
    figure.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=10, labelcolor=SECONDARY)
    figure.tight_layout(rect=(0, .055, 1, 1))
    figure.savefig(FOLDER / 'open_set_acceptance.png', dpi=160, facecolor=SURFACE)


if __name__ == '__main__':
    main()
