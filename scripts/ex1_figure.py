#!/usr/bin/env python3
"""EX1 figure: segment F1 and absolute segment errors of A08+B3 on each stream bank, from the public benchmark file."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

CJK_FONT = '/System/Library/Fonts/STHeiti Medium.ttc'
FOLDER =Path(__file__).resolve().parents[1] / 'docs/results/2026-11-03_exam_level'
BANKS = (('constructed', '构造库'), ('exam', '真实顺序'), ('shuffled', '同检查打乱'), ('rewired', '跨 session 重连'))
HALVES = (('selection', 'selection 半'), ('calibration', 'calibration 半'))
SERIES, PRIMARY, SECONDARY, GRID, SURFACE = '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
MARKER_POINTS = 49
MEAN_HALF_WIDTH = .24
OFFSETS = {0: -.09, 1: 0., 2: .09}


def value(cell, measure):
    return cell['segment_f1'] if measure == 'f1' else cell['T'] + cell['P'] - 2 * cell['M']


def draw(axis, cells, half, measure):
    for x, (bank, _) in enumerate(BANKS):
        rows = [c for c in cells if c['half'] == half and c['bank'] == bank and c['decoder'] == 'b3']
        values = [value(c, measure) for c in rows]
        mean = sum(values) / len(values)
        axis.scatter([x + OFFSETS[c['seed']] for c in rows], values, s=MARKER_POINTS, color=SERIES, edgecolor=SURFACE, linewidth=1.2, zorder=3)
        axis.plot([x - MEAN_HALF_WIDTH, x + MEAN_HALF_WIDTH], [mean, mean], color=PRIMARY, linewidth=2, solid_capstyle='round', zorder=2)
        label = f'{mean:.3f}' if measure == 'f1' else f'{mean:.1f}'
        axis.annotate(label, (x + MEAN_HALF_WIDTH, mean), xytext=(5, 0), textcoords='offset points', va='center', fontsize=10, color=PRIMARY)


def style(axis, measure, show_x):
    axis.set_facecolor(SURFACE)
    axis.grid(axis='y', color=GRID, linewidth=1)
    axis.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        axis.spines[side].set_visible(False)
    axis.spines['bottom'].set_color(SECONDARY)
    axis.tick_params(axis='y', length=0, colors=SECONDARY)
    axis.tick_params(axis='x', length=0, colors=PRIMARY)
    axis.set_xlim(-.5, len(BANKS) - .35)
    axis.set_xticks(range(len(BANKS)))
    axis.set_xticklabels([name for _, name in BANKS] if show_x else [''] * len(BANKS), fontsize=10)
    if measure == 'f1':
        axis.set_ylim(.90, 1.)
    else:
        axis.set_ylim(0, 50)


def main():
    font_manager.fontManager.addfont(CJK_FONT)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=CJK_FONT).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    cells = json.loads((FOLDER / 'benchmark.json').read_text())['cells']
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 6.6), sharey='row', facecolor=SURFACE)
    for column, (half, name) in enumerate(HALVES):
        for row, measure in enumerate(('f1', 'errors')):
            draw(axes[row][column], cells, half, measure)
            style(axes[row][column], measure, show_x=row == 1)
        axes[0][column].set_title(name, fontsize=12, color=PRIMARY, loc='left')
    axes[0][0].set_ylabel('片段 F1@50', fontsize=11, color=SECONDARY)
    axes[1][0].set_ylabel('片段错误数（漏 + 多）', fontsize=11, color=SECONDARY)
    figure.text(.012, .012, '每个点是一个 seed，黑线是三个 seed 的均值。系统是 A08 加 B3，都来自反复使用的开发集。',
                fontsize=10, color=SECONDARY)
    figure.tight_layout(rect=(0, .045, 1, 1))
    figure.savefig(FOLDER / 'segment_f1_and_errors.png', dpi=160, facecolor=SURFACE)


if __name__ == '__main__':
    main()
