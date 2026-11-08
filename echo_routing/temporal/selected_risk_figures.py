"""A63 Stage-S aggregate presentation only: never compute or change a gate.

The plots preserve recorded failures, zero native hazards and all eight auxiliary
loss series. They cannot establish C0/C1 effectiveness or clinical safety. No
private source/action identifiers are written. True visual QA is a later human
or agent inspection of the actual rendered files, never a renderer flag.
"""
import csv
import hashlib
import json
import math
import textwrap
from numbers import Real
from pathlib import Path

SEEDS = (2026110461, 2026110462)
CLASSES = ('PLAX', 'PSAX', 'A4C', 'A5C', 'SC4C')
# Frozen minimum -> aggregate source, not a fresh choice from observed results.
FLOORS = {
    'D_danger30': (30, 'D', 'dangerous_signatures'),
    'D_sessions10': (10, 'D', 'responsible_sessions'),
    'X_danger30': (30, 'X_only', 'dangerous_signatures'),
    'X_sessions10': (10, 'X_only', 'responsible_sessions'),
    'union60': (60, 'union', 'dangerous_signatures'),
    'union_sessions20': (20, 'union', 'responsible_sessions'),
    'Hpositive40': (40, 'union', 'H_positive_signatures'),
    'Hge8_15': (15, 'union', 'H_ge8_signatures'),
    'pureFP8': (8, 'union', 'pure_FP_signatures'),
    'pureFP_sessions5': (5, 'union', 'pure_FP_sessions'),
    'A4C_A5C15': (15, 'union', 'A4C_A5C_signatures'),
    'A4C_A5C_sessions6': (6, 'union', 'A4C_A5C_sessions'),
    'PLAX_PSAX8': (8, 'union', 'PLAX_PSAX_signatures'),
    'PLAX_PSAX_sessions4': (4, 'union', 'PLAX_PSAX_sessions'),
    'fullrun15': (15, 'union', 'full_run_signatures'),
    'partial15': (15, 'union', 'partial_run_signatures'),
    'D_safe100': (100, 'D', 'safe_useful_signatures'),
    'X_safe100': (100, 'X_only', 'safe_useful_signatures'),
    'D_safeJ30': (30, 'D', 'safe_J_signatures'),
    'X_safeJ30': (30, 'X_only', 'safe_J_signatures'),
}
CONCENTRATION = ('max_p_D', 'max_p_X', 'max_p_mix')
UNION_TYPES = ('H_positive_signatures', 'H_ge8_signatures', 'pure_FP_signatures',
               'A4C_A5C_signatures', 'PLAX_PSAX_signatures', 'full_run_signatures',
               'partial_run_signatures', 'geometry_conflicts')
MEMBER_FIELDS = ('dangerous_signatures', 'responsible_sessions', 'safe_useful_signatures',
                 'safe_J_signatures', 'safe_J_conflicts', 'same_draw_effect_conflicts')
FAMILIES = ('01_coverage_floors', '02_danger_types', '03_safe_class_native', '04_auxiliary_loss')


def _count(value):
    if type(value) is not int or value < 0:
        raise ValueError('public exact nonnegative integer counts required')
    return value


def _real(value):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError('finite public aggregate scalar required')
    return float(value)


def _validated(coverage, curves):
    try:
        if (type(coverage['passed']) is not bool or set(coverage['seeds']) != {str(s) for s in SEEDS} or
                coverage['source_isolation']['passed'] is not True or
                _count(coverage['aux_optimizer_updates']) != 48000 or
                _count(coverage['formal_optimizer_updates']) not in (0, 24000)):
            raise ValueError('complete audited Stage-S aggregates with both seeds required')
        if not coverage['passed'] and coverage['formal_optimizer_updates'] != 0:
            raise ValueError('coverage failure requires zero C0/C1 formal updates')
        seeds = {}
        for seed in SEEDS:
            row = coverage['seeds'][str(seed)]; checks = row['checks']
            if type(row['passed']) is not bool or set(checks) != set(FLOORS) | set(CONCENTRATION):
                raise ValueError('all original23 coverage checks required')
            for key, (minimum, group, field) in FLOORS.items():
                c = checks[key]; value = _count(c['actual'])
                if (set(c) != {'actual', 'minimum', 'passed'} or _count(c['minimum']) != minimum or
                        type(c['passed']) is not bool or c['passed'] != (value >= minimum) or
                        value != _count(row[group][field])):
                    raise ValueError('original count/floor/pass/aggregate binding differs')
            for key in CONCENTRATION:
                c = checks[key]; value = _real(c['actual'])
                if (set(c) != {'actual', 'maximum', 'passed'} or _real(c['maximum']) != .2 or not 0 <= value <= 1 or
                        value != _real(row[key]) or type(c['passed']) is not bool or c['passed'] != (value <= .2+1e-15)):
                    raise ValueError('original concentration/maximum/pass binding differs')
            failed = [k for k, c in checks.items() if not c['passed']]
            if (len(row['failed_checks']) != len(failed) or set(row['failed_checks']) != set(failed) or
                    row['passed'] != (not failed)):
                raise ValueError('recorded failure list/gate differs')
            for group in ('D', 'X_only'):
                for field in MEMBER_FIELDS: _count(row[group][field])
            for field in UNION_TYPES: _count(row['union'][field])
            classes = row['union']['replacement_class_signatures']
            if set(classes) != {str(c) for c in range(5)}:
                raise ValueError('all five replacement classes, including zeros, required')
            for v in classes.values(): _count(v)
            for field in ('native_D_danger_actions', 'native_X_danger_actions'): _count(row[field])
            seeds[seed] = row
        if coverage['passed'] != all(r['passed'] for r in seeds.values()):
            raise ValueError('both-seed top-level coverage verdict differs')
        series = {(s, f, r): [] for s in SEEDS for f in (0, 1) for r in ('utility', 'risk')}
        if not isinstance(curves, list) or not curves:
            raise ValueError('eight complete auxiliary saved-point series required')
        clean_curves = []
        for row in curves:
            seed, fold, role = row['seed'], row['held_fold'], row['role']
            if (type(seed) is not int or type(fold) is not int or (seed, fold, role) not in series or
                    not 1 <= _count(row['update']) <= 6000 or _real(row['loss']) < 0):
                raise ValueError('finite actual saved auxiliary curve point required')
            point = dict(seed=seed, held_fold=fold, role=role, update=row['update'], loss=float(row['loss']))
            series[(seed, fold, role)].append(point); clean_curves.append(point)
        for points in series.values():
            updates = [p['update'] for p in points]
            if not updates or updates != sorted(set(updates)) or updates[-1] != 6000:
                raise ValueError('each auxiliary series requires ordered unique saved cursors including final6000')
        return seeds, series, clean_curves
    except (KeyError, TypeError) as exc:
        raise ValueError('complete public Stage-S aggregate/curve schema required') from exc


def render_coverage(pub, coverage, curves, synthetic_fixture=False):
    """Write four immutable PNG/PDF/CSV triplets from provided aggregates only."""
    if type(synthetic_fixture) is not bool:
        raise ValueError('explicit synthetic-fixture boolean required')
    seeds, series, clean_curves = _validated(coverage, curves)
    pub = Path(pub)
    if not synthetic_fixture and pub.name != '2026-11-05_selected_action_risk':
        raise ValueError('actual presentation restricted to the new A63 public directory')
    if pub.is_symlink():
        raise ValueError('public output must not be a symlink')
    folder = pub/('synthetic_stageS_figures' if synthetic_fixture else 'stageS_figures')
    paths = [folder/(name+suffix) for name in FAMILIES for suffix in ('.png', '.pdf', '.csv')]
    manifest = folder/'render_receipt.json'
    if folder.is_symlink() or any(p.exists() or p.is_symlink() for p in [*paths, manifest]):
        raise FileExistsError('preserve existing renders; use an explicit reviewed new version for repairs')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    folder.mkdir(parents=True, exist_ok=True)
    colors = ('#246393', '#c66c25'); files = []
    common = 'Stage S only: source sessions/signatures are not patients or independent trials.'
    status = 'COVERAGE PASS' if coverage['passed'] else 'COVERAGE FAIL: stop before C0/C1 training'
    if synthetic_fixture: status = 'SYNTHETIC FIXTURE - NO EXPERIMENT RESULT'
    fields = ('family', 'seed', 'metric', 'value', 'minimum', 'maximum', 'unit',
              'held_fold', 'role', 'update', 'synthetic_fixture')

    def record(family, seed, metric, value, **extra):
        value = dict(family=family, seed=seed, metric=metric, value=value,
                     minimum='', maximum='', unit='count', held_fold='', role='', update='',
                     synthetic_fixture=synthetic_fixture)
        value.update(extra)
        return value

    def finish(fig, name, rows, note):
        fig.suptitle(status, fontsize=14, color='#842c2b' if synthetic_fixture or not coverage['passed'] else '#235d42')
        fig.text(.02, .01, textwrap.fill(common, 140)+'\n'+textwrap.fill(note, 140), fontsize=8, va='bottom')
        if synthetic_fixture:
            fig.text(.5, .49, 'SYNTHETIC / NOT ACTUAL', fontsize=29, color='#9a4444',
                     alpha=.22, rotation=25, ha='center', va='center')
        fig.tight_layout(rect=(0, .10, 1, .95))
        try:
            for suffix in ('.png', '.pdf'):
                target = folder/(name+suffix)
                fig.savefig(target, dpi=150, facecolor='white')
                raw = target.read_bytes(); files.append(dict(path=str(target.relative_to(pub)),
                                                            sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
            target = folder/(name+'.csv')
            with target.open('x', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            raw = target.read_bytes(); files.append(dict(path=str(target.relative_to(pub)),
                                                        sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
        finally:
            plt.close(fig)

    def grouped(ax, labels, values, title, ylabel='Distinct signatures / count'):
        x = np.arange(len(labels)); width = .36
        for j, (seed, val) in enumerate(zip(SEEDS, values)):
            bars = ax.bar(x+(j-.5)*width, val, width, label=str(seed), color=colors[j])
            ax.bar_label(bars, fontsize=8, padding=2)
        ax.set_xticks(x, labels, rotation=25, ha='right'); ax.set_title(title, fontsize=11)
        ax.set_ylabel(ylabel); ax.set_ylim(bottom=0, top=max(1, max(max(v) for v in values))*1.40)
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True); ax.legend(fontsize=8)

    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 9}):
        name = FAMILIES[0]; rows = []; fig, axes = plt.subplots(1, 3, figsize=(17, 10), gridspec_kw={'width_ratios': [1.2, 1.2, 1.]})
        for j, seed in enumerate(SEEDS):
            c = seeds[seed]['checks']; labels = list(FLOORS)
            ratios = [c[k]['actual']/c[k]['minimum'] for k in labels]
            axes[j].barh(labels, ratios, color=[colors[j] if c[k]['passed'] else '#b64e4a' for k in labels])
            axes[j].axvline(1., color='#222222', linestyle='--'); axes[j].invert_yaxis()
            axes[j].set_xlabel('Actual / fixed minimum (higher is better)'); axes[j].set_title(str(seed))
            axes[j].set_xlim(0, max(1.2, max(ratios)*1.28))
            for i, k in enumerate(labels):
                axes[j].text(ratios[i]+.02, i, f"{c[k]['actual']}/{c[k]['minimum']}", va='center', fontsize=8)
                rows.append(record(name, seed, k, c[k]['actual'], minimum=c[k]['minimum']))
        values = [[seeds[s][k] for k in CONCENTRATION] for s in SEEDS]
        grouped(axes[2], ['D', 'X-only', '1/2 D + 1/2 X'], values, 'Responsible-session participation', 'Probability (lower is better)')
        axes[2].axhline(.2, color='#222222', linestyle='--', label='Fixed maximum .20')
        axes[2].set_ylim(0, max(.25, max(max(v) for v in values)*1.25)); axes[2].legend(fontsize=8)
        for seed in SEEDS:
            for key in CONCENTRATION: rows.append(record(name, seed, key, seeds[seed][key], maximum=.2, unit='probability'))
        finish(fig, name, rows, 'All count floors must pass for each seed; source-probability limits use actual signature-first responsibilities.')

        name = FAMILIES[1]; rows = []; fig, ax = plt.subplots(figsize=(13, 6))
        grouped(ax, [k.replace('_signatures', '').replace('_', ' ') for k in UNION_TYPES],
                [[seeds[s]['union'][k] for k in UNION_TYPES] for s in SEEDS], 'Dangerous union types and geometry')
        for seed in SEEDS:
            for key in UNION_TYPES: rows.append(record(name, seed, key, seeds[seed]['union'][key]))
        finish(fig, name, rows, 'Types overlap. Full/partial buckets require all selected members to agree; conflicts do not fill either quota.')

        name = FAMILIES[2]; rows = []; fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        labels = ['D useful', 'X-only useful', 'D safe-J', 'X-only safe-J']
        safe_keys = [('D', 'safe_useful_signatures'), ('X_only', 'safe_useful_signatures'), ('D', 'safe_J_signatures'), ('X_only', 'safe_J_signatures')]
        grouped(axes[0, 0], labels, [[seeds[s][g][k] for g, k in safe_keys] for s in SEEDS], 'Safe useful supply and all-member safe-J')
        grouped(axes[0, 1], CLASSES, [[seeds[s]['union']['replacement_class_signatures'][str(i)] for i in range(5)] for s in SEEDS], 'Dangerous replacement-class coverage')
        native = ('native_D_danger_actions', 'native_X_danger_actions')
        grouped(axes[1, 0], ['Native D', 'Native X'], [[seeds[s][k] for k in native] for s in SEEDS], 'Native selected hazards (zeros retained)', 'Actions, not signatures')
        conflicts = [(g, k) for g in ('D', 'X_only') for k in ('safe_J_conflicts', 'same_draw_effect_conflicts')]
        grouped(axes[1, 1], ['D J conflict', 'D effect conflict', 'X J conflict', 'X effect conflict'],
                [[seeds[s][g][k] for g, k in conflicts] for s in SEEDS], 'Conflicts retained in supply audit')
        for seed in SEEDS:
            for group, key in safe_keys+conflicts: rows.append(record(name, seed, group+'_'+key, seeds[seed][group][key]))
            for i in range(5): rows.append(record(name, seed, 'replacement_'+CLASSES[i], seeds[seed]['union']['replacement_class_signatures'][str(i)]))
            for key in native: rows.append(record(name, seed, key, seeds[seed][key], unit='actions'))
        finish(fig, name, rows, 'Safe-J counts require every member to be useful and J-recovering. Zero native hazards do not establish learned native safety.')

        name = FAMILIES[3]; rows = []; fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for j, role in enumerate(('utility', 'risk')):
            for fold in (0, 1):
                ax = axes[j, fold]
                for c, seed in enumerate(SEEDS):
                    points = series[(seed, fold, role)]
                    ax.plot([p['update'] for p in points], [p['loss'] for p in points], color=colors[c], label=str(seed),
                            linewidth=1.2, linestyle='-' if c == 0 else '--')
                    for point in points:
                        rows.append(record(name, seed, 'auxiliary_loss', point['loss'], unit='loss',
                                           held_fold=fold, role=role, update=point['update']))
                ax.set_title(f'Training complement of held fold {fold}\nAuxiliary {role}'); ax.set_xlabel('Optimizer update')
                ax.set_ylabel('MSE + ranking' if role == 'utility' else 'Position CE + candidate BCE')
                ax.set_xlim(0, 6000); ax.set_ylim(bottom=0); ax.grid(alpha=.2); ax.legend(fontsize=8)
        finish(fig, name, rows, 'Lines connect saved training points. Fold-specific targets/scales differ. Coarse curves do not replace full6000-row trace/Adam audit or measure held performance. No C0/C1/outer curve is implied.')

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    public_binding = {str(s): {k: seeds[s][k] for k in ['checks', 'D', 'X_only', 'union', *CONCENTRATION,
                                                       'native_D_danger_actions', 'native_X_danger_actions']} for s in SEEDS}
    receipt = dict(render_complete=True, synthetic_fixture=synthetic_fixture,
                   scientific_result=not synthetic_fixture, result_scope='Stage-S aggregate coverage only',
                   sampling_hypothesis_tested=False, actual_visual_QA=False,
                   coverage_passed=coverage['passed'], aux_optimizer_updates=48000,
                   formal_optimizer_updates=coverage['formal_optimizer_updates'],
                   families=list(FAMILIES), files=files,
                   aggregate_input_sha256=digest(public_binding), curve_points_sha256=digest(clean_curves),
                   auxiliary_saved_point_counts={f'{s}/{f}/{r}': len(points) for (s, f, r), points in series.items()},
                   no_private_identifiers_plotted=True, no_new_NN_or_optimizer_updates=True,
                   independent_clinical_safety_claim=False)
    with manifest.open('x') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
    return receipt
