"""Only watermarked fabricated Stage-S aggregate plots; no data/NN/SGD."""
import csv
import hashlib
import importlib

import pytest

m = importlib.import_module('echo_routing.temporal.selected_risk_figures')
MINIMUMS = {'D_danger30': 30, 'D_sessions10': 10, 'X_danger30': 30, 'X_sessions10': 10,
            'union60': 60, 'union_sessions20': 20, 'Hpositive40': 40, 'Hge8_15': 15,
            'pureFP8': 8, 'pureFP_sessions5': 5, 'A4C_A5C15': 15, 'A4C_A5C_sessions6': 6,
            'PLAX_PSAX8': 8, 'PLAX_PSAX_sessions4': 4, 'fullrun15': 15, 'partial15': 15,
            'D_safe100': 100, 'X_safe100': 100, 'D_safeJ30': 30, 'X_safeJ30': 30}


def fixture():
    seeds = {}
    for seed in [2026110461, 2026110462]:
        checks = {k: {'actual': v, 'minimum': v, 'passed': True} for k, v in MINIMUMS.items()}
        for k in ['max_p_D', 'max_p_X', 'max_p_mix']:
            checks[k] = {'actual': .1, 'maximum': .2, 'passed': True}
        checks['pureFP8'] = {'actual': 0, 'minimum': 8, 'passed': False}
        checks['pureFP_sessions5'] = {'actual': 0, 'minimum': 5, 'passed': False}
        checks['Hpositive40']['actual'] = 60
        checks['max_p_mix']['actual'] = .05
        member = dict(dangerous_signatures=30, responsible_sessions=10, safe_useful_signatures=100,
                      safe_J_signatures=30, safe_J_conflicts=0, same_draw_effect_conflicts=0)
        union = dict(dangerous_signatures=60, responsible_sessions=20, H_positive_signatures=60,
                     H_ge8_signatures=15, pure_FP_signatures=0, pure_FP_sessions=0,
                     A4C_A5C_signatures=15, A4C_A5C_sessions=6, PLAX_PSAX_signatures=8, PLAX_PSAX_sessions=4,
                     full_run_signatures=15, partial_run_signatures=15, geometry_conflicts=0,
                     replacement_class_signatures={str(i): 12 for i in range(5)})
        seeds[str(seed)] = dict(passed=False, checks=checks, failed_checks=['pureFP8', 'pureFP_sessions5'], D=dict(member),
                                X_only=dict(member), union=union, max_p_D=.1, max_p_X=.1, max_p_mix=.05,
                                native_D_danger_actions=0, native_X_danger_actions=0)
    curves = [dict(seed=seed, held_fold=fold, role=role, update=u, loss=loss)
              for seed in [2026110461, 2026110462] for fold in [0, 1] for role in ['utility', 'risk']
              for u, loss in [(100, 1.), (6000, .25)]]
    return dict(passed=False, seeds=seeds, aux_optimizer_updates=48000, formal_optimizer_updates=0,
                source_isolation={'passed': True}), curves


def render(*args, **kwargs):
    assert callable(getattr(m, 'render_coverage', None)), 'new pure Stage-S renderer missing'
    return m.render_coverage(*args, **kwargs)


def test_four_png_pdf_csv_families_with_all_records_and_no_claimed_actual_QA(tmp_path):
    from PIL import Image
    coverage, curves = fixture()
    receipt = render(tmp_path, coverage, curves, synthetic_fixture=True)
    assert receipt['render_complete'] and receipt['synthetic_fixture']
    assert not receipt['actual_visual_QA'] and not receipt['scientific_result']
    assert receipt['coverage_passed'] is False and receipt['formal_optimizer_updates'] == 0
    assert len(receipt['families']) == 4 and len(receipt['files']) == 12
    csvs = []
    for pin in receipt['files']:
        p = tmp_path/pin['path']; raw = p.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == pin['sha256']
        if p.suffix == '.png':
            with Image.open(p) as im: assert im.width >= 1200 and im.height >= 800
        if p.suffix == '.pdf': assert raw.startswith(b'%PDF')
        if p.suffix == '.csv':
            with p.open() as f: records = list(csv.DictReader(f))
            assert records and all(r['synthetic_fixture'] == 'True' for r in records)
            csvs.append((p.name, records))
    curve_rows = next(rows for name, rows in csvs if 'loss' in name)
    assert len(curve_rows) == len(curves)
    native_rows = [r for _, records in csvs for r in records if r['metric'].startswith('native_')]
    assert len(native_rows) == 4 and all(float(r['value']) == 0 for r in native_rows)
    with pytest.raises(FileExistsError): render(tmp_path, coverage, curves, synthetic_fixture=True)


@pytest.mark.parametrize('kind', ['missing_seed', 'inconsistent_gate', 'wrong_floor', 'NaN', 'missing_head', 'duplicate_cursor', 'no_final6000', 'formal_before_coverage'])
def test_bad_or_partial_schema_fails_before_plot_writes(tmp_path, kind):
    coverage, curves = fixture()
    if kind == 'missing_seed': del coverage['seeds']['2026110462']
    if kind == 'inconsistent_gate': coverage['passed'] = True
    if kind == 'wrong_floor': coverage['seeds']['2026110461']['checks']['pureFP8']['minimum'] = 1
    if kind == 'NaN': curves[0]['loss'] = float('nan')
    if kind == 'missing_head': curves = [r for r in curves if (r['seed'], r['held_fold'], r['role']) != (2026110462, 1, 'risk')]
    if kind == 'duplicate_cursor': curves.append(dict(curves[0]))
    if kind == 'no_final6000': curves[1]['update'] = 5999
    if kind == 'formal_before_coverage': coverage['formal_optimizer_updates'] = 24000
    with pytest.raises(ValueError): render(tmp_path, coverage, curves, synthetic_fixture=True)
    assert not list(tmp_path.rglob('*.png')) and not list(tmp_path.rglob('*.csv'))
