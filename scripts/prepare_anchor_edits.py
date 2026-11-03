#!/usr/bin/env python3
"""A59 construction-only metadata preparation; no neural forward or metrics."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.anchor_common import RUN, PUB, CFG, read, write, verify_pin, verify_freeze, save_array, array_pin, sha
from echo_routing.temporal.anchor_inputs import construct_bank, verify_native_groups, workload


def qualify():
    verify_freeze('preparation_freeze.json')
    from scripts.cache_anchor_edits import build_source
    import torch
    from scripts.anchor_common import digest
    cfg, records = read(CFG), {}
    for name, keys in [('a08', 400), ('jepa', 214)]:
        model = build_source(name, cfg, 'cpu')
        state = model.state_dict()
        if len(state) != keys or any(p.requires_grad for p in model.parameters()):
            raise ValueError('complete frozen producer qualification failed')
        records[name] = {'state_keys': len(state), 'parameters': sum(p.numel() for p in model.parameters()),
                         'all_parameters_frozen': True, 'source_sha256': cfg['source'][name]['sha256'],
                         'key_shape_dtype_digest': digest([(k, list(v.shape), str(v.dtype)) for k, v in state.items()]),
                         'model_forwards': 0}
        del model
    if torch.cuda.is_initialized():
        raise ValueError('CPU source qualification unexpectedly initialized CUDA')
    write(PUB / 'source_qualification.json', {'passed': True, 'sources': records,
          'CUDA_initialized': False, 'model_forwards': 0, 'new_optimizer_updates': 0})
    print({'A59_source_qualification_passed': True, 'sources': records}, flush=True)


def metadata():
    verify_freeze('preparation_freeze.json')
    if (PUB / 'input_manifest.json').exists():
        raise ValueError('A59 metadata already sealed; never regenerate a bank')
    cfg = read(CFG)
    verify_pin(cfg['card'])
    for pin in cfg['data']['old_A58_pins'].values():
        verify_pin(pin)
    old_groups = {role: np.load(verify_pin(pin), allow_pickle=True).tolist()
                  for role, pin in cfg['data']['roles'].items()}
    native = {role: [q for q in qs if q['kind'] == 'native'] for role, qs in old_groups.items()}
    isolation = verify_native_groups(native)
    if [len(native[r]) for r in ['fit', 'inner_cal']] != [2736, 628]:
        raise ValueError('fixed A57 native split changed')
    outer_path = ROOT / cfg['data']['old_outer_metadata']
    if sha(outer_path) != cfg['data']['old_outer_metadata_sha256']:
        raise ValueError('old OPEN outer metadata changed')
    outer = np.load(outer_path, allow_pickle=True).item()
    training_sources = {r[0] for qs in native.values() for q in qs for r in q['refs']}
    roles, recipes, summaries = {}, {}, {}
    def seal(role, qs, summary=None):
        path = RUN / 'metadata' / f'{role}.private.npy'
        save_array(path, np.asarray(qs, dtype=object))
        roles[role] = {**array_pin(path), **workload(qs),
                       'native': sum(q['kind'] == 'native' for q in qs)}
        if summary is not None:
            summaries[role] = summary
    for index, role in enumerate(['fit', 'inner_cal']):
        generated, recipe, summary = construct_bank(native[role], cfg['banks']['streams'][role], role, index)
        recipes[role] = recipe
        seal(role, [*native[role], *generated], summary)
    old_manifest = read(verify_pin(cfg['data']['old_A58_pins']['input_manifest.json']))
    for index, half in enumerate(['selection', 'calibration']):
        main = outer[half]['sequences']
        if {r[0] for q in main for r in q['refs']} & training_sources:
            raise ValueError('training source footprint overlaps OPEN evaluation half')
        seal(half, main)
        old_stress = np.load(verify_pin(old_manifest['roles']['stress_' + half]), allow_pickle=True).tolist()
        seal('old_stress_' + half, old_stress)
        role = 'confirmation_' + half
        generated, recipe, summary = construct_bank([q for q in main if q['kind'] == 'native'], 120, role, index + 2)
        recipes[role] = recipe
        seal(role, generated, summary)
    recipe_pins = {}
    for role, recipe in recipes.items():
        path = RUN / 'metadata' / f'{role}_recipes.private.json'
        write(path, recipe)
        recipe_pins[role] = array_pin(path)
    result = {'passed': True, 'roles': roles, 'recipes': recipe_pins, 'construction_summaries': summaries,
              'native_split_isolation': isolation, 'training_outer_recording_overlap': 0,
              'construction_only_reads_source_class_metadata': True,
              'fresh_confirmation_labels_created_and_sealed_before_head_results': True,
              'fresh_confirmation_metrics_or_utility_computed': False,
              'A08_historical_inner_exposure_not_OOF': True,
              'source_patient_or_quarantine_files_opened': 0, 'protected_media_forwards': 0,
              'protected_labels_used_for_training_or_selection': False,
              'new_model_forwards': 0, 'new_optimizer_updates': 0,
              'physical_counts_not_replaced_by_historical_estimates': True}
    write(PUB / 'input_manifest.json', result)
    print({'A59_metadata_sealed': True, 'roles': roles}, flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['metadata']:
        metadata()
    elif sys.argv[1:] == ['qualify']:
        qualify()
    else:
        raise ValueError('explicit metadata/qualify stage required')
