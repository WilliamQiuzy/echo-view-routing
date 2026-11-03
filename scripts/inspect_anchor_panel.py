#!/usr/bin/env python3
"""Aggregate FIT engineering-input degeneracy only; no model/optimizer forward."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from scripts.anchor_common import RUN, PUB, CFG, read, write, verify_pin, verify_freeze, sequences
from echo_routing.temporal.anchor_runtime import prepare_training
from echo_routing.temporal.anchor_features import fit_normalizer, normalize
from echo_routing.temporal.baselines.registry import decode_with_prob


def main():
    freeze = verify_freeze('panel_metadata_freeze.json')
    gpu = read(PUB / 'FIT_GPU_checks.json')
    profile = np.load(verify_pin(gpu['posteriors']), allow_pickle=True).item()
    fit, cfg = sequences('fit'), read(CFG)
    qs = [fit[i] for i in profile['selected_sequence_indices']]
    a, j = profile['probabilities']['a08'], profile['probabilities']['jepa']
    base = [decode_with_prob('b3', p, {'decode': {'b3': cfg['source']['a08']['original_B3']}})[0] for p in a]
    data = prepare_training(qs, a, j, base)
    rows = {}
    for arm, features in data['features'].items():
        normalized = normalize(features, fit_normalizer(features))
        rows[arm] = {'candidate_rows': len(features),
                     'unique_feature_rows': len(np.unique(features, axis=0)),
                     'nonconstant_feature_columns': int(np.count_nonzero(np.std(features, axis=0))),
                     'nonzero_normalized_entries': int(np.count_nonzero(normalized)),
                     'maximum_absolute_normalized_feature': float(np.max(np.abs(normalized)))}
    write(PUB / 'FIT_panel_variance.json', {'passed': True, 'heads': rows,
          'kind_candidate_counts': {k: len(v) for k, v in data['kind_indices'].items()},
          'target_unique_values': np.unique(data['targets']).tolist(),
          'target_nonzero_count': int(np.count_nonzero(data['targets'])),
          'engineering_panel_Ffit_only': data['F_fit'], 'formal_Ffit_or_normalizer_fitted': False,
          'new_model_forwards': 0, 'new_optimizer_updates': 0,
          'scope': 'aggregate engineering-panel variance/target diagnosis; not efficacy or held-out evidence',
          'freeze_git': freeze['git_commit']})


if __name__ == '__main__':
    if sys.argv[1:] != ['run']:
        raise ValueError('explicit aggregate panel inspection stage required')
    main()
