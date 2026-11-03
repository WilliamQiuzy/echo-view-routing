"""A59 label-free feature/score/policy path and separate FIT supervision."""
import time

import numpy as np
import torch

from echo_routing.temporal.anchor_edits import components, decode, counts, utility
from echo_routing.temporal.anchor_features import candidate_features, normalize


@torch.no_grad()
def infer(model, normalizer, prob_a, prob_j, base, arm):
    start = time.perf_counter()
    raw = np.asarray(prob_a).argmax(1)
    spans = components(base, raw)
    if spans:
        # A needs no JEPA producer. Its second block duplicates pA explicitly.
        evidence_j = prob_a if arm == 'A' else prob_j
        features = np.stack([candidate_features(prob_a, evidence_j, base, span, arm) for span in spans])
        scores = model(torch.from_numpy(normalize(features, normalizer))).cpu().numpy()
    else:
        scores = np.empty(0, np.float32)
    result = {policy: decode(base, raw, scores, guarded=policy == 'G') for policy in ['U', 'G']}
    result.update({'scores': scores, 'spans': spans, 'cached_feature_head_policy_seconds': time.perf_counter() - start,
                   'full_system_inference_seconds': None, 'producer_cost_excluded': True,
                   'operational_required_producers': ['A08'] if arm == 'A' else ['A08', 'JEPA']})
    return result


def prepare_training(sequences, a08, jepa, baselines):
    if not len(sequences) == len(a08) == len(jepa) == len(baselines):
        raise ValueError('complete matched FIT sequence axes required')
    pooled = [counts(base, q['labels']) for q, base in zip(sequences, baselines) if q['kind'] != 'native']
    if not pooled:
        raise ValueError('new FIT construction operating point is missing')
    m, p, t = np.asarray(pooled, dtype=np.int64).sum(0).tolist()
    fit_f1 = 2 * m / (p + t)
    features, targets, rows = {'A': [], 'J': []}, [], []
    kinds = {'native': [], 'constructed': []}
    for index, (q, a, j, base) in enumerate(zip(sequences, a08, jepa, baselines)):
        raw = np.asarray(a).argmax(1)
        for span in components(base, raw):
            kind = 'native' if q['kind'] == 'native' else 'constructed'
            kinds[kind].append(len(rows))
            value = utility(base, raw, span, q['labels'], fit_f1)
            for arm in ['A', 'J']:
                features[arm].append(candidate_features(a, j, base, span, arm))
            targets.append(value['utility'])
            rows.append({'sequence_index': index, 'span': span, 'kind': kind,
                         'delta_M': value['delta_M'], 'delta_P': value['delta_P'],
                         'feature_positions': min(len(base), span[1] + 16) - max(0, span[0] - 16)})
    return {'F_fit': fit_f1, 'FIT_operating_counts': {'M': m, 'P': p, 'T': t},
            'features': {k: np.asarray(v, np.float64).reshape(-1, 98) for k, v in features.items()},
            'targets': np.asarray(targets, np.float32), 'candidate_rows': rows,
            'kind_indices': {k: np.asarray(v, np.int64) for k, v in kinds.items()},
            'native_target_is_count_stability_proxy_not_macroF1_surrogate': True}
