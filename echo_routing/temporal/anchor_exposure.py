"""Private-ID-free counts of candidate axes versus conservative whole contexts."""
import numpy as np

from echo_routing.evaluate.failure_units import session_of


def summarize(sequences, candidate_rows, actual_indices=()):
    native = {q['id']: int(q['labels'][0]) for q in sequences if q['kind'] == 'native'}
    if any(r[0] not in native for q in sequences for r in q['refs']):
        raise ValueError('training contexts must stay inside native FIT allowlist')
    def selected(indices):
        contexts, sources, positions, pixel_refs = set(), set(), set(), set()
        for index in set(map(int, indices)):
            if not 0 <= index < len(candidate_rows):
                raise ValueError('candidate exposure index outside complete FIT dataset')
            row = candidate_rows[index]
            qi = int(row['sequence_index'])
            q = sequences[qi]
            s, e = row['span']
            left, right = max(0, s - 16), min(len(q['refs']), e + 16)
            if not 0 <= s < e <= len(q['refs']) or row['feature_positions'] != right - left:
                raise ValueError('candidate feature-axis exposure count changed')
            contexts.add(qi)
            for position in range(left, right):
                ref = tuple(q['refs'][position])
                sources.add(ref[0])
                pixel_refs.add(ref)
                positions.add((qi, position))
        whole = {r[0] for qi in contexts for r in sequences[qi]['refs']}
        return {'sequences': len(contexts), 'feature_axis_sources': len(sources),
                'feature_axis_sessions': len({session_of(v) for v in sources}),
                'whole_sequence_context_sources': len(whole),
                'whole_sequence_context_sessions': len({session_of(v) for v in whole}),
                'whole_sequence_context_source_class_counts': np.bincount([native[v] for v in whole], minlength=5).tolist(),
                'unique_sequence_axis_positions': len(positions), 'unique_source_frame_gamma_refs': len(pixel_refs)}
    eligible = selected(range(len(candidate_rows)))
    actual_indices = list(map(int, actual_indices))
    actual = selected(actual_indices)
    return {'all_forwarded_native_sources': len(native), 'all_native_sessions': len({session_of(v) for v in native}),
            'eligible_sequences': eligible.pop('sequences'),
            'ineligible_sequences': len(sequences) - len({r['sequence_index'] for r in candidate_rows}),
            **{'eligible_' + k: v for k, v in eligible.items()},
            **{'actual_' + k: v for k, v in actual.items()},
            'ineligible_whole_sequence_context_sources': len(native) - eligible['whole_sequence_context_sources'],
            'actual_physical_candidate_exposures': len(actual_indices),
            'actual_unique_candidates': len(set(actual_indices)),
            'actual_feature_axis_position_exposures': sum(candidate_rows[i]['feature_positions'] for i in actual_indices),
            'not_every_forwarded_source_is_a_feature_axis_optimizer_exposure': True,
            'context_scope': 'whole candidate-bearing sequence is conservative context/supervision envelope for upstream16/64 windows; feature-axis counts do not claim exact upstream dependency',
            'sessions_are_not_confirmed_patients': True}
