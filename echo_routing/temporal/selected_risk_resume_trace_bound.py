"""Source-pinned raw JSON bound for the unchanged formal event trace writer.

No compression ratio, observed trace, labels, loss values, or model outcomes
enter this bound. Consumers must verify every returned source pin against
current bytes (for example with Context.raw) before using the arithmetic.
This pure function performs no qualification I/O or scientific execution.
"""
import json


_SOURCE_PINS = (
    ('echo_routing/temporal/event_risk_model.py',
     '3f569d19897ce017822adb79de05739913fb56248db4b2951d964e68bc438622'),
    ('echo_routing/temporal/selected_risk_resume_formal.py',
     'b6f27562abaa7b2c624642834ed3df02fb5ca30c68c5c116395111059037ce25'),
    ('echo_routing/temporal/proposal_training.py',
     '314e20f842132de63cc30436cb41703706c771461a29206915d7c9b564122713'),
)
_IDENTITY_KEYS = ('code', 'config', 'sources', 'input', 'draw_plan',
                  'normalizer', 'target_scales')
_GRADIENT_KEYS = ('network.0.weight', 'network.0.bias',
                  'network.2.weight', 'network.2.bias')
_UPDATES = 6000
_FLOAT_TOKEN_UPPER = 32


def _object_bytes(value_bytes):
    # Default json.dumps delimiters: braces, quoted key, ': ', and ', '.
    return (2 + sum(len(json.dumps(key)) + 2 + size
                    for key, size in value_bytes.items())
            + 2 * max(0, len(value_bytes) - 1))


def _list_bytes(count, item_bytes):
    return 2 + count * item_bytes + 2 * max(0, count - 1)


def formal_trace_raw_bound(candidate_upper):
    """Return a structural per-row and full 6000-row raw-byte upper bound.

    ``candidate_upper`` is a builtin int in [1, 2**63-1]; each of the 64
    global indices lies in [0, candidate_upper-1]. Every fixed key and value
    token below corresponds to the three complete source files in source_pins.
    Field sizes exclude their enclosing key/colon/comma; the row includes its
    final newline and the default spaces in json.dumps(sort_keys=True,
    allow_nan=False). All keys, identities, and number tokens are ASCII, so
    their character counts equal UTF-8 byte counts.

    The five values produced by float(...) or perf_counter subtraction are
    finite binary64 whenever allow_nan=False writes successfully. Python's
    shortest binary64 repr needs at most 24 characters (sign, up to 17
    significant decimal digits, decimal point, and signed three-digit exponent);
    32 characters per float conservatively include further room. Both boolean
    values fit the longer 'false' token. No model is constructed or run here.
    """
    if type(candidate_upper) is not int:
        raise TypeError('candidate_upper must be a builtin positive int')
    if not 1 <= candidate_upper <= 2**63 - 1:
        raise ValueError('candidate_upper must fit positive signed int64')
    index_digits = len(str(candidate_upper - 1))
    fields = {
        # Complete event_risk_model.step return: nine fields.
        'u': 4,  # 0..5999
        'candidate_BCE': _FLOAT_TOKEN_UPPER,
        'loss': _FLOAT_TOKEN_UPPER,
        'lr': _FLOAT_TOKEN_UPPER,
        'physical_candidates': 2,  # exactly 64
        'unsafe_targets': 2,  # 0..64
        'finite_gradients': 5,
        'gradient_norm': _FLOAT_TOKEN_UPPER,
        'nonzero_gradient_tensors': _object_bytes({key: 5 for key in _GRADIENT_KEYS}),
        # Complete train_explicit row.update: three fields.
        'candidate_indices': _list_bytes(64, index_digits),
        'source_kind_blocks': 6,  # exactly [0, 1]
        'optimizer_update_wall_seconds': _FLOAT_TOKEN_UPPER,
        # UpdateJournal.finish adds only this seven-SHA identity object.
        'identity': _object_bytes({key: 66 for key in _IDENTITY_KEYS}),
    }
    row_structure = _object_bytes({key: 0 for key in fields})
    per_row = row_structure + sum(fields.values()) + 1
    return {
        'candidate_upper': candidate_upper,
        'candidate_index_digit_upper': index_digits,
        'per_row_upper': per_row,
        'per_head_raw_bytes': _UPDATES * per_row,
        'updates_per_head': _UPDATES,
        'field_upper_bytes': fields,
        'row_keys_and_punctuation_bytes': row_structure,
        'newline_bytes': 1,
        'float_token_upper': _FLOAT_TOKEN_UPPER,
        'source_pins': [{'path': path, 'sha256': sha256} for path, sha256 in _SOURCE_PINS],
        'legacy_8192': {'per_row_upper': 8192, 'per_head_raw_bytes': _UPDATES * 8192},
    }
