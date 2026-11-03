"""Label-assisted A59 action-domain diagnosis, forbidden as a learned policy.

These helpers run only after all system readouts freeze. Their upper bound is
limited to one raw-supported patch per sequence (with/without the stated guard).
"""
from fractions import Fraction

from echo_routing.temporal.anchor_edits import aligned, apply_action, components, counts, majority
from echo_routing.evaluate.failure_units import match_segments
from echo_routing.temporal.segments import labels_to_segments


def action_options(base, raw, truth, guarded=False):
    base, raw = aligned(base, raw)
    base, truth = aligned(base, truth)
    actions = [(None, base)]
    for span in components(base, raw):
        edited = apply_action(base, raw, span)
        if not guarded or majority(edited) == majority(base):
            actions.append((span, edited))
    return [{'M': m, 'P': p, 'T': t, 'span': span}
            for span, edited in actions for m, p, t in [counts(edited, truth)]]


def fractional_oracle(options):
    if not options:
        raise ValueError('nonempty complete sequence action domains required')
    for domain in options:
        if not domain or domain[0]['span'] is not None:
            raise ValueError('each domain must start with unchanged Base')
        for row in domain:
            if any(not isinstance(row[k], int) or row[k] < 0 for k in ['M', 'P', 'T']):
                raise ValueError('exact nonnegative integer matcher counts required')
            if not row['P'] + row['T'] or row['M'] > min(row['P'], row['T']) or row['T'] != domain[0]['T']:
                raise ValueError('positive denominator and fixed truth count required')
    numerator, denominator, iterations = 0, 1, 0
    visited = set()
    while True:
        choices = []
        for domain in options:
            def key(index):
                row = domain[index]
                priority = (0, 0, 0) if row['span'] is None else (1, *row['span'])
                return (-(2 * row['M'] * denominator - numerator * (row['P'] + row['T'])), *priority)
            choices.append(min(range(len(domain)), key=key))
        chosen = [domain[index] for domain, index in zip(options, choices)]
        matched = sum(row['M'] for row in chosen)
        predicted = sum(row['P'] for row in chosen)
        truth = sum(row['T'] for row in chosen)
        new_numerator, new_denominator = 2 * matched, predicted + truth
        iterations += 1
        if new_numerator * denominator == numerator * new_denominator:
            return {'numerator': new_numerator, 'denominator': new_denominator,
                    'F1': new_numerator / new_denominator, 'M': matched, 'P': predicted, 'T': truth,
                    'choices': choices, 'iterations': iterations,
                    'scope': 'exact one-action raw-supported declared domain only; label assisted, not deployable'}
        current = Fraction(new_numerator, new_denominator)
        if current in visited or new_numerator * denominator < numerator * new_denominator:
            raise ValueError('finite exact fractional maximization failed monotonicity')
        visited.add(current)
        numerator, denominator = new_numerator, new_denominator


def missed_coverage(base, raw, truth, guarded=False):
    base, truth = aligned(base, truth)
    true_segments = labels_to_segments(truth)
    base_matched = set(match_segments(labels_to_segments(base), true_segments))
    union = set()
    for span in components(base, raw):
        edited = apply_action(base, raw, span)
        if guarded and majority(edited) != majority(base):
            continue
        union.update(set(match_segments(labels_to_segments(edited), true_segments)) - base_matched)
    return {'baseline_missed_true_units': len(true_segments) - len(base_matched),
            'candidate_union_recovered_true_units': len(union),
            'union_is_jointly_reachable_claim': False}
