#!/usr/bin/env python3
"""EX2 X2: A60's one-action oracle with true frame and false-short filters. Label assisted and never deployable. CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')
import ex2_a60_exam as x1
from echo_routing.temporal.anchor_edits import counts
from echo_routing.temporal.proposal_actions import apply
from echo_routing.temporal.proposal_features import targets
from echo_routing.temporal.proposal_oracle import actions, solve
from echo_routing.temporal.proposal_readout_plan import HALVES, TABLES
from exam_level import io
from open_routing import data, safe
from scripts.proposal_common import CFG, PUB, read
from scripts.readout_jepa_proposals import contexts, role_for
from scripts.train_jepa_proposals import role_values

DOMAINS = ('old', 'expanded')
GUARD = 'G'


def scored_actions(base, raw_a, raw_j, truth, domain, fit_f1):
    """Each guarded action once, with its segment counts and its true effects."""
    rows = []
    for action in actions(base, raw_a, raw_j, domain, True):
        m, p, t = counts(apply(base, action), truth)
        rows.append(({'M': m, 'P': p, 'T': t, 'span': action.key}, targets(base, action, truth, fit_f1)))
    return rows


def solve_filters(streams, domain, fit_f1):
    """The oracle under every declared filter. The unchanged Base is always an option."""
    domains = []
    for q, a, j, b in streams:
        m, p, t = counts(b, q['labels'])
        domains.append(({'M': m, 'P': p, 'T': t, 'span': None}, scored_actions(b, a.argmax(1), j.argmax(1), q['labels'], domain, fit_f1)))
    out = {}
    for name in safe.FILTERS:
        options = [[first] + [row for row, value in scored if safe.keep(name, value)] for first, scored in domains]
        out[name] = {k: v for k, v in solve(options, domain, GUARD).items() if k != 'choices'}
    return out


def base_f1(streams):
    m = p = t = 0
    for q, _, _, b in streams:
        x, y, z = counts(b, q['labels'])
        m, p, t = m + x, p + y, t + z
    return 2 * m / (p + t)


def a60_tables():
    """A60's multi streams with their source and Base values, per table and half."""
    ctx = contexts(read(CFG))
    for table in TABLES:
        for half in HALVES:
            roles = [half] if table == 'main' else [half, role_for(table, half)]
            lookup = {}
            for role in roles:
                qs, aa, jj, bb, _ = role_values(role)
                lookup.update({q['id']: (q, a, j, b) for q, a, j, b in zip(qs, aa, jj, bb)})
            yield table, half, [lookup[q['id']] for q in ctx[table][half]['sequences'] if q['kind'] == 'multi']


def check_reproduction(rows):
    """Unfiltered results must equal A60's published guarded oracle rows."""
    published = {(r['table'], r['half'], r['domain']): r for r in read(PUB / 'action_space_diagnostic.json')['oracle_domains'] if r['guard'] == GUARD}
    checked = 0
    for r in rows:
        if r['filter'] == 'none' and r['table'] not in x1.BANKS:
            prior = published[(r['table'], r['half'], r['domain'])]
            if any(r[k] != prior[k] for k in ('M', 'P', 'T')):
                raise SystemExit('the unfiltered oracle does not reproduce A60')
            checked += 1
    if checked != 2 * len(TABLES) * len(HALVES):
        raise SystemExit('unexpected number of reproduced oracle rows')
    return checked


def main():
    fit_f1 = read(PUB / 'training_inputs.json')['F_fit']
    rows = []

    def add(table, half, streams):
        base = base_f1(streams)
        for domain in DOMAINS:
            for name, solved in solve_filters(streams, domain, fit_f1).items():
                rows.append({'table': table, 'half': half, 'filter': name, 'base_f1': base, 'gain': solved['F1'] - base, **solved})
        io.write_json(data.RUN / 'public/x2_safe_oracle.partial.json', rows)

    for table, half, streams in a60_tables():
        add(table, half, streams)
    reproduced = check_reproduction(rows)
    for bank in x1.BANKS:
        for half in io.HALVES:
            qs, a, j, b = x1.load_bank(bank, half)
            add(bank, half, list(zip(qs, a, j, b)))
    gains = {r['half']: r['gain'] for r in rows if r['table'] == 'exam' and r['domain'] == 'expanded' and r['filter'] == 'frame_and_fp'}
    io.write_json(data.RUN / 'public/x2_safe_oracle.json', {'rows': rows, 'a60_rows_reproduced_exactly': reproduced, 'exam_safe_gain': gains,
                                                            'little_safe_room_on_real_order': safe.little_room(gains['selection'], gains['calibration'])})
    print('x2 written', len(rows), reproduced)


if __name__ == '__main__':
    main()
