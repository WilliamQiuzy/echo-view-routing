#!/usr/bin/env python3
"""EX2 D2: unblind the two AI readings and report aggregates. Readings are pseudo-labels; official labels stay unchanged. CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import collections

import numpy as np

import ex1_clip_context as cc
from exam_level import clip_models as cm, io, metrics as mx
from open_routing import data, reading as rd

READING = data.RUN / 'private/reading'
N_CLASSES = 5


def frame_table(ids, frames):
    """How many of the four frames this reader saw the outflow tract in, across recordings."""
    values = [frames[b] for b in ids]
    return {'mean': float(np.mean(values)), 'counts': [values.count(k) for k in range(rd.FRAMES + 1)]}


def stratum_table(entries, first, second, frames_first, frames_second):
    table = {}
    for name in sorted({e['stratum'] for e in entries.values()}):
        ids = [b for b, e in entries.items() if e['stratum'] == name and e['repeat_of'] is None]
        flagged = sum(rd.both_disagree(entries[b]['label'], first[b], second[b]) for b in ids)
        agree = sum(first[b] == second[b] == entries[b]['label'] for b in ids)
        table[name] = {'recordings': len(ids), 'official_label': entries[ids[0]]['label'],
                       'reader1': dict(collections.Counter(first[b] for b in ids)), 'reader2': dict(collections.Counter(second[b] for b in ids)),
                       'reader1_outflow_frames': frame_table(ids, frames_first), 'reader2_outflow_frames': frame_table(ids, frames_second),
                       'both_against_label': flagged, 'both_against_label_rate': flagged / len(ids), 'both_against_label_ci95': rd.wilson(flagged, len(ids)),
                       'both_agree_with_label': agree, 'both_agree_with_label_rate': agree / len(ids),
                       'any_uncertain': sum('uncertain' in (first[b], second[b]) for b in ids)}
    return table


def repeat_agreement(entries, answers):
    """How often one reader gave the same answer to the same recording under two identifiers."""
    pairs = rd.repeat_pairs(entries)
    return {'pairs': len(pairs), 'same': sum(answers[a] == answers[b] for a, b in pairs)}


def clip_metrics(y, pred, excluded_mask):
    keep = ~excluded_mask
    confusion = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(confusion, (y[keep], pred[keep]), 1)
    five, four = y[keep] == cm.A5C, y[keep] == cm.A4C
    return {'clips': int(keep.sum()), 'macro_f1': float(mx.macro_f1(confusion)), 'a5c': int(five.sum()), 'a5c_correct': int((pred[keep][five] == cm.A5C).sum()),
            'a4c': int(four.sum()), 'a4c_correct': int((pred[keep][four] == cm.A4C).sum())}


def cell(sessions, excluded):
    """Clip metrics under two rules: the mean-evidence base and the published position vote."""
    ids = np.array([v for s in sessions for v in s['ids']])
    y = np.concatenate([s['y'] for s in sessions])
    mask = np.isin(ids, sorted(excluded))
    out = {'recordings_excluded': int(mask.sum())}
    for name, prediction in (('mean_evidence', np.concatenate([cc.base_labels(s) for s in sessions])), ('position_vote', np.concatenate([s['vote'] for s in sessions]))):
        out[name] = {'all': clip_metrics(y, prediction, np.zeros_like(mask)), 'excluded': clip_metrics(y, prediction, mask)}
    return out


def sensitivity(excluded):
    """Baseline clip metrics on official labels, with and without the excluded recordings."""
    return {'D1_B0': cell([s for fold in cc.load_d1().values() for s in fold], excluded),
            'D2_A08': [{'half': half, 'seed': seed, **cell(cc.dev_sessions(half, seed), excluded)} for half in io.HALVES for seed in io.SEEDS]}


def main():
    entries = io.read_json(READING / 'key.json')
    first, frames_first, text_first = rd.unpack_answers(io.read_json(READING / 'answers_r1.json'))
    second, frames_second, text_second = rd.unpack_answers(io.read_json(READING / 'answers_r2.json'))
    first, second = rd.check_answers(first, list(entries)), rd.check_answers(second, list(entries))
    table = stratum_table(entries, first, second, frames_first, frames_second)
    originals = [b for b, e in entries.items() if e['repeat_of'] is None]
    flagged = table['train_a5c_voted_a4c']
    control = table['train_a5c_control']
    flagged_ids = {entries[b]['video_id'] for b in originals if rd.both_disagree(entries[b]['label'], first[b], second[b])}
    conflicts = set(io.read_json(data.RUN / 'private/d1_private.json')['label_conflict_recordings'])
    io.write_json(READING / 'flagged_private.json', sorted(flagged_ids))
    io.write_json(data.RUN / 'public/d2_reading_results.json', {
        'strata': table, 'kappa_between_readers': rd.cohen_kappa([first[b] for b in originals], [second[b] for b in originals]),
        'exact_agreement_between_readers': float(np.mean([first[b] == second[b] for b in originals])),
        'repeat_agreement': {'reader1': repeat_agreement(entries, first), 'reader2': repeat_agreement(entries, second)},
        'view_text_reported': {'reader1': len(text_first), 'reader2': len(text_second)},
        'records_against_written_criterion': {'reader1': len(rd.inconsistent_records(first, frames_first)), 'reader2': len(rd.inconsistent_records(second, frames_second))},
        'label_signal_a5c': rd.label_signal(flagged['both_against_label'], flagged['recordings'], control['both_against_label'], control['recordings']),
        'recordings_both_against_label': len(flagged_ids), 'duplicate_label_conflicts_in_audit': len(conflicts),
        'sensitivity_reader_flags_only': sensitivity(flagged_ids), 'sensitivity_reader_flags_and_duplicate_conflicts': sensitivity(flagged_ids | conflicts),
        'pseudo_labels_not_reference_standard': True})
    print('d2 scored', len(originals), len(flagged_ids))


if __name__ == '__main__':
    main()
