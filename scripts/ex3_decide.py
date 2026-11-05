#!/usr/bin/env python3
"""EX3: apply the plan's reading rules to stored results. Written before any trained output exists. CPU, no labels.

  --seeds 0        the stop rule after the first seed
  --seeds 0 1 2    the final reading
  --arms Z --seeds 0 1 2 --dev a.json b.json --external c.json d.json    one arm, from stored results split over several files
"""
import argparse
import json

from helena_input import data, verdict as vd

METRIC = 'multi segment F1@50'
HALVES = vd.HALVES


def load(name):
    return json.loads((data.PUBLIC / name).read_text())


def external_file(arm_tags):
    return load(f"external_{'_'.join(arm_tags)}.json")


def arm_rows(dev, arm, seeds):
    results = dev['results']
    intervals = {h: [results[h][f'{arm}|{s}']['against_a08'][METRIC]['ci95'] for s in seeds] for h in HALVES}
    floors = [{'half': h, 'boundary_recall': results[h][f'{arm}|{s}']['row']['boundary_recall_multi'], 'native_macro_f1': results[h][f'{arm}|{s}']['row']['native_macro_f1']} for h in HALVES for s in seeds]
    return intervals, floors


def decide_arm(dev, external, arm, seeds):
    intervals, floors = arm_rows(dev, arm, seeds)
    tags = {s: external['tags'][f'{arm}_s{s}'] for s in seeds}
    endpoints = sorted(tags[seeds[0]]['endpoints'])
    by_endpoint = {e: [tags[s]['endpoints'][e]['ci95'] for s in seeds] for e in endpoints}
    usable = {m: all(tags[s]['usable_this_seed'][m] for s in seeds) for m in tags[seeds[0]]['usable_this_seed']}
    open_set = any(usable.values())
    ev9v_ok, external_ok = vd.ev9v_better(intervals, vd.floors_hold(floors)), vd.external_better(by_endpoint)
    signs = {h: vd.direction(intervals[h]) for h in HALVES}
    out = {'seeds': seeds, 'ev9v_signs': signs, 'floors_hold': vd.floors_hold(floors), 'external_signs': {e: vd.direction(v) for e, v in by_endpoint.items()}, 'open_set_usable': usable,
           'reading': vd.reading(list(signs.values()), ev9v_ok, external_ok, open_set)}
    if len(seeds) == 1:
        out['stop'] = vd.stop_after_first_seed([intervals[h][0] for h in HALVES], [by_endpoint[e][0] for e in endpoints], open_set)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--arms', nargs='+', choices=data.ARMS, default=list(data.ARMS))
    parser.add_argument('--dev', nargs='+', help='stored development files; by default the one named after the seeds')
    parser.add_argument('--external', nargs='+', help='stored external files; by default the one named after the tags')
    args = parser.parse_args()
    suffix = '_'.join(f's{s}' for s in args.seeds)
    dev = vd.merge_dev([load(name) for name in args.dev]) if args.dev else load(f'dev_trained_{suffix}.json')
    tags = [f'{arm}_s{s}' for arm in args.arms for s in args.seeds]
    external = vd.merge_external([load(name) for name in args.external]) if args.external else external_file(tags)
    decision = {arm: decide_arm(dev, external, arm, args.seeds) for arm in args.arms}
    data.write_public(f'decision_{data.arm_prefix(args.arms)}{suffix}.json', decision)
    print(json.dumps(decision, indent=1))


if __name__ == '__main__':
    main()
