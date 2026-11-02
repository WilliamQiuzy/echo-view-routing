"""A56 paired update/input checks independent of the frozen trainer."""


def matched_journals(left,right,updates):
    if len(left)!=updates or len(right)!=updates:raise ValueError('incomplete matched journals')
    for i,(l,r) in enumerate(zip(left,right)):
        for row,coefficient in ((l,0),(r,1)):
            if row['u']!=i or row['schedule_index']!=i or row['coefficient']!=coefficient:
                raise ValueError('unmatched cursor, schedule or replay coefficient')
            if row['physical_forwards']!=16 or row['physical_backwards']!=16:
                raise ValueError('unmatched physical forward/backward exposure')
        if l['source_plan_sha256']!=r['source_plan_sha256'] or l['replay_plan_sha256']!=r['replay_plan_sha256']:
            raise ValueError('unmatched source/replay plan')
        if l['input_window_sha256']!=r['input_window_sha256']:
            raise ValueError('actual source/replay pixels differ between arms')
    return {'matched_updates':updates,'matched_actual_window_inputs':updates*16,
            'physical_forwards_per_arm':updates*16,'physical_backwards_per_arm':updates*16,
            'same_source_replay_pixels_plans_and_compute':True}
