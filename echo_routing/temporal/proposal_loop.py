"""A60 pure fixed-step loop, durable inner claims and complete CPU lifecycle."""
import json,os,time
from pathlib import Path
import numpy as np,torch
from echo_routing.temporal.proposal_training import *
from echo_routing.temporal.proposal_features import normalize,scale_targets
from echo_routing.temporal.anchor_cache import sha,atomic_json
def read(path):return json.loads(Path(path).read_text())
def write(path,value):atomic_json(path,value)
def verify_pin(pin):
    path=Path(__file__).resolve().parents[2]/pin['path']
    if sha(path)!=pin['sha256']:raise ValueError('completed inner callback checksum changed')
    return path

def train_one(model, opt, data, plan, normalizer, scales, identity, folder, stop=6000,
              latest_every=100, milestones=(2000, 4000, 6000), on_inner=None, on_progress=None):
    if not 0 <= stop <= 6000 or latest_every < 1:
        raise ValueError('bounded fixed-schedule update cursor required')
    journal = UpdateJournal(folder, identity)
    folder = Path(folder)
    inner_pending = folder / 'inner_pending.json'
    if inner_pending.exists():
        raise ValueError('ambiguous pending inner readout; no automatic replay')
    latest = folder / 'latest.pt'
    if not latest.exists() and journal.cursor == 0:
        atomic_save(checkpoint(model, opt, 0, identity, normalizer, scales), latest)
    start = journal.resume_cursor()
    restore(model, opt, torch.load(latest, map_location='cpu', weights_only=True), identity, normalizer, scales)
    if start > stop:
        raise ValueError('cannot move the committed optimizer cursor backward')
    feature = torch.from_numpy(normalize(data['features'][model.arm], normalizer))
    target = torch.from_numpy(scale_targets(data['targets'], scales))
    def inner(cursor):
        path = folder / f'step{cursor}.pt'
        marker = folder / f'inner_step{cursor}.json'
        expected = {'identity': identity, 'weight_sha256': sha(path), 'step': cursor}
        if marker.exists():
            value = read(marker)
            if any(value[k] != v for k, v in expected.items()):
                raise ValueError('completed inner/checkpoint binding changed')
            result = value['callback_result']
            if isinstance(result, dict) and 'path' in result and 'sha256' in result:
                verify_pin(result)
            return
        if inner_pending.exists():
            raise ValueError('ambiguous pending inner readout')
        with inner_pending.open('x') as handle:
            json.dump(expected, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        fsync_directory(folder)
        result = on_inner(model, cursor, path) if on_inner is not None else None
        write(marker, {**expected, 'callback_result': result,
                       'engineering_helper_without_inner_requested': on_inner is None})
        inner_pending.unlink()
        fsync_directory(folder)
    if start in milestones:
        inner(start)
    for update in range(start, stop):
        batches, indices = [], []
        for kind in ['native', 'constructed']:
            picked = data['kind_indices'][kind][plan[kind][update]]
            indices.extend(picked.tolist())
            batches.append((feature[picked], target[picked]) if len(picked) else None)
        journal.begin(update)
        t0 = time.perf_counter()
        row = step(model, opt, batches, update)
        row.update({'candidate_indices': indices,
                    'physical_feature_positions': sum(data['candidate_rows'][i]['feature_positions'] for i in indices),
                    'optimizer_update_wall_seconds': time.perf_counter() - t0})
        cursor = update + 1
        payload = checkpoint(model, opt, cursor, identity, normalizer, scales) if cursor % latest_every == 0 or cursor in milestones else None
        if cursor in milestones:
            atomic_save(payload, folder / f'step{cursor}.pt')
        journal.finish(row, payload)
        if cursor in milestones:
            inner(cursor)
        if on_progress is not None and cursor % 25 == 0:
            on_progress(cursor, row)
    if torch.cuda.is_initialized():
        raise ValueError('CPU head unexpectedly initialized CUDA; resource correction required')
    rows = [json.loads(line) for line in journal.trace.read_text().splitlines()] if journal.trace.exists() else []
    actual = [float(opt.state[p]['step']) for p in model.parameters()] if stop else []
    if actual and any(value != stop for value in actual):
        raise ValueError('actual Adam counters differ from committed cursor')
    return {'committed_physical_updates': len(rows), 'actual_Adam_update': stop,
            'new_updates_this_process': stop - start,
            'physical_candidate_exposures': sum(row['physical_candidates'] for row in rows),
            'physical_feature_position_exposures': sum(row['physical_feature_positions'] for row in rows),
            'all_tensor_gradient_liveness_over_history': {name: any(row['nonzero_gradient_tensors'][name] for row in rows)
                                                         for name, _ in model.named_parameters()},
            'journal_sha256': sha(journal.trace) if journal.trace.exists() else None,
            'CUDA_initialized': False}

