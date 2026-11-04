#!/usr/bin/env python3
"""A61 pinned original5 A08 and frozen JEPA; all new joins truly forwarded."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scripts.safety_common import RUN, PUB, CFG, read, write, sha, verify_pin, verify_freeze, sequences, digest
from echo_routing.temporal.anchor_cache import ProbabilityCache
from echo_routing.temporal.safety_barrier import validate as validate_preouter
from echo_routing.temporal.anchor_features import probabilities
from echo_routing.temporal.anchor_inputs import native_reuse_index
from echo_routing.temporal.structured_inputs import windows
from echo_routing.evaluate.window_target import nearest_centre, window_frames
from echo_routing.vit import target_mvit as tm
from echo_routing.temporal.dense_sequence import DenseSequence, chunk_starts, chunk_positions, stitch_probabilities
from echo_routing.temporal import jepa21_source
from echo_routing.temporal.dense_readout import ReadoutPixels


def determinism():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_source(name, cfg, device='cpu'):
    pin = cfg['source'][name]
    path = verify_pin(pin)
    if name == 'a08':
        state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        model = tm.build_model(None, tm.initial_new_state(0))
        expected = 400
    elif name == 'jepa':
        source = ROOT / pin['source_manifest']
        if sha(source) != pin['source_manifest_sha256']:
            raise ValueError('JEPA source manifest changed')
        source = read(source)
        for path, expected_sha in source['loaded_model_subset_SHA'].items():
            if sha(jepa21_source.UPSTREAM / path) != expected_sha:
                raise ValueError('pinned official JEPA source subset changed')
        model = DenseSequence(jepa21_source.build_encoder(), 'J-high', seed=2026110257,
                              inactive_names=source['inactive_video_names'])
        state = torch.load(verify_pin(pin), map_location='cpu', weights_only=True, mmap=True)['model']
        expected = 214
    else:
        raise ValueError('declared frozen producer required')
    if len(state) != expected:
        raise ValueError('complete producer state-key count changed')
    model.load_state_dict(state, strict=True)
    if not all(torch.equal(model.state_dict()[k], value) for k, value in state.items()):
        raise ValueError('strict loaded source differs from pinned weights')
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.eval().to(device)


@torch.no_grad()
def a08_windows(model, q, pixels, centres, device='cuda', batch=32):
    out = []
    centres = np.asarray(centres, int)
    for begin in range(0, len(centres), batch):
        cs = centres[begin:begin + batch]
        samples = [window_frames(int(c), len(q['refs']), 16, 16) for c in cs]
        frames = np.stack([[pixels.frame(*q['refs'][int(i)]) for i in row] for row in samples])
        markers = torch.tensor([tm.marker_slot(row, int(c)) for row, c in zip(samples, cs)], device=device)
        out.append(model(tm.to_rgb(torch.from_numpy(frames).to(device)), tm.marker_planes(markers)).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def a08_sequence(model, q, pixels, device='cuda', batch=32):
    n = len(q['refs'])
    grid = windows(n, 5)
    logits = a08_windows(model, q, pixels, grid['centre'], device, batch)
    result = torch.softmax(torch.from_numpy(logits), 1).numpy()[nearest_centre(n, grid['centre'])]
    probabilities(result)
    return result


@torch.no_grad()
def jepa_sequence(model, q, pixels, device='cuda'):
    n = len(q['refs'])
    starts, chunks = chunk_starts(n), []
    for begin in range(0, len(starts), 2):
        frames, masks = [], []
        for start in starts[begin:begin + 2]:
            indices, mask = chunk_positions(n, start)
            frames.append(np.stack([pixels.frame(*q['refs'][int(i)]) for i in indices]))
            masks.append(mask)
        chunks.extend(model.probabilities(torch.from_numpy(np.stack(frames)).to(device),
                                           torch.from_numpy(np.stack(masks)).to(device)).cpu().numpy())
    result = stitch_probabilities(chunks, n).astype(np.float32)
    probabilities(result)
    return result


def original_a08(half, qs, seed, cfg):
    path = ROOT / cfg['data']['outer_inputs']
    if sha(path) != cfg['data']['outer_inputs_sha256']:
        raise ValueError('original A08 input/cache pins changed')
    pin = read(path)['A08_cache_pins'][half][str(seed)]
    path = verify_pin(pin)
    values = []
    with np.load(path) as value:
        if value['sequence_ids'].tolist() != [q['id'] for q in qs] or str(value['identity']) != pin['identity']:
            raise ValueError('original A08 full sequence order/identity changed')
        prob = torch.softmax(torch.from_numpy(value['logits']), 1).numpy()
        for index, q in enumerate(qs):
            rows = np.flatnonzero(value['seq'] == index)
            if not np.array_equal(value['centre'][rows], windows(len(q['refs']), 5)['centre']):
                raise ValueError('original A08 centre axis changed')
            p = prob[rows][nearest_centre(len(q['refs']), value['centre'][rows])]
            saved = value['sample_probs'][value['sample_offsets'][index]:value['sample_offsets'][index + 1]]
            if p.shape != (len(q['refs']), 5) or np.max(np.abs(p - saved)) > 1e-7:
                raise ValueError('original A08 raw-logit/probability assignment parity failed')
            probabilities(p)
            values.append(p)
    return values, pin['original_params']



def reader(source,role,readonly=True):
    manifest=read(PUB/'input_manifest.json');qs=sequences(role)
    row=manifest['source_cache_roles'].get(role,{}).get(source)
    if row is not None:
        if not readonly:raise ValueError('prior A59 posterior caches are strictly read-only')
        folder=ROOT/row['path']
        if sha(folder/'committed.jsonl')!=row['journal_sha256']:raise ValueError('frozen prior posterior journal changed')
        return ProbabilityCache(folder,row['identity'],[len(q['refs']) for q in qs],readonly=True)
    if role not in ['a61_confirmation_selection','a61_confirmation_calibration']:raise ValueError('new posterior role is not prospective confirmation')
    freeze=verify_freeze('outer_freeze.json')
    cfg=read(CFG);identity=digest({'source':source,'source_contract':cfg['source'][source],'input':manifest['roles'][role]['sha256'],'objects':freeze['objects']})
    return ProbabilityCache(RUN/'cache'/source/role,identity,[len(q['refs']) for q in qs],readonly=readonly)


def cache():
    verify_freeze('outer_freeze.json');cfg=read(CFG)
    if not read(PUB/'outer_admission.json')['all_caps_fit']:raise ValueError('fresh joined contexts require complete admission')
    models=read(PUB/'models_manifest.json')
    if not models['passed'] or models['formal_optimizer_updates']!=60000 or not models['all10finals24inner8taus56sets_frozen']:raise ValueError('all10finals/24inner/8taus/56sets must freeze first')
    seal=read(PUB/'outer_identity.json')
    validate_preouter(models,seal,sha(PUB/'models_manifest.json'),sha(PUB/'calibration.json'),sha(PUB/'readout_plan.json'))
    for row in models['models']:verify_pin({'path':row['path'],'sha256':row['sha256']})
    determinism();records=[]
    for source in ['a08','jepa']:
        model=build_source(source,cfg,'cuda')
        for role in ['a61_confirmation_selection','a61_confirmation_calibration']:
            qs=sequences(role);target=reader(source,role,readonly=False);pixels=ReadoutPixels(qs);start=time.time();units=0
            for i in range(target.cursor,len(qs)):
                target.run_one(i,lambda i=i:(a08_sequence if source=='a08' else jepa_sequence)(model,qs[i],pixels))
                units+=len(windows(len(qs[i]['refs']),5)['centre']) if source=='a08' else (len(qs[i]['refs'])+31)//32
                if (i+1)%25==0:write(RUN/'runtimecoordination.json',{'phase':'fresh_A61_confirmation_cache','source':source,'role':role,'sequences_done':i+1,'sequences':len(qs),'new_neural_units':units,'model_pid':os.getpid(),'epoch':time.time(),'next_action':'both frozen producers on all240 fresh streams,then fixed314 readout and240paired/60oracle/56native'})
            records.append({'source':source,'role':role,'identity':target.identity,'journal_sha256':sha(target.journal),'committed_sequences':target.cursor,'input_sha256':read(PUB/'input_manifest.json')['roles'][role]['sha256'],'new_neural_units':units,'wall_seconds':time.time()-start,'complete':target.cursor==len(qs)})
        del model;torch.cuda.empty_cache()
    write(PUB/'cache_outer_manifest.json',{'passed':True,'records':records,'old_source_roles_reused_read_only':20,'formal_optimizer_updates':60000,'new_protected_media_forwards':0})

if __name__=='__main__':cache()
