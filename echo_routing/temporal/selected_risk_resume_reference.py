"""Strictly read-only A62 reference banks; never materialize missing scores.

Original public receipts lock journals, each journal locks raw private payloads.
C6 references are checked against their original physical source, not supplied
incoming arrays. Future stage callers must also freeze this dependency closure
with actual Git before invoking this reader. No preparation private IO occurs.
"""
import hashlib
import io
import json
import os
from pathlib import PurePosixPath

import numpy as np

from echo_routing.temporal.selected_risk_resume_pairing import _digest

SEEDS=(2026110461,2026110462)
OUTER=('selection','calibration','old_stress_selection','old_stress_calibration',
       'confirmation_selection','confirmation_calibration','a60_confirmation_selection',
       'a60_confirmation_calibration','a61_confirmation_selection','a61_confirmation_calibration',
       'a62_confirmation_selection','a62_confirmation_calibration')
ROLES=('CAL','AUDIT')+OUTER
PREFIX='docs/results/2026-11-04_anchor_context_risk/'
PUBLIC={
 PREFIX+'input_manifest.json':'d0bc23cbab300b1b15a3694fdac94d1e9cc3b80ad2999062ec888645174898fb',
 PREFIX+'calibration.json':'fce8ff8d290091533de6008d98243af7e68ceaa579d41e94ce52290d1c954db5',
 PREFIX+'outer_metrics.json':'4efa0c9780b24521cfb257aa4b4978e8c280a17a3e63533ce38fea3873d028bc',
 PREFIX+'models_manifest.json':'2421febaf3acbf47e0651fa09058813b7db30f215ba2f709d1aabb58f638e4a4'}


def _pairs(rows):
    value={}
    for k,v in rows:
        if k in value:raise ValueError('duplicate frozen JSON key')
        value[k]=v
    return value


def _json(raw):
    def invalid(value):raise ValueError('nonfinite frozen JSON number')
    return json.loads(raw,object_pairs_hook=_pairs,parse_constant=invalid)


def _sha(raw):return hashlib.sha256(raw).hexdigest()
def _hash(value):return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def _role(role):
    if role not in ROLES:raise ValueError('original CAL/AUDIT or12 old outer roles only; no new source fallback')


def cache_records(rows):
    """CAL's two recorded uses may repeat identity; never add their work twice."""
    out={};keys=('seed','role','path','identity','journal_sha256','sequences','candidate_vectors')
    for r in rows:
        _role(r['role']);key=(r['seed'],r['role'])
        if type(r['seed']) is not int or r['seed'] not in SEEDS:raise ValueError('two original fixed seeds only')
        if r['path']!=f'runs/a62_anchor_context_risk_v1/q_scores/seed{r["seed"]}/{r["role"]}' or not all(_hash(r[k]) for k in ['identity','journal_sha256']):raise ValueError('original exact role cache path/identity required')
        if any(type(r[k]) is not int or r[k]<0 for k in ['sequences','candidate_vectors']) or not r['sequences']:raise ValueError('complete committed reference count required')
        if key in out and any(r[k]!=out[key][k] for k in keys):raise ValueError('conflicting duplicate reference journal/input binding')
        out.setdefault(key,r)
    return out


def native_mapping(role,qs,original,mapping):
    expected=319 if role=='CAL' else 309 if role=='AUDIT' else None
    if expected is None or len(original)!=928 or len(qs)!=expected+150 or mapping.get('new_mixed_role')!=role+'_mixed':raise ValueError('original928 and original319/309+150 CAL/AUDIT required')
    ix={int(k):v for k,v in mapping['native_original_inner_indices'].items()}
    native={i for i,q in enumerate(qs) if q['kind']=='native'}
    if len(native)!=expected or set(ix)!=native or len(set(ix.values()))!=expected:raise ValueError('exact unique original inner native subset required')
    for i,j in ix.items():
        if type(j) is not int or not 0<=j<928 or original[j]['kind']!='native' or _digest(qs[i])!=_digest(original[j]):raise ValueError('CAL/AUDIT original native index/axis/reference changed')
    return ix


def committed_rows(raw,record,lengths):
    if _sha(raw)!=record['journal_sha256']:raise ValueError('public committed journal raw SHA changed')
    rows=[_json(line) for line in raw.splitlines()]
    if len(rows)!=len(lengths) or len(rows)!=record['sequences']:raise ValueError('exact committed complete original axis required')
    for i,(r,n) in enumerate(zip(rows,lengths)):
        if set(r)!={'index','identity','positions','sha256'} or type(r['index']) is not int or r['index']!=i or r['identity']!=record['identity'] or type(r['positions']) is not int or r['positions']!=n or not _hash(r['sha256']):raise ValueError('original ordered complete journal source/axis differs')
    return rows


def parse_q_payload(raw,base,raw_a,raw_j,reference):
    # _check is the qualified pure canonical-action/probability contract. Bypass
    # its IO constructor: every read here is external-SHA/NOFOLLOW Context.raw.
    from echo_routing.temporal.event_risk_cache import EventCache
    with np.load(io.BytesIO(raw),allow_pickle=False) as z:
        if len(z.files)!=6 or set(z.files)!={'old_scores','q','actions_json','old_reference_json','taus_json','seconds'} or _json(str(z['taus_json'])) is not None:raise ValueError('original unthresholded q payload exact schema required')
        old=z['old_scores'];q=z['q']
        if old.dtype!=np.dtype('float64') or q.dtype!=np.dtype('float64'):raise ValueError('original physical float64 C6/C2 required')
        value=dict(old_scores=old.copy(),q=q.copy(),action_payload=_json(str(z['actions_json'])),old_reference=_json(str(z['old_reference_json'])),seconds=float(z['seconds']))
    cache=object.__new__(EventCache);cache.bases=[base];cache.raw_a=[raw_a];cache.raw_j=[raw_j];cache.taus=None;cache.old_references=[reference]
    result=cache._check(value,0)
    return {**result,'old_q':result['q']}


class ReferenceBank:
    def __init__(self,context,inputs):
        from scripts.selected_risk_common import prior_path,relative,read
        self.ctx=context;self.inputs=inputs;self.public_pins=[];public={}
        for p,h in PUBLIC.items():
            if relative(prior_path(p))!=p:raise ValueError('unexpected old public alias')
            pin={'path':p,'sha256':h};public[p.rsplit('/',1)[1]]=context.read(pin);self.public_pins.append(pin)
        self.manifest=public['input_manifest.json'];self.calibration=public['calibration.json'];self.models=public['models_manifest.json']
        self.mapping=context.read(self.manifest['partition_mapping'])
        self.records=cache_records(self.calibration['q_cache_records']+public['outer_metrics.json']['joint_q_cache_cells'])
        if set(self.records)!={(s,r) for s in SEEDS for r in ROLES}:raise ValueError('exact28 original seed/role reference journals required')
        for role in (*ROLES,'inner_cal','CAL_mixed','AUDIT_mixed'):
            if self.manifest['roles'][role]!=inputs.manifest['roles'][role]:raise ValueError('unchanged approved predecessor input manifest binding required')
        oldcfg=read(prior_path('configs/experiments/anchor_context_risk_v1.json'))
        self.A61=context.read(oldcfg['prior_A61']['models_manifest.json'])
        self._banks={};self._bank_hashes={};self._values={};self._inner={};self._raw={}

    def _bytes(self,pin):
        key=(pin['path'],pin['sha256'])
        if key not in self._raw:self._raw[key]=self.ctx.raw(pin)
        return self._raw[key]

    def bank(self,role):
        _role(role)
        if role in self._banks:return self._banks[role]
        if role not in ('CAL','AUDIT'):bank=self.inputs.bank(role,purpose='outer')
        else:
            qs=self.inputs.sequences(role);original=self.inputs.sequences('inner_cal');mapping=self.mapping[role];ix=native_mapping(role,qs,original,mapping)
            sources=self.inputs._teachers('inner_cal',original);mixed=self.inputs.bank(mapping['new_mixed_role'],purpose='inner');pa=[];pj=[];offset=0
            for i,q in enumerate(qs):
                if i in ix:pa.append(sources['a08'][ix[i]]);pj.append(sources['jepa'][ix[i]])
                else:
                    if _digest(q)!=_digest(mixed[0][offset]):raise ValueError('original ordered150 mixed CAL/AUDIT axis changed')
                    pa.append(mixed[1][offset]);pj.append(mixed[2][offset]);offset+=1
            if offset!=150:raise ValueError('exact150 mixed CAL/AUDIT required')
            from scripts.selected_risk_common import CFG,read
            from echo_routing.temporal.baselines.registry import decode_with_prob
            params=read(CFG)['source']['a08']['original_B3'];bb=[decode_with_prob('b3',p,{'decode':{'b3':params}})[0] for p in pa];bank=(qs,pa,pj,bb)
        self._banks[role]=bank;self._bank_hashes[role]=_digest(bank)
        return bank

    def _C6(self,reference,seed,role,i,actions):
        from echo_routing.temporal.proposal_policy_cache import payload
        kind=reference.get('kind')
        native_inner=role in ('CAL','AUDIT') and str(i) in self.mapping[role]['native_original_inner_indices']
        if (kind=='readonly_original_A61_inner')!=native_inner:
            raise ValueError('original native inner positions must use inner928 C6; physical C6 only for mixed/outer')
        if kind=='readonly_original_A61_inner':
            if role not in ('CAL','AUDIT') or set(reference)!={'kind','source_receipt_sha256','payload_sha256','original_inner_index','scores_sha256'}:raise ValueError('exact original native inner C6 provenance required')
            ix=self.mapping[role]['native_original_inner_indices'];index=ix.get(str(i))
            if index is None or reference['original_inner_index']!=index:raise ValueError('C6 original inner index differs')
            receipts=[r for r in self.A61['inner_records'] if (r['seed'],r['arm'],r['step'])==(seed,'S-L',6000)]
            if len(receipts)!=1 or receipts[0]['sha256']!=reference['source_receipt_sha256']:raise ValueError('original A61 exact6000 S-L inner receipt required')
            receipt=self.ctx.read({k:receipts[0][k] for k in ['path','sha256']});pin=receipt['private_outputs']
            if pin['sha256']!=reference['payload_sha256']:raise ValueError('original private inner C6 payload pin changed')
            key=(pin['path'],pin['sha256'])
            if key not in self._inner:
                self._inner[key]=np.load(io.BytesIO(self._bytes(pin)),allow_pickle=True).item()
                if len(self._inner[key]['scores'])!=928:raise ValueError('complete original928 inner scores required')
            matrix=np.asarray(self._inner[key]['scores'][index],np.float64)
        elif kind=='readonly_exact_fixed_C6_payload':
            if set(reference)!={'kind','path','payload_sha256','journal_sha256','scores_sha256'}:raise ValueError('exact fixed physical C6 source reference required')
            path=PurePosixPath(reference['path']);folder=str(path.parent)
            oldrole=role if role in OUTER else role+'_mixed'
            original_index=i if role in OUTER else sum(q['kind']!='native' for q in self.bank(role)[0][:i])
            if role in ('CAL','AUDIT') or role.startswith('a62_confirmation_'):
                expected=f'runs/a62_anchor_context_risk_v1/fixed_scores/seed{seed}/{oldrole}'
            else:
                weights=[r for r in self.models['fixed_A61_models'] if (r['seed'],r['arm'])==(seed,'S-L')]
                if len(weights)!=1:raise ValueError('exact original seed/S-L fixed weight pin required')
                expected=f'runs/a61_action_safety_ranking_v1/policy_readout/{weights[0]["sha256"]}/{oldrole}'
            if path.name!=f'sequence{original_index:06d}.private.npz' or folder!=expected:raise ValueError('physical old C6 must retain original seed/arm/6000/role/index tuple')
            journal=self._bytes({'path':folder+'/committed.jsonl','sha256':reference['journal_sha256']});rows=[_json(line) for line in journal.splitlines()]
            if original_index>=len(rows) or rows[original_index].get('index')!=original_index or rows[original_index].get('sha256')!=reference['payload_sha256']:raise ValueError('C6 payload must be committed in its pinned original journal')
            with np.load(io.BytesIO(self._bytes({'path':str(path),'sha256':reference['payload_sha256']})),allow_pickle=False) as z:
                if len(z.files)!=7 or set(z.files)!={'labels_E','labels_N','labels_C','score_matrix','actions_json','seconds','tau'} or _json(str(z['actions_json']))!=payload(actions):raise ValueError('original C6 payload canonical actions/schema changed')
                matrix=z['score_matrix'].copy()
        else:raise ValueError('only pinned original inner/physical C6 provenance allowed')
        if matrix.dtype!=np.dtype('float64') or matrix.shape!=(len(actions),6) or not np.isfinite(matrix).all() or _sha(matrix.tobytes(order='C'))!=reference['scores_sha256']:raise ValueError('original physical C6 byte/axis digest changed')
        return matrix

    def scores(self,seed,role,bank):
        _role(role)
        if type(seed) is not int or seed not in SEEDS:raise ValueError('original fixed seed required')
        original=self.bank(role)
        if _digest(bank)!=self._bank_hashes[role] or _digest(original)!=self._bank_hashes[role]:raise ValueError('reference cannot qualify caller-swapped or mutated source/Base axes')
        key=(seed,role)
        if key in self._values:return self._values[key]
        qs,aa,jj,bb=original;r=self.records[key]
        fd,_=self.ctx._parent(r['path']+'/committed.jsonl')
        try:
            try:os.stat('pending.json',dir_fd=fd,follow_symlinks=False)
            except FileNotFoundError:pass
            else:raise ValueError('ambiguous old q pending operation; no replay')
        finally:os.close(fd)
        rows=committed_rows(self._bytes({'path':r['path']+'/committed.jsonl','sha256':r['journal_sha256']}),r,[len(q['refs']) for q in qs]);values=[]
        for i,row in enumerate(rows):
            raw=self._bytes({'path':r['path']+f'/sequence{i:06d}.private.npz','sha256':row['sha256']})
            with np.load(io.BytesIO(raw),allow_pickle=False) as z:reference=_json(str(z['old_reference_json']))
            value=parse_q_payload(raw,bb[i],np.asarray(aa[i]).argmax(1),np.asarray(jj[i]).argmax(1),reference)
            matrix=self._C6(reference,seed,role,i,value['actions'])
            if not np.array_equal(matrix,value['old_scores']):raise ValueError('q cache incoming C6 differs from pinned physical C6 source')
            values.append(value)
        if sum(len(v['actions']) for v in values)!=r['candidate_vectors']:raise ValueError('original candidate vector total changed')
        self._values[key]=values;return values

    def thresholds(self):
        rows=self.calibration['thresholds']+self.calibration['legacy_thresholds'];out={s:{} for s in SEEDS}
        for r in rows:
            if r['seed'] not in out or r['policy'] not in ('L0','M0','R1','R2') or r['policy'] in out[r['seed']] or isinstance(r['tau'],bool) or not isinstance(r['tau'],(int,float)) or not np.isfinite(r['tau']) or not 0<=r['tau']<=1:raise ValueError('eight exact frozen original unit-range thresholds required')
            out[r['seed']][r['policy']]=r['tau']
        if any(set(v)!={'L0','M0','R1','R2'} for v in out.values()):raise ValueError('complete original thresholds required')
        return out

    def model_pins(self):
        rows=[{k:r[k] for k in ['seed','arm','path','sha256']} for r in self.models['models']+self.models['fixed_A61_models']]
        if len(rows)!=8 or {(r['seed'],r['arm']) for r in rows}!={(s,a) for s in SEEDS for a in ['S-L','risk','R1','R2']} or any(not _hash(r['sha256']) for r in rows):raise ValueError('original exact8 fixed/new model pins required')
        return rows
