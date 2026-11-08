"""Read-only predecessor metadata/teachers and separately authorized new caches.

Legacy object-NPY metadata is loaded only after external raw-SHA verification;
no new object pickle is written. Observable native alignment never reads labels.
Only the new producer manifest supplies newly materialized mixed teacher paths.
"""
import io
from echo_routing.temporal.selected_risk_resume_contract import RUN
from echo_routing.temporal.selected_risk_label_free_inputs import native_indices

METADATA_SHA='50f8d33945cb17f6de5c5d34234529a357acf30681c65c580c38fdd541cb8a15'
FIT_ROLES=('FIT_native','F0_native','F1_native','F0_mixed','F1_mixed')


def producer_records(value,phase,metadata_sha,owner_sha):
    if phase not in ['FIT','CONFIRM']:raise ValueError('exact original producer phase required')
    roles=['F0_mixed','F1_mixed'] if phase=='FIT' else ['a63_confirmation_selection','a63_confirmation_calibration']
    n=1500 if phase=='FIT' else 120;rows=value.get('records',[])
    if value.get('complete') is not True or value.get('phase')!=phase or value.get('input_manifest_sha256')!=metadata_sha or value.get('runtime_authorization_sha256')!=owner_sha or (value.get('aux_updates_at_production'),value.get('formal_updates_at_production'),value.get('committed_payloads'))!=(0 if phase=='FIT' else 48000,0,4*n):raise ValueError('complete exact-phase same-owner pre-formal producer required')
    if len(rows)!=4 or {(r['source'],r['role']) for r in rows}!={(s,r) for s in ['a08','jepa'] for r in roles}:raise ValueError('four unique phase/source/role records required')
    for row in rows:
        if row.get('complete') is not True or type(row.get('committed_sequences')) is not int or row['committed_sequences']!=n or not isinstance(row.get('path'),str) or not row['path'].startswith(RUN+'/'):raise ValueError('all new teacher payloads in new RUN must actually be committed')
    return {r:{s:next(v for v in rows if (v['source'],v['role'])==(s,r)) for s in ['a08','jepa']} for r in roles}


class ReadOnlyInputs:
    def __init__(self,context,metadata_pin,*,FIT_producer_pin=None,CONFIRM_producer_pin=None):
        self.context=context
        if metadata_pin['sha256']!=METADATA_SHA:raise ValueError('original A63 metadata and recipes must be reused without redraw')
        self.metadata_pin=metadata_pin;self.manifest=context.read(metadata_pin)
        self.producers={}
        for phase,pin in [('FIT',FIT_producer_pin),('CONFIRM',CONFIRM_producer_pin)]:
            if pin is not None:self.producers[phase]=producer_records(context.read(pin),phase,METADATA_SHA,context.owner_sha)

    def sequences(self,role):
        import numpy as np
        if role not in self.manifest['roles']:raise ValueError('role outside original frozen metadata')
        v=self.manifest['roles'][role];qs=np.load(io.BytesIO(self.context.raw({k:v[k] for k in ['path','sha256']})),allow_pickle=True).tolist()
        if not isinstance(qs,list) or len(qs)!=v['sequences'] or any(len(q['refs'])!=len(q['labels']) for q in qs):raise ValueError('complete original metadata axis lengths required')
        return qs

    def partition(self):
        value=self.context.read(self.manifest['partition_mapping'])
        if len(value['sessions'])!=2 or [len(v) for v in value['sessions']]!=[201,201] or set(value['sessions'][0])&set(value['sessions'][1]):raise ValueError('exact original disjoint 201+201 source session partition required')
        return value

    def _teachers(self,role,qs):
        from echo_routing.temporal.anchor_cache import ProbabilityCache,sha
        source=self.manifest['source_cache_roles'].get(role)
        for phase,roles in self.producers.items():
            if role in roles:source=roles[role]
        if source is None:raise ValueError('complete producer absent; no teacher fallback or implicit source forward')
        out={}
        for name in ['a08','jepa']:
            r=source[name];path=r['path']
            # raw path traversal gate before the qualified eager reader opens
            # files; constructor is readonly and validates every payload.
            fd,leaf=self.context._parent(path+'/journal.jsonl')
            import os
            os.close(fd)
            c=ProbabilityCache(self.context.root/path,r['identity'],[len(q['refs']) for q in qs],readonly=True)
            if c.cursor!=len(qs) or r['input_sha256']!=self.manifest['roles'][role]['sha256'] or sha(c.journal)!=r['journal_sha256']:raise ValueError('complete teacher journal/input/each payload SHA binding required')
            out[name]=[c.load(i) for i in range(len(qs))]
        return out

    def bank(self,role,*,purpose):
        if purpose not in ['geometry','selector','complement_targets','outer','inner']:raise ValueError('explicit data-access purpose required')
        if purpose in ['geometry','selector','complement_targets'] and role not in FIT_ROLES:raise ValueError('FIT-only preparation/held-selector bank')
        if purpose=='outer' and role.startswith('a63_confirmation'):
            # Context independently checks the three-core actual-Git seal and
            # readonly adoption before the readout driver calls this reader.
            if not getattr(self,'outer_adoption_verified',False):raise ValueError('new outer reader is closed until actual head seal and readonly adoption')
        qs=self.sequences(role)
        if role in ['FIT_native','F0_native','F1_native']:
            original=self.sequences('fit');ix=native_indices(original,qs);ps=self._teachers('fit',original)
            ps={k:[v[i] for i in ix] for k,v in ps.items()}
        elif role in ['F0_mixed','F1_mixed'] and 'FIT' not in self.producers:raise ValueError('new FIT teachers have not been produced; no old bank substitution')
        elif role.startswith('a63_confirmation') and 'CONFIRM' not in self.producers:raise ValueError('new confirmation teachers absent')
        else:ps=self._teachers(role,qs)
        from scripts.selected_risk_common import CFG,read
        from echo_routing.temporal.baselines.registry import decode_with_prob
        params=read(CFG)['source']['a08']['original_B3']
        bases=[decode_with_prob('b3',p,{'decode':{'b3':params}})[0] for p in ps['a08']]
        return qs,ps['a08'],ps['jepa'],bases

    def fold_bank(self,fold,*,purpose):
        if type(fold) is not int or fold not in [0,1]:raise ValueError('original source fold0/1 required')
        banks=[self.bank(f'F{fold}_{kind}',purpose=purpose) for kind in ['native','mixed']]
        return tuple(banks[0][i]+banks[1][i] for i in range(4))
