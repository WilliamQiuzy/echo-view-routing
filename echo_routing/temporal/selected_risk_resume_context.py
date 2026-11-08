"""Continuation IO after a separately pinned new-window owner receipt.

No model/Torch imports, queue acquisition, deployment or writes at construction.
Stage drivers supply their literal required source/input registry; an actual Git
proof and current bytes are both required. These IO checks do not establish that
an auxiliary audit or budget forecast is scientifically correct.
"""
import hashlib
import json
import os
import stat
import time
import uuid
from pathlib import Path,PurePosixPath

from echo_routing.temporal.selected_risk_resume_contract import (
    RUN,PUB,PREP,OLD_MANIFEST,digest,output_relative,require_phase,
    validate_execution_authority,
)


CLOSED_MANIFEST_ALIAS=PREP+'/readonly_closed_publication_manifest.json'
CLOSED_PAYLOAD_ALIAS=RUN+'/readonly_closed_publication/'
# Explicit immutable administrative versions, not current budget authority.
HISTORICAL_ADMIN_VERSIONS={'docs/results/2026-11-05_selected_action_risk/README.md': {'observed_sha256': '70410e078febf0cac674351e8879a1b625602b1a106adc63db815f5e6d664425', 'closed_sha256': '9db3e25903ba452ba12cfb5ee0fbb13318aaf036dd8ba1da199819f478e36942', 'observation_path': 'docs/results/2026-11-05_a63_resume_preparation/observed_historical_README.md'}, 'docs/results/2026-11-05_selected_action_risk/local_cpu_receipts.json': {'observed_sha256': '30f81eb1cb498495a5aaa642852eb95a5feeddf00dc0e9966449bb9281173693', 'closed_sha256': 'fb9b3d5194ba9014b6403a70621c5739ca5df1e4598a50dc31cd2ad2bdb182af', 'observation_path': 'docs/results/2026-11-05_a63_resume_preparation/observed_historical_local_cpu_receipts.json'}, 'docs/results/2026-11-05_selected_action_risk/sources_admission.json': {'observed_sha256': 'a7427ba209f573108428ec547b33c4f244d466994177e191abc5d1a601319e01', 'closed_sha256': 'a90229d5e2f05127e785b0e3f5a956013b17b022fb9817ec9c552cf9be4c8abb', 'observation_path': 'docs/results/2026-11-05_a63_resume_preparation/observed_historical_sources_admission.json'}}


def strict_json(raw):
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('duplicate JSON keys forbidden')
            out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=unique,parse_constant=lambda s:(_ for _ in ()).throw(ValueError('nonfinite JSON token')))


def _sha(value,n=64):
    if not isinstance(value,str) or len(value)!=n or any(c not in '0123456789abcdef' for c in value):raise ValueError('canonical external SHA required')
    return value


class Context:
    def __init__(self,root,owner_pin,proposal_pin,baseline_pin,*,now=None):
        # Do not mkdir or touch a queue before authority. The external pins are
        # required CLI arguments, not defaults recovered from the owner itself.
        self.root=Path(root).resolve(strict=True)
        self.owner_pin=owner_pin;self.proposal_pin=proposal_pin;self.baseline_pin=baseline_pin
        self._clock=(lambda:now) if now is not None else time.time
        if any(not isinstance(p,dict) for p in [owner_pin,proposal_pin,baseline_pin]):raise ValueError('preparation only: independently supplied raw owner/proposal/account pins absent')
        self.owner_sha=owner_pin.get('sha256');self.revalidate()

    def _relative(self,path):
        if not isinstance(path,str):raise ValueError('canonical project-relative path required')
        pp=PurePosixPath(path)
        if not path or pp.is_absolute() or str(pp)!=path or any(p in ['.','..'] for p in pp.parts):raise ValueError('canonical project-relative path required')
        return pp.parts

    def _parent(self,path,*,create=False):
        parts=self._relative(path);fd=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            for name in parts[:-1]:
                if create:
                    try:os.mkdir(name,mode=0o700,dir_fd=fd);os.fsync(fd)
                    except FileExistsError:pass
                nxt=os.open(name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                os.close(fd);fd=nxt
            return fd,parts[-1]
        except (OSError,ValueError) as e:
            os.close(fd);raise ValueError('ordinary unsymlinked project ancestors required') from e

    def raw(self,pin):
        if not isinstance(pin,dict) or set(pin)!={'path','sha256'}:raise ValueError('exact external file pin required')
        _sha(pin['sha256']);fd,name=self._parent(pin['path'])
        try:
            leaf=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
            with os.fdopen(leaf,'rb') as f:
                before=os.fstat(f.fileno())
                if not stat.S_ISREG(before.st_mode):raise ValueError('ordinary pinned file required')
                raw=f.read();after=os.fstat(f.fileno())
                if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):raise ValueError('pinned file changed during read')
        except OSError as e:raise ValueError('ordinary unsymlinked pinned file required') from e
        finally:os.close(fd)
        if hashlib.sha256(raw).hexdigest()!=pin['sha256']:raise ValueError('external pinned current bytes differ: '+pin['path'])
        return raw

    def read(self,pin):return strict_json(self.raw(pin))

    def bind_runtime_spec(self,pin,value):
        self.revalidate();parsed=self.read(pin)
        if digest(parsed)!=digest(value) or parsed.get('schema')!='A63_resume_runtime_spec_v1' or parsed.get('owner')!=self.owner_pin or parsed.get('proposal')!=self.proposal_pin or parsed.get('baseline')!=self.baseline_pin:raise ValueError('raw runtime spec and independently supplied owner/account/proposal differ')
        from echo_routing.temporal.selected_risk_resume_contract import stage_graph
        if parsed.get('stage') not in stage_graph():raise ValueError('exact runtime stage required')
        semantics=digest({k:v for k,v in parsed.items() if k!='freeze'})
        if hasattr(self,'_runtime_semantics') and semantics!=self._runtime_semantics:raise ValueError('only self freeze pin may change when finalizing a runtime spec')
        self._runtime_semantics=semantics;self._bound_runtime_spec_pin=pin
        return semantics

    @property
    def runtime_spec_semantics_sha256(self):
        if not hasattr(self,'_runtime_semantics'):raise ValueError('external raw runtime spec must be bound before stage verification')
        return self._runtime_semantics

    def contract_proof(self,contract,pin):
        proof=self.read(pin)
        if 'proof_encoding' not in contract and 'shared_object_pool' not in contract:return proof
        if contract.get('proof_encoding')!='A63_shared_Git_objects_v1' or contract.get('shared_object_pool',{}).get('path')!=RUN+'/Git_proofs/shared_pool.json':raise ValueError('exact optional immutable new-RUN shared Git pool required')
        pool=self.read(contract['shared_object_pool'])
        if pool.get('object_format')!='sha1' or not isinstance(pool.get('objects'),dict):raise ValueError('raw SHA1 object pool required')
        from echo_routing.temporal.selected_risk_storage_engineering import expand_proof
        return expand_proof(proof,pool)

    def _historical_expected(self):
        from scripts.selected_risk_common import CFG,read,verify_pin,validate_source_refs
        cfg=read(CFG);basis=read(verify_pin(cfg['basis']));refs=read(verify_pin(cfg['source_refs']));validate_source_refs(refs,basis)
        return refs['historical_frozen_imports']

    def revalidate(self):
        raw=self.raw(self.owner_pin);self.raw(self.proposal_pin);self.raw(self.baseline_pin)
        validate_execution_authority(raw,self.owner_sha,self.proposal_pin['sha256'],self.baseline_pin['sha256'],self._clock())
        self.owner=strict_json(raw)
        if self.owner['schema']=='A63_explicit_resume_execution_authorization_v2':
            from echo_routing.temporal.selected_risk_resume_contract import STORAGE_APPROVAL,STORAGE_AMENDMENT_PATH,STORAGE_AMENDMENT_SHA
            self.raw(STORAGE_APPROVAL);self.raw({'path':STORAGE_AMENDMENT_PATH,'sha256':STORAGE_AMENDMENT_SHA})
        return True

    def output(self,path,*,public=False):
        self.revalidate();relative=output_relative(path,public=public)
        # Parent fd traversal in write_json/save_value performs the authoritative
        # check. This check also protects writers such as the existing journals.
        p=self.root/relative
        for q in [p,*p.parents]:
            if q==self.root:break
            if q.is_symlink():raise ValueError('new continuation output symlink forbidden')
        return p

    def write_json(self,path,value,*,public=False):
        self.revalidate();relative=output_relative(path,public=public)
        raw=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        fd,name=self._parent(relative,create=True);tmp='.'+name+'.owned-'+uuid.uuid4().hex
        try:
            child=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
            with os.fdopen(child,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            # Hardlink publication cannot overwrite an existing receipt or any
            # symlink. On failure retain the owned temporary as evidence.
            os.link(tmp,name,src_dir_fd=fd,dst_dir_fd=fd,follow_symlinks=False)
            os.fsync(fd);os.unlink(tmp,dir_fd=fd);os.fsync(fd)
        finally:os.close(fd)
        return {'path':relative,'sha256':hashlib.sha256(raw).hexdigest()}

    def save_value(self,path,value):
        from echo_routing.temporal.selected_risk_resume_storage import save_value
        p=self.output(path);out=save_value(p,value)
        return {**out,'path':str(p.relative_to(self.root))}

    def load_value(self,pin):
        from echo_routing.temporal.selected_risk_resume_storage import load_value
        self.raw(pin) # independent NOFOLLOW and external raw-SHA gate
        return load_value(self.root/pin['path'],pin['sha256'])

    def verify_stage(self,stage,pin,*,required):
        """Pure offline actual-Git verification: never invokes git on server."""
        from echo_routing.temporal.safety_git import verify
        self.revalidate();c=self.read(pin)
        fields={'schema','stage','git_commit','objects','historical_frozen_imports','actual_Git_object_proof','runtime_authorization_sha256','runtime_spec_semantics_sha256'}
        if not isinstance(c,dict) or set(c) not in [fields,fields|{'proof_encoding','shared_object_pool'}] or c['schema']!='A63_resume_stage_freeze_v1' or c['stage']!=stage or c['runtime_authorization_sha256']!=self.owner_sha:raise ValueError('exact stage/new-owner freeze required')
        spec=self.read(self._bound_runtime_spec_pin) if hasattr(self,'_bound_runtime_spec_pin') else None
        if spec is None or spec.get('stage')!=stage or spec.get('freeze')!=pin or c['runtime_spec_semantics_sha256']!=self.runtime_spec_semantics_sha256 or digest({k:v for k,v in spec.items() if k!='freeze'})!=self.runtime_spec_semantics_sha256:raise ValueError('actual freeze pin plus independently bound complete runtime semantics required')
        from echo_routing.temporal.selected_risk_resume_freeze import required as full_required
        try:complete=set(full_required(self.root,stage,self._bound_runtime_spec_pin,spec))
        except (KeyError,TypeError,FileNotFoundError) as exc:raise ValueError('mandatory stage source/input/import registry incomplete') from exc
        _sha(c['git_commit'],40)
        if not isinstance(c['objects'],dict) or not set(required)|complete<=set(c['objects']):raise ValueError('stage source/input closure omitted mandatory inputs')
        for p,h in c['objects'].items():self.raw({'path':p,'sha256':h})
        current=dict(c['objects'])
        historical=self._historical_expected()
        if set(c['historical_frozen_imports'])!=set(historical):raise ValueError('complete literal historical import table required, not an optional subset')
        if historical:
            # The historical table is locked by the immutable scientific source
            # refs, not chosen by the new freeze or a matching Git commit.
            for p,v in c['historical_frozen_imports'].items():
                expected=historical.get(p)
                if expected is None or {k:v[k] for k in ['git_commit','blob_oid','sha256']}!=expected or current.pop(p,None)!=v['sha256']:raise ValueError('literal approved historical imports required')
                verify(self.contract_proof(c,v['proof']),v['git_commit'],{p:v['sha256']})
        verify(self.contract_proof(c,c['actual_Git_object_proof']),c['git_commit'],current)
        return c

    def evidence(self,descriptor):
        from echo_routing.temporal.safety_git import verify
        keys={'payload_pin','git_commit','actual_Git_object_proof','runtime_authorization_sha256'}
        if not isinstance(descriptor,dict) or set(descriptor)!=keys or descriptor['runtime_authorization_sha256']!=self.owner_sha:raise ValueError('same separately authorized window evidence required')
        value=self.read(descriptor['payload_pin']);proof=self.read(descriptor['actual_Git_object_proof'])
        verify(proof,descriptor['git_commit'],{descriptor['payload_pin']['path']:descriptor['payload_pin']['sha256']})
        return {'payload':value,'payload_sha256':digest(value),'git_commit':descriptor['git_commit'],'object_proof_sha256':descriptor['actual_Git_object_proof']['sha256'],'runtime_authorization_sha256':self.owner_sha}

    def enter(self,stage,freeze_pin,descriptors,*,required):
        from scripts.selected_risk_common import CFG,read,validate_config,validate_frozen_prior
        validate_config(read(CFG))
        freeze=self.verify_stage(stage,freeze_pin,required=required)
        validate_frozen_prior(freeze['objects'])
        pins=[self.owner_pin,self.proposal_pin,self.baseline_pin]
        if self.owner['schema']=='A63_explicit_resume_execution_authorization_v2':
            from echo_routing.temporal.selected_risk_resume_contract import STORAGE_APPROVAL,STORAGE_AMENDMENT_PATH,STORAGE_AMENDMENT_SHA
            pins.extend([STORAGE_APPROVAL,{'path':STORAGE_AMENDMENT_PATH,'sha256':STORAGE_AMENDMENT_SHA}])
        for pin in pins:
            if freeze['objects'].get(pin['path'])!=pin['sha256']:raise ValueError('actual stage Git must bind independently supplied owner/proposal/closed account')
        evidence={k:self.evidence(v) for k,v in descriptors.items()}
        require_phase(stage,evidence,runtime_authorization_sha=self.owner_sha)
        if stage=='observe_geometry':
            from echo_routing.temporal.selected_risk_resume_runtime import RuntimeState
            from echo_routing.temporal.selected_risk_resume_estimate import _closed_storage_stage
            spec=self.read(self._bound_runtime_spec_pin)
            _closed_storage_stage(self,spec,{k:v['payload'] for k,v in evidence.items()},RuntimeState(self,spec,self._bound_runtime_spec_pin)._worker_rows(),'source_FIT_adoption')
        if stage in ['admit_formal','prepare_formal','source_CONFIRM','admit_formal_after_CONFIRM','train_formal']:
            spec=self.read(self._bound_runtime_spec_pin);pin=spec['arguments'].get('complete_backend_readiness')
            if pin is None:raise ValueError('coverage pass cannot authorize formal/CONFIRM before independently reviewed complete backend readiness')
            ready=self.read(pin)
            from echo_routing.temporal.selected_risk_resume_contract import stage_graph
            if ready.get('software_prepared') is not True or ready.get('scientific_runtime_completed') is not False or ready.get('stage_count')!=len(stage_graph()) or ready.get('SPEC_Approved') is not True or ready.get('QUALITY_Approved') is not True:raise ValueError('complete backend static review is mandatory and never a runtime/model success claim')
            needed={p:h for p,h in freeze['objects'].items() if p.endswith('.py') and ('selected_risk_resume_' in p or p.endswith('selected_risk_analysis.py') or p=='scripts/a63_resume_selected_risk.py')}
            if not needed or any(ready.get('code_pins',{}).get(p)!=h for p,h in needed.items()):raise ValueError('backend readiness must bind the exact current complete stage source closure')
            receipts=ready.get('review_receipts')
            if not isinstance(receipts,list) or len(receipts)<2 or len({r['sha256'] for r in receipts})!=len(receipts):raise ValueError('distinct independent SPEC and QUALITY receipt pins required')
            for r in receipts:self.raw(r)
        return freeze,evidence

    def closed_publication_path(self,path):
        parts=self._relative(path);ancestor=self.root
        for part in parts[:-1]:
            ancestor=ancestor/part
            if ancestor.is_symlink() or (ancestor.exists() and not ancestor.is_dir()):raise ValueError('original closed-publication ancestor conflict, never use mirror fallback')
        original=self.root/path
        return path if original.exists() or original.is_symlink() else CLOSED_PAYLOAD_ALIAS+path

    def verify_closed_publication(self,pin):
        if pin!={'path':CLOSED_MANIFEST_ALIAS,'sha256':OLD_MANIFEST}:raise ValueError('permanent closed A63 raw manifest alias identity required')
        value=self.read(pin);rows=value['files']
        if value['file_count']!=839 or len(rows)!=839 or len({r['path'] for r in rows})!=839:raise ValueError('exact frozen predecessor public payload required')
        aliases=[];originals=0;different=[]
        for row in rows:
            # Existing, conflicting, or dangling-symlink originals never fall
            # back. Only an actually absent original may use the new RUN copy.
            path=self.closed_publication_path(row['path'])
            coexistent=False;chosen={'path':path,'sha256':row['sha256']}
            try:raw=self.raw(chosen)
            except ValueError:
                version=HISTORICAL_ADMIN_VERSIONS.get(row['path'])
                if path!=row['path'] or version is None or row['sha256']!=version['closed_sha256']:raise
                observed={'path':row['path'],'sha256':version['observed_sha256']}
                saved={'path':version['observation_path'],'sha256':version['observed_sha256']}
                if self.raw(observed)!=self.raw(saved):raise ValueError('exact observed historical admin bytes required')
                chosen={'path':CLOSED_PAYLOAD_ALIAS+row['path'],'sha256':row['sha256']};raw=self.raw(chosen);path=chosen['path'];coexistent=True
                different.append({'original_observed_pin':observed,'observed_raw_snapshot_pin':saved,'expected_closed_snapshot_pin':chosen,'not_closed_or_current_budget_authority':True})
            if len(raw)!=row['bytes']:raise ValueError('old publication byte length changed')
            if path==row['path']:originals+=1
            elif not coexistent:aliases.append({'original_path':row['path'],'alias_pin':chosen})
        return {'passed':True,'manifest_pin':pin,'original_paths_present_and_verified':originals,'original_missing_aliases_used':aliases,'different_historical_administrative_snapshots':different,'old_namespaces_written':False}
