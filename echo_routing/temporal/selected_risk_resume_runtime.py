"""Concrete runtime services; unavailable without a separately pinned new owner.

Preparation imports are inert. No .git executable is used on the server, no
second queue is created, and the original A63 artifacts are never written.
Receipt-only publication after a stop verifies the unchanged raw authority and
own intent while allowing honest OS/cost evidence after the work deadline.
"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import resource
import stat
import subprocess
import sys
import time
import uuid

from echo_routing.temporal.selected_risk_resume_context import Context,strict_json
from echo_routing.temporal.selected_risk_resume_contract import RUN,PUB,SHARED_QUEUE,stage_graph,storage_cap


def read_pin(root,pin):
    if not isinstance(pin,dict) or set(pin)!={'path','sha256'}:raise ValueError('external exact path/raw SHA required')
    p=PurePosixPath(pin['path'])
    if not str(p) or p.is_absolute() or str(p)!=pin['path'] or '..' in p.parts:raise ValueError('canonical relative bootstrap pin required')
    fd=os.open(Path(root).resolve(),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in p.parts[:-1]:
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
        leaf=os.open(p.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
        with os.fdopen(leaf,'rb') as f:
            if not stat.S_ISREG(os.fstat(f.fileno()).st_mode):raise ValueError('ordinary bootstrap JSON file required')
            raw=f.read()
    finally:os.close(fd)
    if hashlib.sha256(raw).hexdigest()!=pin['sha256']:raise ValueError('external raw bootstrap SHA mismatch')
    return strict_json(raw)


def bootstrap(root,spec_pin,owner_sha):
    s=read_pin(root,spec_pin)
    keys={'schema','stage','owner','proposal','baseline','freeze','predecessors','metadata','FIT_producer','CONFIRM_producer','prep_ledger','local_storage','expected_boot_id','arguments'}
    if not isinstance(s,dict) or set(s)!=keys or s['schema']!='A63_resume_runtime_spec_v1' or s['stage'] not in stage_graph() or s['owner']['sha256']!=owner_sha:raise ValueError('exact independently pinned future runtime spec/new-owner SHA required')
    ctx=Context(root,s['owner'],s['proposal'],s['baseline']);ctx.bind_runtime_spec(spec_pin,s)
    return ctx,s


class RuntimeState:
    def __init__(self,context,spec,spec_pin):
        self.ctx=context;self.spec=spec;self.spec_pin=spec_pin;self.owner=context.owner;self.stage=spec['stage'];self.intent=None;self.log_paths=[]

    def authorize(self):
        self.ctx.revalidate();self.ctx.raw(self.spec_pin);self.ctx.output('namespace_preflight');self.ctx.output('namespace_preflight',public=True);self.caller_ticket()

    def caller_ticket(self):
        from echo_routing.temporal.selected_risk_resume_guardian import start_ticks
        path=os.environ.get('A63_RESUME_CALLER_INTENT_PATH');h=os.environ.get('A63_RESUME_CALLER_INTENT_SHA')
        if not isinstance(path,str) or not path.startswith(RUN+'/caller_intents/') or not path.endswith('.json'):raise ValueError('whole-process directly waiting caller raw intent required before guardian launch')
        pin={'path':path,'sha256':h};v=self.ctx.read(pin)
        if (v.get('caller_pid'),v.get('caller_start_ticks'),v.get('runtime_authorization_sha256'),v.get('runtime_spec'),v.get('stage'),v.get('boot_id'))!=(os.getppid(),start_ticks(os.getppid()),self.ctx.owner_sha,self.spec_pin,self.stage,self.spec['expected_boot_id']):raise ValueError('actual caller boot/PID/start/spec/owner differs from independent raw intent')
        return pin,v

    @contextmanager
    def queue(self):
        fd,name=self.ctx._parent(SHARED_QUEUE)
        try:lock=os.open(name,os.O_RDWR|os.O_NOFOLLOW,dir_fd=fd)
        finally:os.close(fd)
        try:
            if not stat.S_ISREG(os.fstat(lock).st_mode):raise ValueError('original ordinary shared lock must already exist')
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);yield lock
        finally:fcntl.flock(lock,fcntl.LOCK_UN);os.close(lock)

    def inventory(self):
        from echo_routing.temporal.selected_risk_resume_guardian import start_ticks
        if sys.platform!='linux':raise ValueError('actual scientific runtime is server-only; local preparation cannot launch')
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        if boot!=self.spec['expected_boot_id']:raise ValueError('fresh boot differs from independently recorded inventory')
        old=self.ctx.root/'runs/a63_selected_action_risk_v1/active_process.json'
        if old.exists() or old.is_symlink():raise ValueError('old shared queue claim requires explicit independent reconciliation; never remove it')
        own_caller,own_value=self.caller_ticket();closed=self._worker_rows();by_nonce={r['nonce']:r for r in closed if r.get('nonce') is not None}
        caller_closed={r.get('caller_nonce'):r for r in closed};caller_dir=self.ctx.root/RUN/'caller_intents'
        for p in caller_dir.glob('*.json') if caller_dir.exists() else []:
            if p.is_symlink():raise ValueError('caller intent symlink forbidden')
            raw=p.read_bytes();v=strict_json(raw);h=hashlib.sha256(raw).hexdigest()
            if str(p.relative_to(self.ctx.root))==own_caller['path'] and h==own_caller['sha256']:continue
            r=caller_closed.get(v.get('caller_nonce'))
            if r is None or r.get('caller_intent_sha256')!=h:raise ValueError('another unclosed caller intent requires independent reconciliation')
            if start_ticks(v['caller_pid'])==v['caller_start_ticks']:raise ValueError('another prior caller PID/start is still live')
        folder=self.ctx.root/RUN/'launches'
        if folder.exists():
            for p in folder.glob('*.intent.json'):
                if p.is_symlink():raise ValueError('launch intent symlink forbidden')
                raw=p.read_bytes();v=strict_json(raw);r=by_nonce.get(v['nonce'])
                if r is None or r.get('source_intent_sha256')!=hashlib.sha256(raw).hexdigest():raise ValueError('unclosed owned launch intent: no automatic replay or orphan cleanup')
                pid=r.get('model_pid');ticks=r.get('model_start_ticks')
                if pid is not None and ticks is not None and start_ticks(pid)==ticks:raise ValueError('prior owned model PID/start is still alive')
        out=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],check=True,capture_output=True,text=True,timeout=5)
        owners=[int(x.strip()) for x in out.stdout.splitlines() if x.strip()]
        return {'GPU_owners':owners,'boot_id':boot,'guardian_start_ticks':start_ticks(os.getpid())}

    def _worker_rows(self):
        p=self.ctx.root/RUN/'processes.jsonl'
        if not p.exists() and not p.is_symlink():return []
        if p.is_symlink():raise ValueError('new worker ledger symlink forbidden')
        rows=[]
        for line in p.read_bytes().splitlines():
            v=strict_json(line);r=self.ctx.read(v['receipt'])
            if r['receipt_id']!=v['receipt_id']:raise ValueError('canonical live ledger/raw worker receipt binding differs')
            rows.append(r)
        return rows

    def spent(self):
        from echo_routing.temporal.selected_risk_resume_budget import totals
        baseline=self.ctx.read(self.ctx.baseline_pin)
        if baseline.get('preparation_ledger_pin')!=self.spec['prep_ledger']:raise ValueError('new owner-approved account must bind the complete final preparation ledger; no empty/new cutoff reset')
        prep=self.ctx.read(self.spec['prep_ledger'])
        for p,h in prep['file_pins'].items():self.ctx.raw({'path':p,'sha256':h})
        workers=self._worker_rows()
        if any(r.get('unresolved_worker_uncertainty') is True or r.get('GPU_interval_unknown') is True or r.get('inner_worker_OS_unknown') is True for r in workers):raise ValueError('unresolved actual worker GPU/OS uncertainty forbids new scientific admission; preserve partial closure')
        extra_pins=self.spec.get('arguments',{}).get('additional_administrative_fee_receipts',[])
        if not isinstance(extra_pins,list) or len({(p['path'],p['sha256']) for p in extra_pins})!=len(extra_pins):raise ValueError('distinct explicit private administrative fee pins required')
        for worker in workers:
            if any(p not in extra_pins for p in worker.get('additional_administrative_fee_receipts',[])):raise ValueError('previously charged postcutoff fee pin cannot disappear')
        seen={r.get('source_sha256') for r in prep['receipts'] if r.get('source_sha256')};covered={p['sha256'] for p in prep.get('review_raw_components_covered_not_recharged',[])}
        groups={r.get('charge_group',r['receipt_id']) for r in prep['receipts']+workers}
        for worker in workers:
            groups.update(worker.get('covered_charge_groups',[]))
            if worker.get('guardian_snapshot'):covered.add(worker['guardian_snapshot']['sha256'])
        extra=[]
        from echo_routing.temporal.selected_risk_resume_contract import _sha
        for pin in extra_pins:
            if not pin['path'].startswith(RUN+'/administrative_fees/'):raise ValueError('postcutoff fee metadata must use the explicit private new RUN administrative namespace')
            row=self.ctx.read(pin)
            if row.get('measurement_kind')!='administrative_partial_receipt' or row.get('GPU_job') is not False or row.get('CUDA_context_occupied') is not False:raise ValueError('only explicit CPU administrative receipts may supplement preparation')
            h=_sha(row['source_sha256']);source={'path':row['source_receipt'],'sha256':h}
            if not source['path'].startswith((RUN+'/administrative_fees/','docs/results/2026-11-05_a63_resume_preparation/')):raise ValueError('no closed predecessor or foreign source may be recharged')
            original=self.ctx.read(source);value=original
            for key in row['source_CPU_field'].replace('/','.').split('.'):value=value[key]
            if value!=row['measured_CPU_core_seconds']:raise ValueError('actual raw administrative CPU scalar differs')
            group=row['charge_group'];declared=row.get('covers_raw_sha256',[])
            if not isinstance(group,str) or group!=original.get('receipt_id','raw:'+h) or not isinstance(declared,list):raise ValueError('canonical physical receipt charge group required')
            cs={_sha(v) for v in declared}
            if h in seen or h in covered or h in cs or group in groups or cs&(seen|covered):raise ValueError('administrative raw/group/aggregate/child charge already covered')
            seen.add(h);covered.update(cs);groups.add(group);extra.append(row)
        allowances=[]
        for row in workers:
            v=row.get('caller_exit_CPU_allowance_seconds')
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not 0<=v<=1.:raise ValueError('explicit bounded caller exit allowance required, never folded into measured CPU')
            if v:
                t=row['finished_epoch'];allowances.append({'receipt_id':row['receipt_id']+':exit_allowance','started_epoch':t,'finished_epoch':t,'full_process_wall_seconds':0.,'measured_CPU_core_seconds':None,'measurement_kind':'administrative_partial_receipt','CPU_accounting_allowance_seconds':v,'GPU_job':False,'CUDA_context_occupied':False,'observed_worker_exit_code':None,'measurement_host':None,'model_pid':None,'source_CPU_field':'caller_exit_CPU_allowance_seconds','source_parent_receipt_id':row['receipt_id'],'scope':'unmeasured final caller write/exit accounting allowance; timestamp marker is not a physical worker interval'})
        return totals(baseline,prep['receipts']+extra+allowances,workers)

    def storage(self):
        local=self.ctx.read(self.spec['local_storage'])
        if local.get('schema')!='A63_resume_local_storage_v1' or local.get('includes_all_closed_A63_local_worktree_Git_and_ownedtemps') is not True or local.get('future_publication_room_included') is not True or type(local.get('storage_A63_local_upper_bytes')) is not int:raise ValueError('explicit complete local closed/prep/publication storage upper required')
        total=local['storage_A63_local_upper_bytes']
        for relative in ['runs/a63_selected_action_risk_v1','docs/results/2026-11-05_selected_action_risk',RUN,PUB]:
            root=self.ctx.root/relative
            if root.is_symlink():raise ValueError('tracked namespace root symlink forbidden')
            for p in root.rglob('*') if root.exists() else []:
                if p.is_symlink():raise ValueError('tracked A63 namespace symlink forbidden')
                if p.is_file():total+=p.stat().st_size
        return total

    def admission(self):
        from echo_routing.temporal.selected_risk_resume_budget import forecast,remaining_limits
        from echo_routing.temporal.selected_risk_resume_estimate import estimate
        from echo_routing.temporal.selected_risk_resume_inputs import ReadOnlyInputs
        stage=self.stage;self.ctx.enter(stage,self.spec['freeze'],self.spec['predecessors'],required={'echo_routing/temporal/selected_risk_resume_guardian.py','echo_routing/temporal/selected_risk_resume_runtime.py','scripts/a63_resume_selected_risk.py'})
        verified={k:self.ctx.evidence(v)['payload'] for k,v in self.spec['predecessors'].items()}
        closing=stage in ['close_negative','close_full'];arguments=self.spec['arguments'];declared_cpu=arguments['stage_CPU_upper_seconds'];declared_wall=arguments['stage_wall_upper_seconds']
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not __import__('math').isfinite(v) or v<=0 for v in [declared_cpu,declared_wall]):raise ValueError('explicit finite positive stage allocations required')
        if closing:
            if declared_cpu>900 or declared_wall>900:raise ValueError('only bounded900CPU/900wall receipt closure may consume reserved closing resources')
            work={'GPU_seconds':0.,'CPU_seconds':declared_cpu+10.,'storage_upper_bytes':self.storage()+64000000,'remaining_wall_seconds':declared_wall+10.,'scope':'bounded receipt-only closure; original3600CPU/7200wall/128MB reserves may now be consumed, no unrun scientific workload is demanded'}
        else:
            inputs=ReadOnlyInputs(self.ctx,self.spec['metadata'],FIT_producer_pin=self.spec['FIT_producer'],CONFIRM_producer_pin=self.spec['CONFIRM_producer'])
            work=estimate(self.ctx,inputs,self.storage(),verified)
        spent=self.spent();now=time.time();full=forecast(spent,work['GPU_seconds'],work['CPU_seconds'],work['storage_upper_bytes'],now,work['remaining_wall_seconds'],self.owner,closing=closing)
        limits=remaining_limits(spent,now,self.owner,potential_GPU_context=stage in ['source_FIT','source_CONFIRM'],closing=closing)
        return {**full,'remaining_complete_workload':work,'already_spent':spent,'stage_CPU_limit':min(float(declared_cpu),limits['remaining_CPU_seconds']-10.),'stage_wall_limit':min(float(declared_wall),limits['stage_wall_seconds']-10.),'actual_storage_cap':min(storage_cap(self.owner),40000000000-38650409858),'fresh_inventory_and_ledger_recomputed_under_original_queue':True,'evaluated_epoch':now}

    def claim(self,stage,nonce,live):
        lock=os.stat(self.ctx.root/SHARED_QUEUE,follow_symlinks=False)
        value={'stage':stage,'nonce':nonce,'boot_id':live['boot_id'],'guardian_pid':os.getpid(),'guardian_start_ticks':live['guardian_start_ticks'],'runtime_spec':self.spec_pin,'runtime_authorization_sha256':self.ctx.owner_sha,'queue_device':lock.st_dev,'queue_inode':lock.st_ino,'state':'durable_before_Popen_intent','started_epoch':time.time()}
        self.intent=self.ctx.write_json(f'launches/{stage}.{nonce}.intent.json',value);return self.intent

    def command(self,stage,intent,lock_fd):
        return [sys.executable,str(self.ctx.root/'scripts/a63_resume_selected_risk.py'),'--execute','--child','--stage',stage,'--runtime-spec',self.spec_pin['path'],'--runtime-spec-sha',self.spec_pin['sha256'],'--owner-sha',self.ctx.owner_sha,'--guardian-intent',intent['path'],'--guardian-intent-sha',intent['sha256'],'--guardian-lock-fd',str(lock_fd)]

    def popen_options(self,stage):
        tmp=self.ctx.output('tmp');tmp.mkdir(parents=True,exist_ok=True)
        return {'cwd':self.ctx.root,'env':{**os.environ,'CUDA_VISIBLE_DEVICES':'0' if stage in ['source_FIT','source_CONFIRM'] else '','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1','TMPDIR':str(tmp),'MPLCONFIGDIR':str(tmp/'matplotlib'),'CUDA_CACHE_PATH':str(tmp/'cuda_cache'),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}}

    @contextmanager
    def logs(self,stage,nonce):
        a=self.ctx.output(f'logs/{stage}.{nonce}.stdout');b=self.ctx.output(f'logs/{stage}.{nonce}.stderr')
        fd,name=self.ctx._parent(str(a.relative_to(self.ctx.root)),create=True)
        try:
            x=os.open(a.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd);y=os.open(b.name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd);os.fsync(fd)
            with os.fdopen(x,'wb') as out,os.fdopen(y,'wb') as err:
                try:yield out,err
                finally:
                    out.flush();err.flush();os.fsync(out.fileno());os.fsync(err.fileno());self.log_paths=[a,b]
            os.fsync(fd)
        finally:os.close(fd)
        self.log_paths=[a,b]

    def usage(self):
        a=resource.getrusage(resource.RUSAGE_SELF);b=resource.getrusage(resource.RUSAGE_CHILDREN)
        return a.ru_utime+a.ru_stime+b.ru_utime+b.ru_stime

    def record(self,row,intent):
        # Receipt-only path after the deadline: validate identity, not permission
        # for more work. No stage dispatch/model/queue is possible through here.
        required={'receipt_id','supervisor_pid','stage','nonce','model_pid','observed_worker_exit_code','stage_exit_status'}
        if not isinstance(row,dict) or not required<=set(row) or not isinstance(row['nonce'],str) or not row['nonce'] or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789' for c in row['nonce']) or row['receipt_id']!='resume:'+self.stage+':'+row['nonce']:raise ValueError('complete canonical own receipt identity before publication required')
        if row['model_pid'] is None and row['observed_worker_exit_code'] is not None or row['model_pid'] is not None and row.get('actual_child_directly_waited') is not True:raise ValueError('worker OS can only describe an actually created/directly waited child')
        self.ctx.raw(self.ctx.owner_pin);self.ctx.raw(self.ctx.proposal_pin);self.ctx.raw(self.ctx.baseline_pin)
        caller_pin,caller=self.caller_ticket()
        if row['supervisor_pid']!=os.getpid() or row['stage']!=self.stage:raise ValueError('own guardian receipt only')
        if intent is not None:
            own=self.ctx.read(intent)
            if (own['nonce'],own['guardian_pid'],own['runtime_authorization_sha256'])!=(row['nonce'],os.getpid(),self.ctx.owner_sha):raise ValueError('own durable intent identity differs: never replace another receipt')
        elif row['model_pid'] is not None:raise ValueError('created child must have a durable own intent')
        row={**row,'source_intent_sha256':None if intent is None else intent['sha256'],'caller_intent':caller_pin,'caller_intent_sha256':caller_pin['sha256'],'runtime_authorization_sha256':self.ctx.owner_sha,'runtime_spec_sha256':self.spec_pin['sha256'],'window_expired_at_receipt':time.time()>=self.owner['deadline_epoch'],'canonical_charge':False,'additional_administrative_fee_receipts':self.spec.get('arguments',{}).get('additional_administrative_fee_receipts',[]),'CPU_charge_scope':'nested snapshot only; authoritative directly waited caller whole-process CPU replaces this region','log_pins':[{'path':str(p.relative_to(self.ctx.root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in self.log_paths]}
        relative=PUB+f'/process_{row["nonce"]}.json';raw=json.dumps(row,sort_keys=True,separators=(',',':'),allow_nan=False).encode();fd,name=self.ctx._parent(relative,create=True);tmp='.'+name+'.'+uuid.uuid4().hex
        try:
            child=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
            with os.fdopen(child,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            os.link(tmp,name,src_dir_fd=fd,dst_dir_fd=fd,follow_symlinks=False);os.fsync(fd);os.unlink(tmp,dir_fd=fd);os.fsync(fd)
        finally:os.close(fd)
        pin={'path':relative,'sha256':hashlib.sha256(raw).hexdigest()};fd,name=self.ctx._parent(RUN+'/guardian_snapshots.jsonl',create=True)
        try:
            leaf=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_APPEND|os.O_NOFOLLOW,0o600,dir_fd=fd)
            with os.fdopen(leaf,'ab') as f:f.write(json.dumps({'receipt_id':row['receipt_id'],'receipt':pin},sort_keys=True).encode()+b'\n');f.flush();os.fsync(f.fileno())
            os.fsync(fd)
        finally:os.close(fd)
        return {**row,'public_receipt':pin}


def validate_child_ticket(context,spec_pin,intent_pin,lock_fd,stage):
    from echo_routing.temporal.selected_risk_resume_guardian import start_ticks
    context.revalidate();ticket=context.read(intent_pin)
    if ticket['stage']!=stage or ticket['runtime_spec']!=spec_pin or ticket['runtime_authorization_sha256']!=context.owner_sha or ticket['guardian_pid']!=os.getppid() or start_ticks(os.getppid())!=ticket['guardian_start_ticks'] or Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=ticket['boot_id']:raise ValueError('live guardian boot/PID/start/raw-spec/owner ticket required')
    fd=os.fstat(lock_fd);path=os.stat(context.root/SHARED_QUEUE,follow_symlinks=False)
    if not stat.S_ISREG(fd.st_mode) or (fd.st_dev,fd.st_ino)!=(ticket['queue_device'],ticket['queue_inode']) or (path.st_dev,path.st_ino)!=(fd.st_dev,fd.st_ino):raise ValueError('inherited original shared queue descriptor differs')
    fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return True
