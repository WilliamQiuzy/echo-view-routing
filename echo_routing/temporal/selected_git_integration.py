"""Explicit new-A63-path Git integration, independent of research dependencies.

Dry-run objects/indexes live in a temporary quarantine. Apply holds index/HEAD
locks and a prepared expected-old branch transaction. Index and ref publication
are NOT crash-atomic; partial publication keeps an intent, original-index backup
and owned marker locks for manual reconciliation. No rollback/force update.
"""
import contextlib,hashlib,json,os,re,selectors,signal,struct,subprocess,tempfile,time,uuid,zlib
from pathlib import Path,PurePosixPath


class IntegrationError(RuntimeError):
    def __init__(self,message,report):super().__init__(message);self.report=report


def sha(raw):return hashlib.sha256(raw).hexdigest()


def environment(**extra):
    env={k:v for k,v in os.environ.items() if k not in ['GIT_INDEX_FILE','GIT_DIR','GIT_WORK_TREE','GIT_COMMON_DIR','GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES']}
    return {**env,'GIT_OPTIONAL_LOCKS':'0','GIT_NO_REPLACE_OBJECTS':'1',**extra}


def command(repo,*args,env=None,data=None):
    p=subprocess.run(['git','-C',str(repo),'-c','core.hooksPath=/dev/null','-c','core.splitIndex=false','-c','core.fsmonitor=false',*args],input=data,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env or environment(),timeout=10)
    if p.returncode:raise RuntimeError('Git plumbing rejected: '+p.stderr.decode(errors='replace')[:1000])
    return p.stdout


def index_entries(raw):
    if len(raw)<32 or raw[:4]!=b'DIRC' or hashlib.sha1(raw[:-20]).digest()!=raw[-20:]:raise ValueError('complete checksummed Git index required')
    version,count=struct.unpack('!II',raw[4:12])
    if version not in [2,3]:raise ValueError('only ordinary full v2/v3 indexes supported; do not convert user index')
    out={};pos=12
    for _ in range(count):
        start=pos
        if pos+62>len(raw)-20:raise ValueError('truncated original index entry')
        flags=struct.unpack('!H',raw[pos+60:pos+62])[0];pos+=62
        if flags&0x4000:
            extended=struct.unpack('!H',raw[pos:pos+2])[0]
            if extended&0x4000:raise ValueError('skip-worktree/sparse entry is outside full-index eligibility')
            pos+=2
        end=raw.find(b'\0',pos)
        if end<0:raise ValueError('unterminated index path')
        name=raw[pos:end];pos=start+((end+1-start+7)//8)*8;key=(name,(flags>>12)&3)
        if key in out:raise ValueError('duplicate original index entry')
        out[key]=raw[start:pos]
    while pos<len(raw)-20:
        kind=raw[pos:pos+4];length=struct.unpack('!I',raw[pos+4:pos+8])[0];pos+=8+length
        if kind in [b'link',b'sdir']:raise ValueError('split/sparse indexes are outside this narrow helper')
    if pos!=len(raw)-20:raise ValueError('malformed index extension coverage')
    return out


def seed_commit_index(tree):
    entries=[]
    for item in tree.split(b'\0'):
        if not item:continue
        metadata,path=item.split(b'\t',1);mode,kind,oid=metadata.split()
        if mode not in [b'100644',b'100755',b'120000',b'160000'] or kind not in [b'blob',b'commit']:raise ValueError('unsupported existing HEAD tree entry')
        entry=struct.pack('!10I',0,0,0,0,0,0,int(mode,8),0,0,0)+bytes.fromhex(oid.decode())+struct.pack('!H',min(len(path),4095))+path+b'\0';entry+=b'\0'*((-len(entry))%8);entries.append((path,entry))
    raw=b'DIRC'+struct.pack('!II',2,len(entries))+b''.join(e for _,e in sorted(entries));return raw+hashlib.sha1(raw).digest()


def allowed(path):
    if not isinstance(path,str) or not re.fullmatch(r'[A-Za-z0-9_./-]+',path) or str(PurePosixPath(path))!=path or any(x in ['.','..'] for x in path.split('/')):return False
    plans={'docs/superpowers/plans/2026-11-05-a63-card-revision.receipt.json', 'docs/superpowers/plans/2026-11-05-a63-selected-action-risk.md', 'docs/superpowers/plans/2026-11-05-a63-card-review-final.receipt.json', 'docs/superpowers/plans/2026-11-05-a63-card-clarification.receipt.json', 'docs/superpowers/plans/2026-11-05-a63-card-review-1.receipt.json', 'docs/superpowers/plans/2026-11-05-a63-card-review-2.receipt.json', 'docs/superpowers/plans/2026-11-05-a63-card-review-summary.json', 'docs/superpowers/plans/2026-11-05-a63-card-verification.json', 'docs/superpowers/plans/2026-11-05-a63-selected-action-risk.v1-review.basis.json', 'docs/superpowers/plans/2026-11-05-a63-card-preparation.receipt.json', 'docs/superpowers/plans/2026-11-05-a63-selected-action-risk.v1-review.md', 'docs/superpowers/plans/2026-11-05-a63-selected-action-risk.basis.json'}
    # These are exact new A63 execution paths, not a generic selected_risk script
    # wildcard. Earlier frozen science contracts remain unchanged. Main apply
    # still independently requires every payload path to be new in HEAD/index.
    scripts={'scripts/selected_risk_shared_contracts.py', 'scripts/supervise_selected_risk_extension.py',
             'scripts/selected_risk_common.py', 'scripts/probe_selected_risk_shape.py',
             'scripts/cache_selected_risk.py', 'scripts/freeze_shared_selected_risk.py',
             'scripts/freeze_selected_risk.py', 'scripts/supervise_selected_risk.py',
             'scripts/collect_selected_risk_costs.py', 'scripts/prepare_selected_risk.py',
             'scripts/probe_selected_risk.py', 'scripts/admit_selected_risk.py',
             'scripts/admit_selected_risk_refined.py'}
    return (path.startswith('docs/results/2026-11-05_selected_action_risk/')
        or path=='configs/experiments/selected_action_risk_v1.json' or path in plans or path in scripts
        or re.fullmatch(r'echo_routing/temporal/selected_risk[A-Za-z0-9_]*\.py',path) is not None
        or path=='echo_routing/temporal/selected_git_integration.py'
        or re.fullmatch(r'scripts/[A-Za-z0-9_]*selected_action_risk[A-Za-z0-9_]*\.py',path) is not None
        or re.fullmatch(r'scripts/a63_[A-Za-z0-9_]+\.py',path) is not None
        or path=='scripts/crossfit_action_selectors.py'
        or re.fullmatch(r'tests/test_selected_risk[A-Za-z0-9_]*\.py',path) is not None
        or path=='tests/test_selected_git_integration.py')


def eligibility(repo):
    # A configured/new user hook is a gate. The plumbing path below cannot run
    # that gate faithfully, so reject before entering rather than bypass it.
    forbidden={'GIT_CONFIG','GIT_CONFIG_COUNT','GIT_CONFIG_PARAMETERS','GIT_CONFIG_GLOBAL','GIT_CONFIG_SYSTEM','GIT_CONFIG_NOSYSTEM','GIT_INDEX_FILE','GIT_DIR','GIT_WORK_TREE','GIT_COMMON_DIR','GIT_OBJECT_DIRECTORY','GIT_ALTERNATE_OBJECT_DIRECTORIES','GIT_NAMESPACE','GIT_REPLACE_REF_BASE','GIT_NO_REPLACE_OBJECTS','GIT_GRAFT_FILE','GIT_SHALLOW_FILE'}
    if any(k in os.environ for k in forbidden):raise ValueError('Git environment override is outside narrow integration eligibility')
    def plain(*args):
        p=subprocess.run(['git','-C',str(repo),*args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=environment(),timeout=10)
        if p.returncode not in [0,1]:raise ValueError('cannot verify original Git gate/configuration')
        return p.returncode,p.stdout
    rc,value=plain('for-each-ref','--format=%(refname)','refs/replace/')
    if rc or value.strip():raise ValueError('default refs/replace must not change raw expected Git objects')
    rc,value=plain('rev-parse','--git-path','info/grafts')
    if rc:raise ValueError('cannot check legacy graft identity')
    grafts=Path(value.decode().strip());grafts=grafts if grafts.is_absolute() else Path(repo)/grafts
    if grafts.exists() and (not grafts.is_file() or grafts.read_bytes().strip()):raise ValueError('legacy grafts are outside raw Git integration eligibility')
    rc,value=plain('config','--get-all','core.hooksPath')
    if rc==0:raise ValueError('configured user hooksPath gate must not be bypassed')
    for name in ['core.sparseCheckout','core.splitIndex']:
        rc,value=plain('config','--bool','--get',name)
        if rc==0 and value.strip()!=b'false':raise ValueError('split/sparse Git configuration is outside eligibility')
    rc,value=plain('rev-parse','--git-path','hooks')
    if rc:raise ValueError('cannot locate default hooks for gate verification')
    hooks=Path(value.decode().strip());hooks=hooks if hooks.is_absolute() else Path(repo)/hooks
    if hooks.exists():
        if not hooks.is_dir():raise ValueError('unreadable default hook directory')
        if any(p.is_file() and not p.name.endswith('.sample') and os.access(p,os.X_OK) for p in hooks.iterdir()):raise ValueError('active user hook gate must not be bypassed')
    return {'configured_hooksPath':False,'active_default_hooks':False,'split_sparse_configuration':False,'Git_environment_override':False,'default_replace_refs':False,'legacy_grafts':False,'raw_Git_no_replace_objects_invariant':True}


def fsync_dir(path):
    fd=os.open(path,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(fd)
    finally:os.close(fd)


def exclusive_file(path,raw):
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    fsync_dir(path.parent)


def journal_write(path,value,first=False):
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode()
    if first:exclusive_file(path,raw);return
    if json.loads(path.read_bytes())['transaction_id']!=value['transaction_id']:raise ValueError('intent changed concurrently; do not overwrite')
    temp=path.with_name(path.name+'.tmp.'+value['transaction_id']);exclusive_file(temp,raw);os.replace(temp,path);fsync_dir(path.parent)


def expect_output(process,expected):
    output=b'';end=time.monotonic()+10
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout,selectors.EVENT_READ)
        while output!=expected:
            if not expected.startswith(output) or time.monotonic()>=end:raise RuntimeError('bounded Git transaction acknowledgement failed')
            if not selector.select(max(0,end-time.monotonic())):raise TimeoutError('Git transaction acknowledgement timeout')
            part=os.read(process.stdout.fileno(),4096)
            if not part:
                process.wait(timeout=2);detail=process.stderr.read(4096).decode(errors='replace')
                raise RuntimeError('Git transaction closed before acknowledgement: '+detail)
            output+=part


def prepare_transaction(repo,branch,new,old):
    p=subprocess.Popen(['git','-C',str(repo),'-c','core.hooksPath=/dev/null','update-ref','--stdin'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=environment(),bufsize=0)
    try:
        p.stdin.write(f'start\noption no-deref\nupdate {branch} {new} {old}\nprepare\n'.encode());p.stdin.flush();expect_output(p,b'start: ok\nprepare: ok\n');return p
    except BaseException:
        abort_transaction(p);raise


def commit_transaction(process):
    process.stdin.write(b'commit\n');process.stdin.flush();expect_output(process,b'commit: ok\n');process.stdin.close()
    if process.wait(timeout=10):raise RuntimeError('Git ref transaction did not exit successfully')


def abort_transaction(process):
    if process is None:return
    if process.poll() is None:
        try:process.stdin.write(b'abort\n');process.stdin.flush()
        except (BrokenPipeError,ValueError):pass
        try:process.wait(timeout=2)
        except subprocess.TimeoutExpired:process.kill();process.wait(timeout=2)
    for stream in [process.stdin,process.stdout,process.stderr]:
        if stream is not None and not stream.closed:stream.close()


def promote(objects,target):
    for path in objects.glob('*/*'):
        name=path.parent.name+path.name;raw=path.read_bytes()
        if hashlib.sha1(zlib.decompress(raw)).hexdigest()!=name:raise ValueError('quarantined object checksum differs')
        folder=target/path.parent.name
        if not folder.exists():folder.mkdir();fsync_dir(target)
        dest=folder/path.name
        if dest.exists():
            if hashlib.sha1(zlib.decompress(dest.read_bytes())).hexdigest()!=name:raise ValueError('existing object is corrupt')
            continue
        tmp=folder/('tmp_a63_'+uuid.uuid4().hex);exclusive_file(tmp,raw)
        try:os.link(tmp,dest)
        except FileExistsError:
            if hashlib.sha1(zlib.decompress(dest.read_bytes())).hexdigest()!=name:raise ValueError('concurrent object checksum differs')
        tmp.unlink();fsync_dir(folder)


@contextlib.contextmanager
def bounded(seconds=120):
    def expired(*args):raise TimeoutError('bounded integration operation exceeded deadline')
    old=signal.getsignal(signal.SIGALRM);timer=signal.getitimer(signal.ITIMER_REAL);began=time.monotonic();signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,seconds)
    try:yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old)
        if timer[0]:signal.setitimer(signal.ITIMER_REAL,max(.000001,timer[0]-(time.monotonic()-began)),timer[1])


def integrate(repo,expected_head,expected_branch,files,message,*,apply=False,expected_index_sha256=None):
    report={'status':'rejected','index_published':False,'ref_published':False};locks=[];transaction=None;intent=None;path=None
    try:
        with bounded(),tempfile.TemporaryDirectory(prefix='a63-git-integration-') as temp:
            repo=Path(repo).resolve();temp=Path(temp);gate=eligibility(repo)
            if Path(command(repo,'rev-parse','--show-toplevel').decode().strip()).resolve()!=repo:raise ValueError('explicit repository worktree root required')
            if not re.fullmatch('[0-9a-f]{40}',expected_head) or not expected_branch.startswith('refs/heads/'):raise ValueError('exact expected SHA1 HEAD and attached full branch required')
            command(repo,'check-ref-format',expected_branch)
            if command(repo,'rev-parse','--show-object-format').strip()!=b'sha1':raise ValueError('SHA1 Git repository required')
            def attached():return command(repo,'symbolic-ref','-q','HEAD').decode().strip()
            def head():return command(repo,'rev-parse','--verify','HEAD').decode().strip()
            if head()!=expected_head or attached()!=expected_branch:raise ValueError('HEAD or attached branch changed')
            def location(name):
                p=Path(command(repo,'rev-parse','--git-path',name).decode().strip());return p if p.is_absolute() else repo/p
            operations=[location(n) for n in ['MERGE_HEAD','CHERRY_PICK_HEAD','REVERT_HEAD','rebase-merge','rebase-apply','sequencer']]
            if any(p.exists() for p in operations):raise ValueError('active merge/rebase/sequencer is outside narrow public integration')
            index=location('index');headpath=location('HEAD');objectdir=location('objects');original=index.read_bytes();entries=index_entries(original)
            if any(stage!=0 for _,stage in entries):raise ValueError('unmerged index is outside eligibility')
            if expected_index_sha256 is not None and sha(original)!=expected_index_sha256:raise ValueError('original user index changed since reviewed dry-run')
            if apply and expected_index_sha256 is None:raise ValueError('apply requires reviewed original-index SHA256')
            if not files or len({r['path'] for r in files})!=len(files):raise ValueError('explicit nonempty unique new-path copyset required')
            new=[];payload={}
            for r in sorted(files,key=lambda r:r['path']):
                p=r['path']
                if not allowed(p) or not re.fullmatch('[0-9a-f]{64}',r['sha256']) or any((p.encode(),stage) in entries for stage in range(4)) or command(repo,'ls-tree','-r','--name-only',expected_head,'--',p):raise ValueError('only untracked-in-HEAD-and-index new A63 paths allowed')
                target=repo/p
                if any(x.is_symlink() for x in [target,*target.parents] if x!=repo.parent) or not target.is_file():raise ValueError('regular nonsymlink copied payload required')
                raw=target.read_bytes();mode='100755' if target.stat().st_mode&0o111 else '100644'
                if sha(raw)!=r['sha256'] or r.get('mode',mode)!=mode:raise ValueError('copied payload bytes/mode differ from explicit reviewed pins')
                payload[p]=raw;new.append({'path':p,'sha256':r['sha256'],'mode':mode})
            objects=temp/'objects';objects.mkdir();env=environment(GIT_OBJECT_DIRECTORY=str(objects),GIT_ALTERNATE_OBJECT_DIRECTORIES=str(objectdir.resolve()))
            commit_index=temp/'commit-index';preserved=temp/'preserved-index';preserved.write_bytes(original);commit_env={**env,'GIT_INDEX_FILE':str(commit_index)};preserve_env={**env,'GIT_INDEX_FILE':str(preserved)}
            commit_index.write_bytes(seed_commit_index(command(repo,'ls-tree','-r','-z',expected_head)))
            for r in new:
                blob=command(repo,'hash-object','-w','--stdin',env=env,data=payload[r['path']]).decode().strip()
                for e in [commit_env,preserve_env]:command(repo,'update-index','--add','--cacheinfo',r['mode'],blob,r['path'],env=e)
            tree=command(repo,'write-tree',env=commit_env).decode().strip();diff=command(repo,'diff-tree','--no-commit-id','--name-status','-z','-r',expected_head,tree,env=env).split(b'\0');expected=[x for r in new for x in [b'A',r['path'].encode()]]+[b'']
            if diff!=expected:raise ValueError('commit diff is not exactly the explicit new-file copyset')
            kept=preserved.read_bytes();after=index_entries(kept)
            if any(after.get(k)!=v for k,v in entries.items()) or set(after)-set(entries)!={(r['path'].encode(),0) for r in new}:raise ValueError('original index entries/stat/flags changed')
            commit=command(repo,'commit-tree',tree,'-p',expected_head,env=env,data=(message+'\n').encode()).decode().strip()
            report.update(status='dry_run',expected_head=expected_head,attached_branch=expected_branch,expected_index_sha256=sha(original),preserved_index_sha256=sha(kept),commit_tree=tree,new_commit=commit,files=new,all_original_index_entries_preserved=True,original_index_entry_count=len(entries),hooks_not_run=True,fresh_eligibility=gate,dryrun_real_index_ref_and_object_store_unchanged=True)
            if not apply:return report
            eligibility(repo)
            nonce=uuid.uuid4().hex;marker=('A63 integration '+nonce+'\n').encode();path=index.parent/('a63-new-paths-'+nonce+'.json');backup=index.parent/('a63-original-index-'+nonce);candidate=index.parent/('a63-preserved-index-'+nonce)
            for lock in [Path(str(index)+'.lock')]:exclusive_file(lock,marker);locks.append(lock)
            if head()!=expected_head or attached()!=expected_branch or index.read_bytes()!=original:raise ValueError('HEAD/index changed before exclusive publish')
            report.update(status='prepared',journal_path=str(path),original_index_backup=str(backup));intent={**report,'transaction_id':nonce,'manual_reconciliation_required':True,'non_atomic_index_and_ref_publication_disclosed':True};journal_write(path,intent,first=True);exclusive_file(backup,original);exclusive_file(candidate,kept)
            report['dryrun_real_index_ref_and_object_store_unchanged']=False;promote(objects,objectdir);transaction=prepare_transaction(repo,expected_branch,commit,expected_head)
            if not Path(str(headpath)+'.lock').exists():raise ValueError('Git prepared branch update did not lock attached HEAD; unsupported Git behavior')
            if head()!=expected_head or attached()!=expected_branch or index.read_bytes()!=original or any(p.exists() for p in operations):raise ValueError('concurrent HEAD/index/operation change at prepared ref barrier')
            if any(sha((repo/r['path']).read_bytes())!=r['sha256'] or ('100755' if (repo/r['path']).stat().st_mode&0o111 else '100644')!=r['mode'] for r in new):raise ValueError('reviewed new payload changed before publish')
            eligibility(repo)
            intent['phase']='index_publication_pending';journal_write(path,intent);os.replace(candidate,index);report['index_published']=True;fsync_dir(index.parent)
            if index.read_bytes()!=kept or attached()!=expected_branch:raise ValueError('published index or attached branch changed concurrently')
            intent.update(phase='index_published_ref_pending',index_published=True);journal_write(path,intent);commit_transaction(transaction);report['ref_published']=True
            if head()!=commit or attached()!=expected_branch or index.read_bytes()!=kept:raise ValueError('published branch/index final state differs')
            report.update(status='applied',dryrun_real_index_ref_and_object_store_unchanged=False);intent.update(report,phase='applied',manual_reconciliation_required=False);journal_write(path,intent)
            for lock in reversed(locks):
                if lock.read_bytes()!=marker:raise ValueError('owned integration lock changed; do not remove')
                lock.unlink();fsync_dir(lock.parent)
            locks=[];return report
    except BaseException as error:
        if intent is not None:
            try:
                report['index_published']=index.read_bytes()!=original;report['observed_target_branch_head']=command(repo,'rev-parse','--verify',expected_branch).decode().strip();report['ref_published']=report['observed_target_branch_head']==report['new_commit']
            except BaseException:report['publication_observation_incomplete']=True
        partial=report.get('index_published') or report.get('ref_published') or report.get('publication_observation_incomplete')
        report.update(status='partial' if partial else 'rejected',error=repr(error),manual_reconciliation_required=bool(partial))
        if intent is not None:
            try:intent.update(report);journal_write(path,intent)
            except BaseException:report['journal_final_update_failed']=True
        if not partial:
            for lock in reversed(locks):
                if lock.read_bytes()==marker:lock.unlink();fsync_dir(lock.parent)
        raise IntegrationError(str(error),report) from error
    finally:abort_transaction(transaction)
