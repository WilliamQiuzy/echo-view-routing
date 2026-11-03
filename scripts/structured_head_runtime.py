"""A58 head-only cache loading, exact MAP metrics and immutable row identities."""
import hashlib,json,time
from pathlib import Path
import torch
from scripts.structured_common import ROOT,RUN,PUB,read,write,sha,sequences,verify_freeze
from scripts.produce_structured_cache import reader
from echo_routing.temporal.structured_bank import PosteriorBank
from echo_routing.temporal.structured_evaluation import decode_bank
from echo_routing.temporal.dense_runtime import metrics
from echo_routing.temporal.dense_analysis import short_metrics

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def load_bank(role,device='cpu'):
    manifest=read(PUB/('cache_fit_manifest.json' if role in ['fit','inner_cal'] else 'cache_outer_manifest.json'));assert manifest['passed'];qs=sequences(role);a=reader('a08_dense',role);b=reader('jepa',role)
    for source,cache in [('a08_dense',a),('jepa',b)]:
        rows=[r for r in manifest['records'] if r['producer']==source and r['role']==role]
        if len(rows)!=1 or not rows[0]['complete'] or rows[0]['identity']!=cache.identity or rows[0]['committed_sequences']!=len(qs) or rows[0]['journal_sha256']!=sha(cache.journal):raise ValueError('complete frozen source cache manifest binding differs')
    return PosteriorBank(qs,a,b,device)

def head_identity(freeze,seed,arm,backend):return digest({'formal_freeze_git':freeze['git_commit'],'objects':freeze['objects'],'seed':seed,'arm':arm,'backend':backend,'schedule':6000})

def evaluate_inner(model,bank,path,step,identity):
    checkpoint_sha=sha(path);binding=digest({'checkpoint_sha256':checkpoint_sha,'head_identity':identity,'posterior_sources':bank.source_identities,'role':'inner_cal'});folder=Path(path).parent/f'inner_step{step}'
    def progress(done,total):write(RUN/'runtimecoordination.json',{'phase':'formal_inner_MAP','seed':model.seed,'arm':model.arm,'step':step,'sequences_done':done,'sequences':total,'model_pid':__import__('os').getpid(),'epoch':time.time(),'next_action':'complete three declared inner rows per model; no outer results yet'})
    predictions,cache=decode_bank(model,bank,folder,binding,progress);result,_=metrics(bank.sequences,predictions);result.update(short_metrics(bank.sequences,predictions,16));record={'seed':model.seed,'arm':model.arm,'step':step,'checkpoint_sha256':checkpoint_sha,'head_identity':identity,'cache':cache,'inner_only':True,'old_A08_inner_exposure_not_OOF':True,**result};target=PUB/'inner'/f'seed{model.seed}_{model.arm}_step{step}.json'
    if target.exists():
        prior=read(target)
        if any(prior[k]!=v for k,v in record.items() if k!='cache') or any(prior['cache'][k]!=v for k,v in cache.items() if k!='new_decodes'):raise ValueError('immutable inner checkpoint metric/cache record changed')
        return prior
    write(target,record);return record
