"""Future closure bridge for exact independent cost contracts, no exceptions."""
import json
from pathlib import Path
from echo_routing.temporal.anchor_cache import sha


def validate_bindings(root,run,pub,contracts):
    """Require already actual-Git/current-byte-verified contracts as input."""
    root,run,pub=Path(root),Path(run),Path(pub)
    if 'refinement_probe_freeze.json' not in contracts:raise ValueError('actual refinement probe contract missing')
    profile=json.loads((pub/'cost_refinement_profile.json').read_text());probe=contracts['refinement_probe_freeze.json']
    if profile.get('passed') is not True or profile.get('engineering_freeze_git')!=probe['git_commit']:raise ValueError('successful measurement must bind its exact original probe Git')
    if profile['original_cost_profile_sha256']!=sha(pub/'cost_profile.json') or profile['input_manifest_sha256']!=sha(pub/'input_manifest.json'):raise ValueError('original measurement/input bytes changed')
    for stage in ['sources','formal','outer']:
        a=json.loads((pub/f'{stage}_admission.json').read_text());version=a.get('version');name=f'{stage}_refined_admission_v{version}_freeze.json'
        if a.get('stage')!=stage or a.get('all_caps_fit') is not True or name not in contracts:raise ValueError('current complete admission not tied to actual verified engineering contract')
        path=run/'engineering_contracts'/name;expected={'path':str(path.relative_to(root)),'sha256':sha(path)}
        if a.get('independent_engineering_contract')!=expected:raise ValueError('canonical admission engineering contract pin changed')
    return True


def collect(full=True):
    from scripts.event_risk_common import ROOT,RUN,PUB
    from echo_routing.temporal.event_risk_engineering import verify
    values={};records=[];errors=[]
    for path in sorted((RUN/'engineering_contracts').glob('*_freeze.json')):
        try:
            v=verify(path.name);values[path.name]=v;records.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'git_commit':v['git_commit'],'objects':len(v['objects'])})
        except Exception as error:
            if full:raise
            errors.append({'path':str(path.relative_to(ROOT)),'error':repr(error)})
    if full:validate_bindings(ROOT,RUN,PUB,values)
    return records,errors
