"""Verify every original executed prep/check object; allow only pinned cfg metadata."""
import base64,copy,hashlib,json
from pathlib import Path
from echo_routing.temporal.safety_git import verify,unpack

CONFIG='configs/experiments/action_safety_ranking_v1.json'
EXPECTED={
 'preparation_identity':{'commit':'9fac9e26a2a39e3833b54f3883ebd89933928976','objects':84,'identity_sha256':'995e641f82a57fecc94268172e5016a41a48a3efe71439c1f6982619d64d3070'},
 'training_prepare_identity':{'commit':'a436e8bddfd6ea917e54536b30803c565cd038a0','objects':87,'identity_sha256':'a01cbf2720454d94d86bce30c222a6341aeb58b1ba016e1b840615ddd1c233bc'},
 'fullFIT_check_identity':{'commit':'ed3f6b1833a3b0afe5ded57694085398f84ac239','objects':89,'identity_sha256':'3a20ea218cab7263cbd86e89906bbe837dababb1c5e6a4d0d74855d2ea551579'}}
CORRECTION={'old_config_sha256':'af0f786f3a8d8415950fc79627e104493cf78c7590291a7f7da237678667fbb8','new_config_sha256':'8011b59df4f99a34783ef6b6b0310a86b13f53cf4f9fd91da59c23e739c5163d'}
FIELDS=['F_fit','draw_RNG_domains','kind_candidate_batch','sampling']


def digest(raw):return hashlib.sha256(raw).hexdigest()


def projection(old,new):
    old=copy.deepcopy(old);new=copy.deepcopy(new)
    changed={k for k in set(old['training'])|set(new['training']) if (k in old['training'])!=(k in new['training']) or old['training'].get(k)!=new['training'].get(k)}
    if changed!=set(FIELDS):raise ValueError('exact four disclosed descriptive metadata changes required')
    for k in FIELDS:old['training'].pop(k,None);new['training'].pop(k,None)
    if old!=new:raise ValueError('undisclosed config/model/data/recipe change outside four metadata paths')
    return ['/training/'+k for k in FIELDS]


def validate(bundle,raw_identities,root,*,expected=EXPECTED,config_path=CONFIG,correction=CORRECTION):
    if set(bundle.get('contracts',{}))!=set(expected) or set(raw_identities)!=set(expected):raise ValueError('complete exact three original executed contracts required')
    root=Path(root);current=(root/config_path).read_bytes()
    if digest(current)!=correction['new_config_sha256']:raise ValueError('current config is not the pinned descriptive correction')
    records=[]
    for name,pin in expected.items():
        raw=raw_identities[name]
        if digest(raw)!=pin['identity_sha256']:raise ValueError('original executed contract bytes changed')
        identity=json.loads(raw);objects=identity['objects'];proof=bundle['contracts'][name]['proof']
        if identity['git_commit']!=pin['commit'] or len(objects)!=pin['objects'] or objects.get(config_path)!=correction['old_config_sha256']:raise ValueError('historical source commit/object coverage/config identity changed')
        checked=verify(proof,pin['commit'],objects)
        old_oid=proof['files'][config_path]['blob_oid'];kind,old=unpack(base64.b64decode(proof['objects'][old_oid],validate=True),old_oid)
        if kind!='blob' or digest(old)!=correction['old_config_sha256']:raise ValueError('original config actual Git blob differs')
        paths=projection(json.loads(old),json.loads(current));divergence=[]
        for path,h in objects.items():
            actual=digest((root/path).read_bytes())
            if actual!=h:
                if path!=config_path or actual!=correction['new_config_sha256']:raise ValueError('executed historical source object changed outside pinned cfg correction: '+path)
                divergence.append(path)
        if divergence!=[config_path]:raise ValueError('exact one disclosed current cfg divergence required')
        records.append({'name':name,'identity_sha256':pin['identity_sha256'],'git_commit':pin['commit'],'historical_objects':len(objects),'all_actual_Git_commit_tree_blobs_verified':checked['passed'],'proof_object_count':checked['object_count'],'original_config_sha256':correction['old_config_sha256'],'current_config_sha256':correction['new_config_sha256'],'current_divergence':divergence,'exact_projected_metadata_paths':paths})
    return {'passed':True,'contract_count':len(records),'historical_object_bindings':sum(r['historical_objects'] for r in records),'contracts':records,'all_original_executed_objects_and_current_files_verified':True,'config_projection_does_not_claim_all_numeric_metadata_unchanged':True,'only_unused_batch_metadata_removed':True,'no_model_forward_or_optimizer_update':True}
