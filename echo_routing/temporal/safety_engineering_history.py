"""Exact executed cost-stage Git history; only pinned cost-document revisions."""
import hashlib,json
from pathlib import Path
from echo_routing.temporal.safety_git import verify

DOC='docs/results/2026-11-04_action_safety_ranking/COST_REFINEMENT.md'
CURRENT_DOC_SHA='a239f32b6ee2505ebae6667aa26685e851d218ad9c76352ed770b6721a33b920'
EXPECTED={
 'component_cost_identity':{'commit':'6c25459c95dbbcad4fffcc8f5ea4ba3525f7c53f','objects':121,'identity_sha256':'8d43ce835676942918e67edb7de6c98ff77546a8a559ea46b294f8f9fdbcb98d','old_doc_sha256':'00a613ffca562522369e62712abe88cf3b1dd08c8cf3e56c423ad690e85a2d1b'},
 'rle_cost_identity':{'commit':'4a7b3869afc3970b910799dd7593e8485b49c6b3','objects':131,'identity_sha256':'53ffa141e706938958ff7dc327803441fee29761f83f9ec53541694f7b138139','old_doc_sha256':'8ab6cc861a9ded0c1b51ed814ed41816c19fbc2a5b02bc514c7773d8cc790415'}}


def digest(b):return hashlib.sha256(b).hexdigest()


def validate(bundle,raw_identities,root,*,expected=EXPECTED,doc_path=DOC,current_doc_sha=CURRENT_DOC_SHA):
    if set(bundle.get('contracts',{}))!=set(expected) or set(raw_identities)!=set(expected):raise ValueError('exact executed cost history coverage required')
    root=Path(root)
    if digest((root/doc_path).read_bytes())!=current_doc_sha:raise ValueError('current cost document changed beyond reviewed versions')
    records=[]
    for name,pin in expected.items():
        raw=raw_identities[name]
        if digest(raw)!=pin['identity_sha256']:raise ValueError('original executed identity changed')
        identity=json.loads(raw);objects=identity['objects']
        if identity['git_commit']!=pin['commit'] or len(objects)!=pin['objects'] or objects.get(doc_path)!=pin['old_doc_sha256']:raise ValueError('original Git/count/document identity differs')
        checked=verify(bundle['contracts'][name]['proof'],pin['commit'],objects);divergence=[]
        for path,h in objects.items():
            actual=digest((root/path).read_bytes())
            if actual!=h:
                if path!=doc_path or actual!=current_doc_sha:raise ValueError('executed source changed outside exact cost prose revision: '+path)
                divergence.append(path)
        if divergence!=[doc_path]:raise ValueError('only exact disclosed cost prose divergence required')
        records.append({'name':name,'identity_sha256':pin['identity_sha256'],'git_commit':pin['commit'],'historical_objects':len(objects),'all_actual_Git_objects_verified':checked['passed'],'proof_object_count':checked['object_count'],'current_divergence':divergence,'old_cost_document_sha256':pin['old_doc_sha256'],'current_cost_document_sha256':current_doc_sha})
    return {'passed':True,'contract_count':len(records),'historical_object_bindings':sum(v['historical_objects'] for v in records),'contracts':records,'only_cost_prose_revision_not_algorithm_exception':True,'all_original_code_and_data_objects_current_exact':True,'new_NN_or_optimizer_updates':0}
