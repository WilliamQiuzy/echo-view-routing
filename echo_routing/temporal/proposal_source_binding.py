"""Recompute inherited posterior identity from its exact source contract, no forward."""
import hashlib,json

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate(current_sources,old_config,old_producer,old_input,records):
    if current_sources!=old_config['source']:raise ValueError('A60 source contract differs from qualified inherited A59 producer')
    expected={(s,r) for s in ['a08','jepa'] for r in old_input['roles']};seen=[]
    for row in records:
        key=(row['source'],row['role']);seen.append(key)
        wanted=digest({'source':row['source'],'source_contract':old_config['source'][row['source']],'input':old_input['roles'][row['role']]['sha256'],'frozen_objects':old_producer['objects']})
        if row['identity']!=wanted or row['input_sha256']!=old_input['roles'][row['role']]['sha256'] or not row['complete']:raise ValueError('source/cache/input/preprocessing identity does not recompute')
    if len(seen)!=16 or set(seen)!=expected:raise ValueError('exact16 complete inherited source-role identities required')
    return {'passed':True,'source_role_contracts_recomputed':len(seen),'A60_source_contracts_exactly_equal_qualified_A59':True,'frozen_models_and_preprocess_not_reexecuted':True,'N5_shape_alone_not_used_as_semantic_identity':True}
