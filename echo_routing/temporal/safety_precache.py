"""Frozen-producer identity separated from the later action-head Git seal."""
import hashlib,json
ROLES=['a61_confirmation_selection','a61_confirmation_calibration']
CLASSES=['PLAX','PSAX','A4C','A5C','SC4C']


def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def folder(source,role):
    if source not in ['a08','jepa'] or role not in ROLES:raise ValueError('exact new240 frozen-source roles required')
    return 'runs/a61_action_safety_ranking_v1/producer_precache_v1/'+source+'/'+role


def identity(cfg,inputs,freeze,source,role):
    folder(source,role)
    if freeze.get('stage_kind')!='producer_precache_geometry_only' or cfg['channel_mapping']['classes']!=CLASSES or inputs['roles'][role]['sequences']!=120:raise ValueError('frozen headless producer/class/120stream contract required')
    return digest({'source':source,'source_contract':cfg['source'][source],'class_order':CLASSES,'input':inputs['roles'][role]['sha256'],'producer_objects':freeze['objects'],'purpose':'teacher_probability_geometry_only;head_weights_and_headseal_excluded'})


def validate(manifest,cfg,inputs,freeze):
    if not manifest['passed'] or manifest['formal_updates_at_production']!=0 or not manifest['no_A61_action_heads_instantiated_or_forwarded'] or not manifest['no_outer_GT_metrics_or_utility_evaluated'] or manifest['class_order']!=CLASSES:raise ValueError('true headless production point and class mapping required')
    records=manifest['records'];keys=[(r['source'],r['role']) for r in records]
    if len(keys)!=4 or set(keys)!={(s,r) for s in ['a08','jepa'] for r in ROLES}:raise ValueError('exact4 complete source/role journals required')
    for row in records:
        s,r=row['source'],row['role']
        if row['identity']!=identity(cfg,inputs,freeze,s,r) or row['input_sha256']!=inputs['roles'][r]['sha256'] or row['path']!=folder(s,r) or not row['complete'] or row['committed_sequences']!=120:raise ValueError('immutable source/recipe/producer/journal identity differs')
    return {(r['source'],r['role']):r for r in records}


def binding(manifest,models_sha,headseal_sha,updates):
    if updates!=60000 or not models_sha or not headseal_sha or manifest['formal_updates_at_production']!=0:raise ValueError('real60000 head binding must retain original0 production point')
    return {**manifest,'source_production_stage':'producer_precache','read_only_early_producer_cache_adopted':True,'formal_updates_at_readonly_binding':updates,'models_manifest_sha256_at_binding':models_sha,'actual_Git_headseal_sha256_at_binding':headseal_sha,'new_source_forwards_during_readonly_adoption':0,'no_journal_identity_rewrite_or_rename':True}
