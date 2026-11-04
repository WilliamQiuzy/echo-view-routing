"""Fail-closed terminal historical-version bridge, no inferred success flags."""
def validate_history(value,identity,identity_sha,objects_sha,current_divergences):
    names=['preparation_freeze.json','training_prepare_freeze.json']
    if (value.get('passed') is not True or value.get('all_actual_Git_objects_verified') is not True
        or sorted(value.get('validated_freezes',[]))!=sorted(names)
        or value.get('historical_source_git_commit')!=identity['git_commit']
        or value.get('object_count')!=len(identity['objects']) or not identity['objects']
        or value.get('preparation_identity_v3_sha256')!=identity_sha or value.get('objects_SHA256')!=objects_sha
        or sorted(value.get('current_divergent_paths',[]))!=sorted(current_divergences)
        or set(current_divergences)-{'scripts/supervise_jepa_proposals.py'}):
        raise ValueError('historical preparation source verification failed or incomplete/mismatched')
    return True
