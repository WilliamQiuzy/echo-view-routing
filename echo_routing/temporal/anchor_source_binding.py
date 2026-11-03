"""Exact completed posterior journal bindings before frozen head readout."""
ROLES=['selection','calibration','old_stress_selection','old_stress_calibration','confirmation_selection','confirmation_calibration']


def validate(records,observed):
    expected={(s,r) for s in ['a08','jepa'] for r in ROLES}
    keys=[(r['source'],r['role']) for r in records]
    if len(keys)!=12 or set(keys)!=expected or set(observed)!=expected:
        raise ValueError('complete exact12 source-role journal bindings required')
    for record in records:
        actual=observed[(record['source'],record['role'])]
        if record['complete'] is not True or any(record[k]!=actual[k] for k in ['identity','journal_sha256','committed_sequences','input_sha256']):
            raise ValueError('posterior journal/input/complete original axis differs from frozen manifest')
    return True
