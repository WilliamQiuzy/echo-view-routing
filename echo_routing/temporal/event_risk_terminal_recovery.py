"""One explicit terminal-only recovery; executed A62 contracts remain immutable."""
import json,re
from pathlib import Path
from echo_routing.temporal.anchor_cache import sha


def admissions(pub):
    rows=[]
    for path in sorted(Path(pub).glob('*.json')):
        name=path.name
        match=re.fullmatch(r'(sources|formal|outer)(?:_refined)?_admission_v([1-9][0-9]*)\.json',name)
        if match:
            row=json.loads(path.read_text())
            if row.get('stage')!=match[1] or type(row.get('version')) is not int or row['version']!=int(match[2]) or type(row.get('all_caps_fit')) is not bool:raise ValueError('malformed actual versioned admission: '+name)
            rows.append(row)
        elif re.match(r'(sources|formal|outer)(?:_refined)?_admission_v',name):raise ValueError('unrecognized actual admission filename: '+name)
    return rows


def validate_failure(root,binding):
    root=Path(root)
    prefix='docs/results/2026-11-04_anchor_context_risk/';run='runs/a62_anchor_context_risk_v1/'
    expected={prefix+'terminal_failed_v1/'+n:run+n for n in ['collector.stderr.log','collector.stdout.log','collector_caller_intent.json']}
    expected.update({prefix+'collector_OS_exit_receipt.json':prefix+'collector_OS_exit_receipt.json',prefix+'terminal_identity.json':run+'terminal_freeze.json'})
    if binding['server_originals']!=expected or set(binding['objects'])!=set(expected) or binding['failed_caller_receipt']!=prefix+'collector_OS_exit_receipt.json' or binding['failed_intent']!=prefix+'terminal_failed_v1/collector_caller_intent.json' or binding['failed_stderr']!=prefix+'terminal_failed_v1/collector.stderr.log':raise ValueError('exact original failure evidence paths required')
    if binding['scope']!='only original pre-snapshot terminal admission-discovery KeyError recovery':raise ValueError('exact recovery scope required')
    for p,h in binding['objects'].items():
        if sha(root/p)!=h:raise ValueError('original failed terminal evidence changed: '+p)
    for public,original in binding['server_originals'].items():
        if public not in binding['objects'] or sha(root/original)!=binding['objects'][public]:raise ValueError('original failed server evidence differs from frozen public copy')
    old=json.loads((root/binding['failed_caller_receipt']).read_text());intent=json.loads((root/binding['failed_intent']).read_text())
    if old['passed'] is not False or old['observed_collector_OS_exit_code']!=1 or old['collector_exit_directly_waited'] is not True or old['resources_sha256'] is not None or old['terminal_exit_receipt_sha256'] is not None or old['collector_pid']!=intent['collector_pid'] or old['caller_pid']!=intent['caller_pid']:raise ValueError('only directly waited original pre-snapshot failure may recover')
    error=(root/binding['failed_stderr']).read_text()
    if "KeyError: 'all_caps_fit'" not in error:raise ValueError('different failure requires independent review')
    return old


def verify():
    from scripts.event_risk_common import ROOT,RUN,PUB,CFG,read,verify_pin
    from echo_routing.temporal.event_risk_contracts import source_closure
    from echo_routing.temporal.safety_git import verify as gitverify
    c=read(RUN/'terminal_recovery/v2_freeze.json');cfg=read(CFG)
    entries=['scripts/close_event_risk_terminal_recovery.py','scripts/wait_event_risk_terminal_recovery.py']
    inputs=[str(CFG.relative_to(ROOT)),cfg['card']['path'],cfg['authorization']['path'],cfg['basis']['path'],str((PUB/'terminal_failure_binding.json').relative_to(ROOT)),str((PUB/'local_cpu_receipts_terminal_recovery.json').relative_to(ROOT))]
    binding=read(PUB/'terminal_failure_binding.json');old=read(PUB/'terminal_identity.json');required=set(source_closure(ROOT,entries+inputs+list(binding['objects'])+list(old['objects'])))
    if c['stage']!='terminal_recovery_v2' or c['fixed_deadline_epoch']!=1793867561 or not required<=set(c['objects']):raise ValueError('complete exact recovery source/input closure required')
    for p,h in c['objects'].items():
        if sha(ROOT/p)!=h:raise ValueError('recovery frozen current bytes changed: '+p)
    gitverify(read(verify_pin(c['actual_Git_object_proof'])),c['git_commit'],c['objects']);validate_failure(ROOT,binding)
    return c


def receipts():
    from scripts.event_risk_common import ROOT,RUN,PUB,read
    fees=read(PUB/'local_cpu_receipts_terminal_recovery.json')
    for p,h in fees['file_pins'].items():
        if sha(ROOT/p)!=h:raise ValueError('recovery fee raw receipt changed')
    failure=validate_failure(ROOT,read(PUB/'terminal_failure_binding.json'));fee=read(PUB/'root_failed_terminal_CPU_fee.json');expected=failure['collector_waited_CPU_seconds']+failure['caller_SELF_CPU_seconds']
    if abs(fee['CPU_SELF_plus_waitedchildren_seconds']-expected)>1e-9 or fee['raw_sha256']!=sha(PUB/'collector_OS_exit_receipt.json'):raise ValueError('actual failed collector/caller charge must match directly waited raw CPU')
    charges=[r for r in fees['receipts'] if r.get('source_receipt_sha256')==sha(PUB/'root_failed_terminal_CPU_fee.json')]
    if len(charges)!=1 or abs(charges[0]['measured_CPU_core_seconds']-expected)>1e-9:raise ValueError('failed terminal attempt must be charged exactly once')
    base=read(PUB/'local_cpu_receipts.json');byid={r['receipt_id']:r for r in fees['receipts']}
    if any(byid.get(r['receipt_id'])!=r for r in base['receipts']):raise ValueError('original296 fee records must remain byte-semantic exact in recovery ledger')
    return [json.loads(x) for x in (RUN/'processes.jsonl').read_text().splitlines()]+fees['receipts']+[read(p) for p in PUB.glob('entry_CPU_*.json')]
