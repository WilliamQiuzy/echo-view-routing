"""Pure A63 negative-state reporting and explicit, read-only publication snapshots.

These helpers do not run a collector, validate private science, write a report,
complete an OS wait, or integrate main. Caller and root closure remain separate.
"""
import copy
import hashlib
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath

from echo_routing.temporal.selected_git_integration import allowed
from echo_routing.temporal.selected_risk_budget import totals
from echo_routing.temporal.selected_risk_figures import (
    SEEDS, FLOORS, CONCENTRATION, MEMBER_FIELDS, UNION_TYPES,
)

COUNTERS = ('teacher_payloads', 'A08_windows', 'JEPA_chunks', 'teacher_forwards',
            'aux_optimizer_updates', 'formal_optimizer_updates', 'C0_C1_instances')
PASS_BRANCH = dict(C0_C1_models=4, formal_optimizer_updates=24000, calibrated_taus=4,
                   outer_rows=238, inner_rows=34, paired_comparisons=140,
                   paired_draws_per_comparison=5000, support_records=56,
                   native_records=34, regression_records=32, confirm_teacher_payloads=480)
PUB = 'docs/results/2026-11-05_selected_action_risk/'


def _count(v, unknown=False):
    if unknown and v is None:
        return None
    if type(v) is not int or v < 0:
        raise ValueError('exact nonnegative integer counts required; None is not zero')
    return v


def _coverage(value):
    """Validate the already-produced aggregate gate, never alter a minimum."""
    try:
        if (not isinstance(value, dict) or value['passed'] is not False or
                set(value['seeds']) != {str(s) for s in SEEDS} or
                value['source_isolation']['passed'] is not True or
                _count(value['aux_optimizer_updates']) != 48000 or
                _count(value['formal_optimizer_updates']) != 0):
            raise ValueError('complete isolated coverage-negative Stage-S proof required')
        failed = []
        for seed in SEEDS:
            row = value['seeds'][str(seed)]; checks = row['checks']
            if type(row['passed']) is not bool or set(checks) != set(FLOORS) | set(CONCENTRATION):
                raise ValueError('both seeds and all original23 checks required')
            for key, (minimum, group, field) in FLOORS.items():
                c = checks[key]; actual = _count(c['actual'])
                if (set(c) != {'actual', 'minimum', 'passed'} or _count(c['minimum']) != minimum or
                        type(c['passed']) is not bool or c['passed'] != (actual >= minimum) or
                        actual != _count(row[group][field])):
                    raise ValueError('count/floor/verdict must bind its original aggregate')
            for key in CONCENTRATION:
                c = checks[key]; actual = c['actual']
                if (set(c) != {'actual', 'maximum', 'passed'} or isinstance(actual, bool) or
                        not isinstance(actual, (int, float)) or not math.isfinite(actual) or
                        not 0 <= actual <= 1 or type(c['maximum']) not in (int, float) or
                        c['maximum'] != .2 or type(c['passed']) is not bool or
                        c['passed'] != (actual <= .2+1e-15) or actual != row[key]):
                    raise ValueError('original concentration limit and aggregate required')
            keys = [k for k, c in checks.items() if not c['passed']]
            if (not isinstance(row['failed_checks'], list) or len(row['failed_checks']) != len(keys) or
                    set(row['failed_checks']) != set(keys) or row['passed'] != (not keys)):
                raise ValueError('retain exactly every failed coverage check')
            for group in ('D', 'X_only'):
                for field in MEMBER_FIELDS:
                    _count(row[group][field])
            for field in UNION_TYPES:
                _count(row['union'][field])
            classes = row['union']['replacement_class_signatures']
            if set(classes) != {str(i) for i in range(5)}:
                raise ValueError('all five replacement classes, including zeros, required')
            for n in classes.values():
                _count(n)
            for name in ('native_D_danger_actions', 'native_X_danger_actions'):
                _count(row[name])
            failed.extend(dict(seed=seed, check=key, **copy.deepcopy(checks[key])) for key in keys)
        if not failed or all(value['seeds'][str(s)]['passed'] for s in SEEDS):
            raise ValueError('negative mode requires an actual recorded coverage failure')
        return failed
    except (KeyError, TypeError) as exc:
        raise ValueError('complete public coverage schema required') from exc


def _costs(actual, workers):
    try:
        if not isinstance(actual, dict) or not isinstance(actual['receipt_records'], list) or not isinstance(workers, list):
            raise ValueError('canonical actual costs and explicit OS receipt list required')
        rows = actual['receipt_records']; expected = totals(rows)
        if any(k not in actual or actual[k] != v for k, v in expected.items()):
            raise ValueError('cost and OS aggregate differs from exact canonical receipts')
        by_id = {r['receipt_id']: r for r in rows}; seen = set()
        for row in workers:
            key = row['receipt_id']
            if key in seen or key not in by_id or row != by_id[key]:
                raise ValueError('provided worker OS receipt must bind one exact cost record')
            seen.add(key)
        for row in rows:
            if row['observed_worker_exit_code'] is not None and type(row['observed_worker_exit_code']) is not int:
                raise ValueError('integer actual OS status or explicit unknown required')
        summary = dict(zero_receipts=[r['receipt_id'] for r in rows if r['observed_worker_exit_code'] == 0],
                       nonzero_receipts=[r['receipt_id'] for r in rows if r['observed_worker_exit_code'] not in (None, 0)],
                       unknown_receipts=[r['receipt_id'] for r in rows if r['observed_worker_exit_code'] is None],
                       actual_cost_record_count=len(rows), provided_worker_receipt_count=len(workers),
                       complete_inventory_independently_proven=False)
        allzero = (all(r['observed_worker_exit_code'] == 0 and r.get('child_exit_code_directly_waited') is True
                       for r in workers) if workers else None)
        return summary, allzero
    except (KeyError, TypeError) as exc:
        raise ValueError('canonical actual costs/OS receipts required') from exc


def terminal_state(mode, coverage, aux_audit, source_manifest, actual_costs, worker_receipts):
    """Return a pending-publication negative state; never an overall closed state.

    The source manifest is an aggregate caller projection. teacher_payloads means
    complete committed payloads; attempted forward units stay separate/unknown.
    Auxiliary audit input uses the qualified public physical_models/fullstate_count.
    """
    if mode not in ('coverage-negative', 'resource-stopped'):
        raise ValueError('only explicit coverage-negative or resource-stopped modes supported')
    if source_manifest is not None and not isinstance(source_manifest, dict):
        raise ValueError('aggregate source manifest or explicit unknown required')
    counters = {k: _count((source_manifest or {}).get(k), unknown=True) for k in COUNTERS}
    if counters['formal_optimizer_updates'] not in (None, 0) or counters['C0_C1_instances'] not in (None, 0):
        raise ValueError('these pre-formal negative modes cannot hide C0/C1 execution')
    if counters['aux_optimizer_updates'] is not None and counters['aux_optimizer_updates'] > 48000:
        raise ValueError('unexplained auxiliary updates exceed this mode scope')
    complete = False
    if aux_audit is not None:
        try:
            if type(aux_audit['passed']) is not bool:
                raise ValueError('explicit auxiliary audit verdict required')
            quantities = [_count(aux_audit[k]) for k in ('aux_optimizer_updates', 'physical_models', 'fullstate_count')]
            if any(a > b for a, b in zip(quantities, (48000, 8, 32))):
                raise ValueError('auxiliary audit exceeds the frozen model/update/state scope')
            if counters['aux_optimizer_updates'] is not None and quantities[0] != counters['aux_optimizer_updates']:
                raise ValueError('source actual auxiliary count differs from audit')
            complete = aux_audit['passed'] and quantities == [48000, 8, 32]
            if aux_audit['passed'] and not complete:
                raise ValueError('passed auxiliary seal requires48000/8/32')
        except (KeyError, TypeError) as exc:
            raise ValueError('qualified auxiliary audit public schema required') from exc
    if mode == 'coverage-negative':
        failed = _coverage(coverage)
        if (not complete or counters['aux_optimizer_updates'] != 48000 or
                counters['formal_optimizer_updates'] != 0 or counters['C0_C1_instances'] != 0 or
                counters['teacher_payloads'] != 6000):
            raise ValueError('coverage-negative requires audited full supply and proven zero formal execution')
    else:
        if coverage is not None:
            raise ValueError('pre-coverage resource stop must retain coverage as unassessed')
        failed = []
    os_summary, allzero = _costs(actual_costs, worker_receipts)
    unrun = {k: dict(status='not_run' if mode == 'coverage-negative' else 'not_established',
                    produced=0 if mode == 'coverage-negative' else None, expected_if_admitted=v)
             for k, v in PASS_BRANCH.items()}
    if mode == 'resource-stopped':
        for key, counter in [('formal_optimizer_updates', 'formal_optimizer_updates'), ('C0_C1_models', 'C0_C1_instances')]:
            if counters[counter] == 0:
                unrun[key].update(status='not_run', produced=0)
    return dict(mode=mode, scope='A63 supply qualification; C0/C1 sampling comparison not trained',
                coverage_assessed=mode == 'coverage-negative', coverage_passed=False if failed else None,
                failed_checks=failed, coverage=copy.deepcopy(coverage), auxiliary_audit_complete=complete,
                auxiliary_audit=copy.deepcopy(aux_audit), actual_counters=counters,
                actual_costs=copy.deepcopy(actual_costs), OS_summary=os_summary,
                provided_worker_receipt_ids=[r['receipt_id'] for r in worker_receipts],
                provided_worker_receipts_all_OS0=allzero, not_run=unrun,
                C0_C1_sampling_hypothesis_tested=False, goal_met=False, overall_complete=False,
                publication_pending=['independent_caller_OS_and_cost_tail', 'root_protected_main_integration'],
                automatically_start_A64=False)


def render_negative_report(state):
    """Produce Markdown only; no file writes or additional completion claim."""
    try:
        if (state['mode'] not in ('coverage-negative', 'resource-stopped') or
                state['goal_met'] is not False or state['overall_complete'] is not False or
                state['C0_C1_sampling_hypothesis_tested'] is not False):
            raise ValueError('only a pending negative state may use this report')
        rows = {r['receipt_id']: r for r in state['actual_costs']['receipt_records']}
        rebuilt = terminal_state(state['mode'], state['coverage'], state['auxiliary_audit'],
                                 state['actual_counters'], state['actual_costs'],
                                 [rows[key] for key in state['provided_worker_receipt_ids']])
        if state != rebuilt:
            raise ValueError('report state must bind all original counters, failures and OS records')
        actual = state['actual_counters']; negative = state['mode'] == 'coverage-negative'
        def show(v):
            return 'unknown（未建立实际计数）' if v is None else str(v)
        lines = ['# A63：'+('供样覆盖负结果' if negative else '资源门槛停止，覆盖未评估'), '',
                 '研究问题：来源隔离供样能否覆盖危险的已选中动作，并支持后续固定 S-L/N 的 C0/C1 采样比较。', '',
                 '实际尝试：'+('完整 Stage S 辅助训练和独立审计已由传入证据建立。' if state['auxiliary_audit_complete'] else '仅记录传入的实际进度，未把计划数补成完成数。')]
        lines += [f'- {key}: {show(actual[key])}' for key in COUNTERS]
        lines += ['', '结果：'+('覆盖不通过；以下列出全部失败项，原始两 seed 表及通过项一并保留。' if negative else 'coverage 未评估；这是资源停止，不是供样覆盖失败。')]
        for row in state['failed_checks']:
            limit = ('minimum', row['minimum']) if 'minimum' in row else ('maximum', row['maximum'])
            lines.append(f"- seed {row['seed']} / {row['check']}: actual={row['actual']}, {limit[0]}={limit[1]}, passed=false")
        lines += ['', '后续阶段：']
        lines += [f"- {key}: {value['status']}; produced={show(value['produced'])}; 通过准入后的计划数={value['expected_if_admitted']}" for key, value in state['not_run'].items()]
        lines += ['', '执行边界：非零 OS 收据='+str(state['OS_summary']['nonzero_receipts'])+'；OS unknown 收据='+str(state['OS_summary']['unknown_receipts'])+'。不将行政/prelaunch、失败准入或 collector 等同为全部模型 OS0。', '',
                  'C0/C1 采样假设未被训练检验；没有外层效果、临床安全或整个风险模型族被否定的结论。源会话不是已确认患者，辅助选择器的来源隔离不代表旧视觉模型无历史暴露。', '',
                  'overall_complete=false：独立 caller OS/费用尾和 root 对 main 的保护集成尚须另行完成。不自动启动 A64。', '']
        return '\n'.join(lines)
    except (KeyError, TypeError) as exc:
        raise ValueError('complete terminal state required') from exc


def _path(value):
    if not isinstance(value, str) or not allowed(value) or str(PurePosixPath(value)) != value:
        raise ValueError('only explicit canonical authorized new A63 paths allowed')
    return value


def _read(root_fd, relative):
    directory = os.dup(root_fd); file_fd = None
    try:
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = nxt
        file_fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(file_fd)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('manifest inputs must be ordinary files')
        digest = hashlib.sha256(); size = 0
        while True:
            chunk = os.read(file_fd, 1024*1024)
            if not chunk:
                break
            digest.update(chunk); size += len(chunk)
        after = os.fstat(file_fd)
        if size != before.st_size or any(getattr(before, k) != getattr(after, k) for k in ('st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')):
            raise ValueError('manifest input changed during the read-only snapshot')
        return dict(path=relative, bytes=size, sha256=digest.hexdigest())
    except OSError as exc:
        raise ValueError('manifest input missing, symlinked or unreadable') from exc
    finally:
        if file_fd is not None:
            os.close(file_fd)
        os.close(directory)


def _self_path(root_fd, relative):
    """Check existing self-path components without opening/reading its bytes."""
    directory = os.dup(root_fd)
    try:
        parts = PurePosixPath(relative).parts
        for part in parts[:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = nxt
        value = os.stat(parts[-1], dir_fd=directory, follow_symlinks=False)
        if not stat.S_ISREG(value.st_mode):
            raise ValueError('existing excluded manifest must be an ordinary nonsymlink file')
    except FileNotFoundError:
        # A future manifest or its as-yet unwritten parent is allowed; no mkdir.
        return
    except OSError as exc:
        raise ValueError('excluded self path has a symlink or unreadable ancestor') from exc
    finally:
        os.close(directory)


def exact_manifest(root, paths, source_commit, self_relative_path):
    """Hash only explicit strings or exact {path,sha256,bytes} pins, no glob/write.

    The declared source commit is not a substitute for caller Git-object proof.
    This snapshot is not a concurrent publication transaction; recheck at apply.
    """
    if not isinstance(source_commit, str) or not re.fullmatch('[0-9a-f]{40}', source_commit):
        raise ValueError('full declared Git commit identity required')
    own = _path(self_relative_path)
    if not own.startswith(PUB) or not own.endswith('.json') or 'manifest' not in PurePosixPath(own).name.lower():
        raise ValueError('self exclusion must name the new public JSON manifest')
    if not isinstance(paths, (list, tuple)) or not paths:
        raise ValueError('nonempty explicit path list required')
    entries = []; seen = set()
    for item in paths:
        if isinstance(item, dict):
            if set(item) != {'path', 'sha256', 'bytes'} or not isinstance(item['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', item['sha256']):
                raise ValueError('complete exact expected pin required')
            _count(item['bytes']); path = _path(item['path']); expected = item
        else:
            path = _path(item); expected = None
        if path in seen:
            raise ValueError('duplicate explicit manifest path')
        seen.add(path)
        if path != own:
            entries.append((path, expected))
    if not entries:
        raise ValueError('self exclusion cannot establish an empty publication')
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError('ordinary root directory required')
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _self_path(fd, own)
        files = []
        for path, expected in sorted(entries):
            pin = _read(fd, path)
            if expected is not None and pin != expected:
                raise ValueError('explicit expected publication bytes/SHA differ')
            files.append(pin)
    finally:
        os.close(fd)
    return dict(schema='A63_explicit_public_manifest_v1', source_commit=source_commit,
                files=files, file_count=len(files), total_bytes=sum(x['bytes'] for x in files),
                self_excluded=own, auto_glob=False, Git_objects_independently_verified=False,
                overall_complete=False, concurrent_apply_reverification_required=True)
