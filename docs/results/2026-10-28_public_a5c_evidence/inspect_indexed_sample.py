"""Reuse prior range pages; locate one exact metadata member in ordered USTAR."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tarfile
import time

ROOT = Path('/home/william/echo-view-routing/runs/a47_public_evidence_intake_v1')
spec = importlib.util.spec_from_file_location('range_reader', ROOT / 'inspect_sample.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main():
    selection = json.loads((ROOT / 'sample_selection.private.json').read_text())
    records = [r for r in selection['recordings'] if any(p.startswith('annotations/measurement_') for p in r['recordings/array_paths'])]
    target = min(r['recordings/zarr_path'] for r in records)
    package = json.loads((ROOT / 'package_manifest.json').read_text())
    exam = selection['exam']
    stream = mod.Ranges(f'https://huggingface.co/datasets/Ahus-AIM/EchoXFlow/resolve/{mod.REV}/exams/{exam}.tar', package['tar_bytes'][exam])
    stream.cache = {int(p.stem): p.read_bytes() for p in (ROOT / 'ranges_attempt2').glob('*.bin') if p.stat().st_size == stream.page}
    stream.cache_dir = ROOT / 'ranges_indexed'
    stream.cache_dir.mkdir(exist_ok=True)
    index = json.loads((ROOT / 'tar_header_index.private.json').read_text())
    # Prior indexed headers established local monotonic recording order. Every hit
    # below must still match the exact requested member; ordering cannot fake a hit.
    lo, hi = 0, stream.size - 512
    target_id = target.rsplit('/', 1)[-1][:-5]
    for r in index:
        if r['recording'] < target_id:
            lo = max(lo, r['offset'] + 512 + (r['size'] + 511) // 512 * 512)
        elif r['recording'] > target_id:
            hi = min(hi, r['offset'])

    def read_at(pos, n):
        if stream.transferred + n + stream.page > 10000000 or stream.requests >= 120:
            raise RuntimeError('indexed sample extra10MB/120request limit')
        stream.seek(pos)
        return stream.read(n)

    def headers(pos):
        b = read_at(pos, min(stream.page, stream.size - pos))
        found = []
        for k in range(0, len(b) - 511, 512):
            if b[k + 257:k + 262] != b'ustar':
                continue
            try:
                t = tarfile.TarInfo.frombuf(b[k:k + 512], 'utf-8', 'strict')
            except (tarfile.TarError, UnicodeError, ValueError):
                continue
            m = re.search(r'(recording_[0-9a-f]+)\.zarr(?:/|$)', t.name)
            if m:
                found.append((pos + k, m[1], t))
        return found

    for iteration in range(36):
        if hi - lo <= 2 * stream.page:
            break
        mid = ((lo + hi) // 2 // stream.page) * stream.page
        hs = []
        for j in range(16):
            pos = mid + j * stream.page
            if pos >= hi:
                break
            hs = [h for h in headers(pos) if lo <= h[0] < hi]
            if hs:
                break
        if not hs:
            raise RuntimeError('no bounded valid header around search pivot')
        offset, rid, t = hs[0]
        if rid < target_id:
            lo = offset + 512 + (t.size + 511) // 512 * 512
        else:
            hi = offset
    hits = []
    for pos in range(lo // stream.page * stream.page, min(stream.size, hi + 3 * stream.page), stream.page):
        hits += headers(pos)
    exact = [(p, t) for p, _, t in hits if t.name == target + '/.zmetadata']
    if not exact:
        raise RuntimeError('exact target consolidated metadata not located')
    offset, t = exact[0]
    assert t.isfile() and t.size < 5000000
    blob = read_at(offset + 512, t.size)
    q = json.loads(blob)
    sha = hashlib.sha256(blob).hexdigest()
    (ROOT / f'sample_{sha}.private.json').write_bytes(blob)
    (ROOT / 'indexed_selection.private.json').write_text(json.dumps({'target': target, 'offset': offset, 'sha256': sha}))
    attrs = q['metadata']['.zattrs']
    def strings(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from strings(v)
        elif isinstance(x, list):
            for v in x:
                yield from strings(v)
    labels = sorted({s for s in strings(q) if any(w in s.lower() for w in ('lvot', 'aortic', 'mitral')) and len(s) < 180})
    summary = {'metadata_sha256': sha, 'metadata_bytes': len(blob), 'extra_fetched_bytes': stream.transferred,
               'extra_requests': stream.requests, 'exact_member_verified': True,
               'clinical_label_strings': labels, 'root_keys': sorted(attrs),
               'manifest_keys': sorted(attrs.get('recording_manifest', {}))}
    (ROOT / 'sample_inventory.indexed.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    start, cpu, ok = time.time(), time.process_time(), False
    try:
        main()
        ok = True
    finally:
        rec = {'ok': ok, 'started_unix': start, 'finished_unix': time.time(), 'wall_seconds': time.time() - start,
               'process_cpu_seconds': time.process_time() - cpu, 'gpu_hours': 0,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (ROOT / 'sample_receipt.indexed.json').write_text(json.dumps(rec, indent=2))
