"""Follow checksum-valid cached TAR offsets to the exact fixed schema target."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tarfile
import time

ROOT = Path('/home/william/echo-view-routing/runs/a47_public_evidence_intake_v1')
spec = importlib.util.spec_from_file_location('ranges', ROOT / 'inspect_sample.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def main():
    sel = json.loads((ROOT / 'sample_selection.private.json').read_text())
    chosen = [r for r in sel['recordings'] if any(p.startswith('annotations/measurement_') for p in r['recordings/array_paths'])]
    target = min(r['recordings/zarr_path'] for r in chosen)
    wanted = target + '/.zmetadata'
    rid = target.rsplit('/', 1)[-1][:-5]
    exam = sel['exam']
    pack = json.loads((ROOT / 'package_manifest.json').read_text())
    stream = mod.Ranges(f'https://huggingface.co/datasets/Ahus-AIM/EchoXFlow/resolve/{mod.REV}/exams/{exam}.tar', pack['tar_bytes'][exam])
    for directory in ['ranges_attempt2', 'ranges_indexed']:
        for p in (ROOT / directory).glob('*.bin'):
            if p.stat().st_size == stream.page:
                stream.cache[int(p.stem)] = p.read_bytes()
    stream.cache_dir = ROOT / 'ranges_target'
    stream.cache_dir.mkdir(exist_ok=True)
    start = 0
    for offset, block in stream.cache.items():
        for k in range(0, len(block) - 511, 512):
            buf = block[k:k + 512]
            if buf[257:262] != b'ustar':
                continue
            try:
                t = tarfile.TarInfo.frombuf(buf, 'utf-8', 'strict')
            except (tarfile.TarError, UnicodeError, ValueError):
                continue
            m = re.search(r'(recording_[0-9a-f]+)\.zarr(?:/|$)', t.name)
            if m and m[1] < rid:
                start = max(start, offset + k + 512 + (t.size + 511) // 512 * 512)
    pos = start
    for visited in range(3000):
        if stream.transferred + 2 * stream.page > 10000000 or stream.requests >= 120:
            raise RuntimeError('final schema lookup extra10MB/120request cap')
        stream.seek(pos)
        header = stream.read(512)
        t = tarfile.TarInfo.frombuf(header, 'utf-8', 'strict')
        if t.name == wanted:
            assert t.isfile() and t.size <= 1000000
            if stream.transferred + t.size + 2 * stream.page > 10000000:
                raise RuntimeError('metadata member would exceed final lookup traffic cap')
            blob = stream.read(t.size)
            q = json.loads(blob)
            sha = hashlib.sha256(blob).hexdigest()
            (ROOT / f'sample_{sha}.private.json').write_bytes(blob)
            (ROOT / 'target_selection.private.json').write_text(json.dumps({'target': target, 'offset': pos, 'sha256': sha}))
            attrs = q['metadata']['.zattrs']
            manifest = attrs['recording_manifest']
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
            out = {'exact_member_verified': True, 'metadata_sha256': sha, 'metadata_bytes': len(blob),
                   'extra_fetched_bytes': stream.transferred, 'extra_requests': stream.requests,
                   'headers_followed': visited + 1, 'clinical_label_strings': labels,
                   'manifest_keys': sorted(manifest), 'sampling_gate_keys': sorted(manifest.get('sampling_gate_metadata', {})),
                   'annotation_count': len(manifest.get('annotations', []))}
            (ROOT / 'sample_inventory.target.json').write_text(json.dumps(out, indent=2))
            print(json.dumps(out), flush=True)
            return
        pos += 512 + (t.size + 511) // 512 * 512
    raise RuntimeError('bounded target member lookup not completed')


if __name__ == '__main__':
    wall, cpu, ok = time.time(), time.process_time(), False
    try:
        main()
        ok = True
    finally:
        rec = {'ok': ok, 'started_unix': wall, 'finished_unix': time.time(), 'wall_seconds': time.time() - wall,
               'process_cpu_seconds': time.process_time() - cpu, 'gpu_hours': 0,
               'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (ROOT / 'sample_receipt.target.json').write_text(json.dumps(rec, indent=2))
