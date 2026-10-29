"""Read consolidated Zarr metadata in one pinned, uncompressed remote tar only."""
import collections
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time
import urllib.request

ROOT = Path('/home/william/echo-view-routing/runs/a47_public_evidence_intake_v1')
REV = '7a36922880061791519a8f01553c15cad5df564f'


class Ranges(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.pos = url, size, 0
        self.cache, self.page, self.transferred, self.requests = {}, 65536, 0, 0
        self.started = time.monotonic()
        self.cache_dir = ROOT / 'ranges_attempt2'
        self.cache_dir.mkdir(exist_ok=True)

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else (self.pos + offset if whence == 1 else self.size + offset)
        if not 0 <= target <= self.size:
            raise ValueError('seek outside pinned archive')
        self.pos = target
        return self.pos

    def read(self, count=-1):
        if count < 0 or count > 5000000:
            raise ValueError('unbounded or oversized read refused')
        count = min(count, self.size - self.pos)
        out = []
        while count:
            start = self.pos // self.page * self.page
            end = min(self.size, start + self.page) - 1
            if start not in self.cache:
                if self.transferred + end - start + 1 > 80000000 or self.requests >= 1200 or time.monotonic() - self.started > 900:
                    raise RuntimeError('sample traffic/request/time limit reached')
                req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
                with urllib.request.urlopen(req, timeout=20) as r:
                    expected = f'bytes {start}-{end}/{self.size}'
                    if r.status != 206 or r.headers.get('Content-Range') != expected:
                        raise ValueError('exact Content-Range not honored')
                    b = r.read(end - start + 2)
                if len(b) != end - start + 1:
                    raise ValueError('range response length wrong')
                self.cache[start] = b
                (self.cache_dir / f'{start:012d}.bin').write_bytes(b)
                self.transferred += len(b)
                self.requests += 1
            offset = self.pos - start
            n = min(count, len(self.cache[start]) - offset)
            out.append(self.cache[start][offset:offset + n])
            self.pos += n
            count -= n
        return b''.join(out)


def run():
    selection = json.loads((ROOT / 'sample_selection.private.json').read_text())
    exam = selection['exam']
    package = json.loads((ROOT / 'package_manifest.json').read_text())
    wanted = {r['recordings/zarr_path'] + '/.zmetadata' for r in selection['recordings'] if any(p.startswith('annotations/measurement_') for p in r['recordings/array_paths'])}
    stream = Ranges(f'https://huggingface.co/datasets/Ahus-AIM/EchoXFlow/resolve/{REV}/exams/{exam}.tar', package['tar_bytes'][exam])
    try:
        assert stream.read(512)[257:262] == b'ustar', 'uncompressed USTAR preflight failed'
        stream.seek(0)
        found, members = [], 0
        with tarfile.open(fileobj=stream, mode='r:') as archive:
            for item in archive:
                members += 1
                if item.name not in wanted:
                    continue
                if not item.isfile() or item.size > 5000000:
                    raise ValueError('unexpected consolidated metadata member')
                blob = archive.extractfile(item).read()
                obj = json.loads(blob)
                sha = hashlib.sha256(blob).hexdigest()
                (ROOT / ('sample_' + sha + '.private.json')).write_bytes(blob)
                attrs = obj['metadata']['.zattrs']
                def strings(value):
                    if isinstance(value, str):
                        yield value
                    elif isinstance(value, list):
                        for x in value:
                            yield from strings(x)
                    elif isinstance(value, dict):
                        for x in value.values():
                            yield from strings(x)
                labels = sorted({s for s in strings(attrs) if any(w in s.lower() for w in ('lvot', 'aortic', 'mitral')) and len(s) < 180})
                found.append({'sha256': sha, 'bytes': len(blob), 'root_attribute_keys': sorted(attrs),
                              'clinical_label_strings': labels,
                              'array_metadata_members': sum(k.endswith('/.zarray') for k in obj['metadata'])})
                if any("lvot" in label.lower() for label in labels) or len(found) == len(wanted):
                    break
        summary = {'selected_examinations': 1, 'requested_pw_bmode_metadata': len(wanted), 'found_metadata': len(found),
                   'tar_members_visited': members, 'metadata_records': found, 'archive_compressed': False,
                   'ranges_exact': True, 'fetched_bytes': stream.transferred, 'requests': stream.requests,
                   'all_requested_found': len(found) == len(wanted)}
        (ROOT / 'sample_inventory.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
    finally:
        print(json.dumps({'sample_transferred_bytes': stream.transferred, 'sample_requests': stream.requests}), flush=True)


if __name__ == '__main__':
    start, cpu, ok = time.time(), time.process_time(), False
    try:
        run()
        ok = True
    finally:
        receipt = {'ok': ok, 'started_unix': start, 'finished_unix': time.time(), 'wall_seconds': time.time() - start,
                   'process_cpu_seconds': time.process_time() - cpu, 'gpu_hours': 0,
                   'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        (ROOT / 'sample_receipt.attempt2.json').write_text(json.dumps(receipt, indent=2))
