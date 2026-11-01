#!/usr/bin/env python3
"""Qualify pinned official OCR packages in a private nonroot prefix; no patient pixels."""
from pathlib import Path
import hashlib
import json
import os
import resource
import subprocess
import time
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/a5c_image_dependence_v1"
PACKAGES = {
    "tesseract-ocr": ("5.3.4-1build5", "2dfac382d77215aee0c3de4a2a2205505d5f2195e72e79b54ad32154fc08da77"),
    "libtesseract5": ("5.3.4-1build5", "a38438115dc203b5abc1e6a6b24eb5273ba82161a11faa2f619bf5f7622efb14"),
    "liblept5": ("1.82.0-3build4", "c2bb81d58f032b09917c0d50994cce905287a1596304a4e99681a03601456ea8"),
    "tesseract-ocr-eng": ("1:4.1.0-2", "b1996b3113c78663f4dd16cf18aa1f288b07e9624f8d4a0ddbd7d9b52a234ba7")}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    start, cpu = time.time(), time.process_time()
    prefix, packages = OUT / "ocr_tool", OUT / "ocr_packages"
    prefix.mkdir(parents=True, exist_ok=True); packages.mkdir(exist_ok=True)
    receipts = {}
    for name, (version, expected) in PACKAGES.items():
        r = subprocess.run(["apt-get", "download", name + "=" + version], cwd=packages, capture_output=True, text=True, timeout=60)
        if r.returncode:
            raise RuntimeError("pinned package acquisition failed: " + name + " " + r.stderr[-700:])
        files = [p for p in packages.glob("*.deb") if sha(p) == expected]
        assert len(files) == 1, "full package SHA mismatch: " + name
        p = files[0]
        assert sum(q.stat().st_size for q in packages.iterdir() if q.is_file()) <= 40000000
        subprocess.run(["dpkg-deb", "-x", str(p), str(prefix)], check=True, timeout=20)
        assert sum(q.stat().st_size for q in prefix.rglob("*") if q.is_file()) <= 100000000
        notice = prefix / "usr/share/doc" / name / "copyright"
        assert notice.is_file() and "License" in notice.read_text()
        receipts[name] = {"version": version, "deb_sha256": expected, "download_bytes": p.stat().st_size,
                           "copyright_sha256": sha(notice), "source": "official configured Ubuntu archive; apt metadata and full artifact SHA pinned"}
    binary = prefix / "usr/bin/tesseract"
    lang = list(prefix.rglob("eng.traineddata")); assert len(lang) == 1
    env = dict(os.environ, LD_LIBRARY_PATH=str(prefix / "usr/lib/x86_64-linux-gnu") + ":" + os.environ.get("LD_LIBRARY_PATH", ""), OMP_THREAD_LIMIT="1")
    deps = subprocess.run(["ldd", str(binary)], env=env, capture_output=True, text=True, timeout=10)
    assert deps.returncode == 0 and "not found" not in deps.stdout, "unavailable system dependencies; no global installation allowed"
    version = subprocess.run([str(binary), "--version"], env=env, capture_output=True, text=True, check=True, timeout=10)
    synthetic = Image.new("RGB", (640, 480), "white")
    try: font = ImageFont.truetype("DejaVuSans.ttf", 48)
    except OSError: font = ImageFont.load_default(size=48)
    ImageDraw.Draw(synthetic).text((120, 180), "A5C   72 BPM", fill="black", font=font)
    p = OUT / "ocr_synthetic_control.png"; synthetic.save(p)
    command = [str(binary), str(p), "stdout", "--tessdata-dir", str(lang[0].parent), "--oem", "1", "--psm", "11", "-l", "eng", "tsv"]
    control = subprocess.run(command, env=env, capture_output=True, text=True, check=True, timeout=15)
    assert "A5C" in control.stdout, "fixed synthetic tool control failed; do not tune on patient images"
    members = {str(p.relative_to(OUT)): sha(p) for p in [binary, lang[0], *[q for q in (prefix / "usr/lib/x86_64-linux-gnu").glob("*.so.*") if q.is_file() and not q.is_symlink()]]}
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {"packages": receipts, "members": members, "version": version.stdout.splitlines()[0],
               "binary_relative_to_run": str(binary.relative_to(OUT)), "language_relative_to_run": str(lang[0].relative_to(OUT)),
               "synthetic_fixed_control_pass": True, "private_or_patient_images_read": 0, "GPU_forwards": 0,
               "source_docs": "https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html",
               "copyright_scope": "Exact packaged copyright notices retained in owned prefix; includes upstream and distribution terms.",
               "wall_seconds": time.time() - start, "process_CPU_seconds": time.process_time() - cpu + usage.ru_utime + usage.ru_stime,
               "namespace_added_bytes": sum(q.stat().st_size for q in [*prefix.rglob('*'), *packages.rglob('*')] if q.is_file())}
    (OUT / "ocr_tool_manifest.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
