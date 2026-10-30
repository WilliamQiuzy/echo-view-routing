"""A50 fixed source qualification, CPU only; no model or EV9V inputs."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.temporal.clinical_acquisition_source import VerifiedArchive

OUT = ROOT / "runs/a50_clinical_acquisition_v1"
REV = "7a36922880061791519a8f01553c15cad5df564f"
REPO = "Ahus-AIM/EchoXFlow"
SAMPLE = "d4e9157cddbf1fcc2341e1d8965f5315c0cabfe66d848d3fe118424141ee3fa0"
CARD = "docs/superpowers/plans/2026-10-30-clinical-acquisition-evidence.md"
CODE = ["scripts/qualify_clinical_acquisition.py", "echo_routing/temporal/clinical_acquisition_source.py"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def write(name, value):
    path = OUT / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def freeze():
    path = ROOT / f"runs/a47_public_evidence_intake_v1/sample_{SAMPLE}.private.json"
    assert sha(path) == SAMPLE
    attrs = json.loads(path.read_text())["metadata"][".zattrs"]
    exam, record = attrs["exam_id"], attrs["recording_id"]
    split = ROOT / "runs/a49_lvot_evidence_qualification_v1/splits.csv"
    folds = {r["exam_id"]: int(r["five_fold_split"]) for r in csv.DictReader(split.open())}
    assert folds[exam] in [1, 2, 3, 4]
    rel = f"exams/{exam}.tar"
    request = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{REPO}/paths-info/{REV}",
        data=json.dumps({"paths": [rel], "expand": False}).encode(),
        headers={"Content-Type": "application/json"})
    objects = json.load(urllib.request.urlopen(request, timeout=25))
    assert len(objects) == 1 and objects[0]["path"] == rel
    published = objects[0]
    assert published["size"] == 598_118_400 and len(published["lfs"]["oid"]) == 64
    manifest = {"repo": REPO, "revision": REV, "exam": exam, "anchor_record": record,
                "provider_fold": folds[exam], "publisher": published,
                "old_sample_metadata_sha256": SAMPLE,
                "card_sha256": sha(ROOT / CARD),
                "code_sha256": {p: sha(ROOT / p) for p in CODE},
                "traffic_cap": 1_000_000_000, "peak_new_cap": 750_000_000,
                "retained_sample_cap": 20_000_000, "frozen_unix": time.time()}
    assert not (OUT / "source_selection.private.json").exists()
    write("source_selection.private.json", manifest)
    summary = {"phase": "source_selection_frozen_before_acquisition", "repo": REPO,
               "revision": REV, "archives": 1, "archive_bytes": published["size"],
               "contract_exemplar_fit_only": True, "provider_fold": folds[exam],
               "selection_sha256": sha(OUT / "source_selection.private.json"),
               "card_sha256": manifest["card_sha256"], "code_sha256": manifest["code_sha256"],
               "model_forwards": 0, "GPU_hours": 0}
    write("source_freeze.json", summary)
    return summary


def inventory(commit):
    assert commit
    selected = json.loads((OUT / "source_selection.private.json").read_text())
    assert selected["card_sha256"] == sha(ROOT / CARD)
    assert all(sha(ROOT / p) == h for p, h in selected["code_sha256"].items())
    archive = OUT / "published_sample.tar"
    assert not archive.exists(), "do not repeat a completed acquisition"
    temporary = OUT / "published_sample.tar.part"
    assert not temporary.exists(), "preserve and audit interrupted source attempt"
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/" + selected["publisher"]["path"]
    transferred = 0
    with urllib.request.urlopen(url, timeout=30) as response, temporary.open("xb") as dest:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block: break
            transferred += len(block)
            assert transferred <= selected["traffic_cap"]
            dest.write(block)
    temporary.replace(archive)
    assert archive.stat().st_size < selected["peak_new_cap"]
    rows, labels = [], collections.Counter()
    with VerifiedArchive(archive, selected["publisher"]) as source:
        assert source.proof["size"] == selected["publisher"]["size"]
        metadata_bytes = 0
        for name in sorted(source.members):
            if not re.fullmatch(r"exams/[^/]+/[^/]+\.zarr/\.zattrs", name): continue
            member = source.read(name, max_bytes=1_000_000)
            metadata_bytes += member["bytes"]
            assert metadata_bytes <= selected["retained_sample_cap"]
            attrs = json.loads(member["body"])
            if set(attrs.get("content_types", [])) != {"2d_brightness_mode", "1d_pulsed_wave_doppler"}: continue
            if attrs.get("frame_counts_by_content_type", {}).get("2d_brightness_mode") != 1: continue
            items = []
            for a in attrs["recording_manifest"].get("annotations", []):
                if a.get("kind") != "measurement" or a.get("target", {}).get("semantic_id") != "pw_doppler": continue
                links = a.get("links", {})
                key = (links.get("measurement_kind"), links.get("measure_id"), links.get("init_from_cad"),
                       bool(links.get("source_entry_ref")), a.get("target_data", {}).get("zarr_path"))
                labels[key] += 1
                items.append({"measurement_kind": key[0], "measure_id": key[1], "init_from_cad": key[2],
                              "has_source_entry": key[3], "target_data": key[4]})
            rows.append({"metadata_member": name, "metadata_sha256": member["sha256"],
                         "attrs": attrs, "clinical_measurements": items})
        write("metadata_inventory.private.json", rows)
        summary = {"phase": "metadata_inventory_only", "source_commit": commit,
                   "publisher_archive_sha_verified": True, "publisher_archive_bytes_verified": source.proof["size"],
                   "transferred_bytes": transferred, "metadata_bytes_read": metadata_bytes,
                   "static_Bmode_with_sole_PW_candidates": len(rows),
                   "clinical_measurement_inventory": [{"measurement_kind": k[0], "measure_id": k[1],
                       "init_from_cad": k[2], "has_source_entry": k[3], "target_data": k[4], "count": v}
                       for k, v in sorted(labels.items(), key=lambda kv: str(kv[0]))],
                   "selection_sha256": sha(OUT / "source_selection.private.json"),
                   "private_inventory_sha256": sha(OUT / "metadata_inventory.private.json"),
                   "native_image_decodes": 0, "spectral_value_decodes": 0, "model_forwards": 0, "GPU_hours": 0}
        write("metadata_inventory.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["freeze", "inventory"], required=True)
    parser.add_argument("--commit", default="")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    wall, cpu, ok = time.time(), time.process_time(), False
    try:
        result = freeze() if args.stage == "freeze" else inventory(args.commit)
        ok = True
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        write(f"{args.stage}_failure.private.json", {"type": type(error).__name__, "message": str(error)})
        print(json.dumps({"ok": False, "error_type": type(error).__name__, "model_forwards": 0}), flush=True)
        raise
    finally:
        write(f"{args.stage}_receipt.json", {"ok": ok, "started_unix": wall, "finished_unix": time.time(),
            "wall_seconds": time.time() - wall, "process_CPU_seconds": time.process_time() - cpu,
            "GPU_hours": 0, "source_code_sha256": sha(__file__),
            "namespace_bytes": sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())})


if __name__ == "__main__": main()
