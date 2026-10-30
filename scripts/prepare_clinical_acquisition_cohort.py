"""A50 fixed metadata-selected paired-purpose image intake. CPU only; no models."""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.temporal.clinical_acquisition_source import (
    VerifiedArchive, acquisition_purpose, fixed_exam_pool, validate_static)

OUT = ROOT / "runs/a50_clinical_acquisition_v1"
REV = "7a36922880061791519a8f01553c15cad5df564f"
REPO = "Ahus-AIM/EchoXFlow"
CARD = "docs/superpowers/plans/2026-10-30-clinical-acquisition-cohort.md"
LABELS = "docs/superpowers/plans/2026-10-30-clinical-acquisition-label-contract.json"
CATALOG = "runs/a47_public_evidence_intake_v1/croissant.json"
SPLITS = "runs/a49_lvot_evidence_qualification_v1/splits.csv"
CODE = ["scripts/prepare_clinical_acquisition_cohort.py", "echo_routing/temporal/clinical_acquisition_source.py"]
CATALOG_SHA = "8e90e77079cce8804073ff281558d6c4b5f343117ab042e8791427cd0e369454"
DEFINITION = "Recordings that stored a physical acquisition already published by another recording of the same exam; their annotation documents live in that recording under the listed document IDs."


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def dump(name, data):
    p = OUT / name
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(p)


def catalog():
    assert sha(ROOT / CATALOG) == CATALOG_SHA
    sets = {s["name"]: s for s in json.loads((ROOT / CATALOG).read_text())["recordSet"]}
    assert sets["merged_recordings"]["description"] == DEFINITION
    merged = {}
    for row in sets["merged_recordings"]["data"]:
        key = row["merged_recordings/merged_recording_id"]
        assert key not in merged
        merged[key] = {"recording_id": row["merged_recordings/recording_id"],
                       "annotation_document_ids": row["merged_recordings/annotation_document_ids"]}
    return sets["recordings"]["data"], merged


def freeze():
    assert not (OUT / "cohort_selection.private.json").exists()
    initial = json.loads((OUT / "contract_result.json").read_text())
    assert initial["phase"] == "initial_contract_pass_not_training_qualification"
    recordings, _ = catalog()
    with (ROOT / SPLITS).open() as f:
        folds = {r["exam_id"]: int(r["five_fold_split"]) for r in csv.DictReader(f)}
    pool = fixed_exam_pool(recordings, folds)
    paths = [f"exams/{row['exam']}.tar" for row in pool]
    objects = {}
    for start in range(0, len(paths), 40):
        request = urllib.request.Request(
            f"https://huggingface.co/api/datasets/{REPO}/paths-info/{REV}",
            data=json.dumps({"paths": paths[start:start + 40], "expand": False}).encode(),
            headers={"Content-Type": "application/json"})
        rows = json.load(urllib.request.urlopen(request, timeout=30))
        for row in rows:
            assert row["path"] not in objects
            objects[row["path"]] = row
    initial_selection = json.loads((OUT / "source_selection.private.json").read_text())
    for row, path in zip(pool, paths):
        row["publisher"] = objects.get(path)
        p = row["publisher"]
        row["eligible_publisher_size"] = bool(p and p["size"] <= 2_400_000_000 and len(p.get("lfs", {}).get("oid", "")) == 64)
        row["reuse_completed_initial_pair"] = row["exam"] == initial_selection["exam"]
        if row["reuse_completed_initial_pair"]:
            assert p["size"] == initial_selection["publisher"]["size"] and p["lfs"]["oid"] == initial_selection["publisher"]["lfs"]["oid"]
    selection = {"repo": REPO, "revision": REV, "frozen_unix": time.time(), "pool": pool,
        "card_sha256": sha(ROOT / CARD), "label_contract_sha256": sha(ROOT / LABELS),
        "catalog_sha256": CATALOG_SHA, "splits_sha256": sha(ROOT / SPLITS),
        "initial_contract_result_sha256": sha(OUT / "contract_result.json"),
        "initial_private_proof_sha256": sha(OUT / "contract_proofs.private.json"),
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "shared_caps": {"GPU_hours": 6, "CPU_hours": 3, "new_bytes": 3_000_000_000},
        "wider_transfer_cap": 200_000_000_000, "wider_process_CPU_cap_seconds": 1620,
        "retained_source_cap": 200_000_000}
    planned = sum(r["publisher"]["size"] for r in pool if r["eligible_publisher_size"] and not r["reuse_completed_initial_pair"])
    assert planned <= selection["wider_transfer_cap"]
    dump("cohort_selection.private.json", selection)
    result = {"phase": "fixed_cohort_selection_before_new_media", "selected_exams": len(pool),
        "selected_exams_by_provider_fold": dict(collections.Counter(r["provider_fold"] for r in pool)),
        "excluded_fixed_slots_missing_or_large": sum(not r["eligible_publisher_size"] for r in pool),
        "planned_additional_archive_bytes": planned,
        "initial_completed_pair_reused": sum(r["reuse_completed_initial_pair"] for r in pool),
        "selection_sha256": sha(OUT / "cohort_selection.private.json"),
        "card_sha256": selection["card_sha256"], "code_sha256": selection["code_sha256"],
        "label_contract_sha256": selection["label_contract_sha256"], "model_forwards": 0, "GPU_hours": 0}
    dump("cohort_freeze.json", result)
    return result


def extract_pair(archive, candidates, slot):
    import numpy as np
    import numcodecs
    import zarr
    from echoxflow.loading import open_recording
    from echoxflow.scan.geometry import sector_geometry_from_mapping
    from echoxflow.scan.conversion import sector_to_cartesian, sector_stack_to_cartesian_array
    result = []
    for purpose, (meta, label) in sorted(candidates.items()):
        root = meta["metadata_member"].rsplit("/", 1)[0]
        target = OUT / "cohort_samples" / f"{slot:03d}" / purpose
        assert not target.exists()
        exact = {root + "/.zattrs", root + "/.zgroup", root + "/data/.zgroup", root + "/timestamps/.zgroup"}
        for path in ["data/2d_brightness_mode", "timestamps/2d_brightness_mode"]:
            exact.update(n for n in archive.members if n.startswith(root + "/" + path + "/"))
        members, retained = [], 0
        for name in sorted(exact):
            if name not in archive.members:
                if name.endswith("/.zgroup"): continue
                raise ValueError("required source member absent")
            x = archive.read(name)
            retained += x["bytes"]
            assert retained <= 10_000_000
            out = target / name[len(root) + 1:]
            out.parent.mkdir(exist_ok=True, parents=True)
            out.write_bytes(x.pop("body"))
            members.append(x)
        assert sha(target / ".zattrs") == meta["metadata_sha256"]
        image_path = target / "data/2d_brightness_mode"
        zm = json.loads((image_path / ".zarray").read_text())
        raw = np.asarray(zarr.open_array(str(image_path), mode="r")[:])
        blob = (image_path / zm.get("dimension_separator", ".").join("0" for _ in zm["shape"])).read_bytes()
        body = numcodecs.get_codec(zm["compressor"]).decode(blob) if zm.get("compressor") else blob
        geometry = meta["attrs"]["recording_manifest"]["sectors"][0]["geometry"]
        native = validate_static(zm, geometry, raw, body)
        with open_recording(target) as store:
            assert np.array_equal(store.load_array("data/2d_brightness_mode"), native)
            t = store.load_array("timestamps/2d_brightness_mode")
            assert t.size == 1 and np.isfinite(t).all()
        geo = sector_geometry_from_mapping(geometry, grid_shape=tuple(raw.shape[1:]))
        image = sector_to_cartesian(native[0], geo, output_height=224)
        stack = sector_stack_to_cartesian_array(native, geo, grid=image.grid)
        assert np.array_equal(image.data, stack.data[0]) and np.array_equal(image.mask, stack.mask)
        assert image.data.dtype == np.uint8 and np.ptp(image.data[image.mask]) > 0
        np.save(target / "cartesian_no_overlay.npy", image.data)
        result.append({"purpose": purpose, "sample_relative_path": str(target.relative_to(OUT)),
            "raw_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
            "cartesian_sha256": hashlib.sha256(image.data.tobytes()).hexdigest(),
            "raw_shape": list(raw.shape), "geometry": geometry, "label": label,
            "merged_alias_count": len(meta["attrs"].get("merged_sources", [])), "member_proofs": members,
            "canonical_independent_official_raw_equal": True, "official_scan_conversion_parity": True})
    assert len({r["raw_sha256"] for r in result}) == 2, "opposite-purpose exact same native image"
    return result


def acquire(commit, process_start_cpu):
    assert len(commit) == 40
    assert not (OUT / "cohort_attempts.private.json").exists(), "never repeat a launched intake"
    source = json.loads((OUT / "cohort_selection.private.json").read_text())
    frozen = json.loads((OUT / "cohort_freeze.json").read_text())
    assert frozen["selection_sha256"] == sha(OUT / "cohort_selection.private.json")
    for p, digest in source["code_sha256"].items(): assert sha(ROOT / p) == digest
    assert sha(ROOT / CARD) == source["card_sha256"] and sha(ROOT / LABELS) == source["label_contract_sha256"]
    assert sha(OUT / "contract_result.json") == source["initial_contract_result_sha256"]
    assert sha(OUT / "contract_proofs.private.json") == source["initial_private_proof_sha256"]
    recordings, merged = catalog()
    exams_by_record = {r["recordings/recording_id"]: r["recordings/exam_id"] for r in recordings}
    allowed = {(r["measurement_kind"], r["measure_id"]): r["purpose"] for r in json.loads((ROOT / LABELS).read_text())["exact_whitelist"]}
    sys.path[:0] = [str(ROOT / "runs/a49_lvot_evidence_qualification_v1/vendor"), str(ROOT / "runs/a48_anatomical_qualification_v1/upstream/src")]
    attempts, pairs, transferred, peak = [], [], 0, 0
    cpu_start = process_start_cpu
    dump("cohort_attempts.private.json", attempts)
    for slot, selected in enumerate(source["pool"]):
        if time.process_time() - cpu_start > source["wider_process_CPU_cap_seconds"] - 60:
            dump("cohort_budget_stop.json", {"next_fixed_slot": slot, "reason": "process CPU closing reserve"})
            break
        start, cpu = time.time(), time.process_time()
        entry = {"fixed_slot": slot, "provider_fold": selected["provider_fold"], "exam": selected["exam"], "ok": False, "transferred_bytes": 0}
        archive_path = OUT / "current_archive.tar"
        partial = OUT / "current_archive.tar.part"
        try:
            if not selected["eligible_publisher_size"]: raise ValueError("fixed archive slot missing or above frozen size cap")
            if selected["reuse_completed_initial_pair"]:
                initial = json.loads((OUT / "contract_proofs.private.json").read_text())
                assert len(initial) == 2
                for row in initial:
                    old = OUT / "contract_samples" / row["purpose"]
                    for member in row["members"]:
                        rel = member["name"].split(".zarr/", 1)[1]
                        assert sha(old / rel) == member["sha256"]
                    dest = OUT / "cohort_samples" / f"{slot:03d}" / row["purpose"]
                    assert not dest.exists()
                    shutil.copytree(old, dest)
                pair = [{"purpose": r["purpose"], "raw_sha256": r["raw_sha256"], "cartesian_sha256": r["cartesian_sha256"],
                         "sample_relative_path": str((OUT / "cohort_samples" / f"{slot:03d}" / r["purpose"]).relative_to(OUT)),
                         "reused_initial_private_proof_sha256": sha(OUT / "contract_proofs.private.json"),
                         "canonical_independent_official_raw_equal": True, "official_scan_conversion_parity": True} for r in initial]
                entry.update(reused_completed_initial_pair=True, publisher_full_sha_previously_verified=True)
            else:
                assert not archive_path.exists() and not partial.exists()
                published = selected["publisher"]
                existing = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
                assert existing + published["size"] + 100_000_000 < 3_000_000_000
                url = f"https://huggingface.co/datasets/{REPO}/resolve/{REV}/" + published["path"]
                with urllib.request.urlopen(url, timeout=30) as response, partial.open("xb") as target:
                    while True:
                        b = response.read(8 * 1024 * 1024)
                        if not b: break
                        entry["transferred_bytes"] += len(b); transferred += len(b)
                        assert transferred <= source["wider_transfer_cap"] and entry["transferred_bytes"] <= published["size"]
                        target.write(b)
                partial.replace(archive_path)
                peak = max(peak, existing + archive_path.stat().st_size)
                with VerifiedArchive(archive_path, published) as archive:
                    entry["publisher_full_sha_verified"] = archive.proof
                    candidates, exclusions, metadata_bytes = {}, [], 0
                    for name in sorted(archive.members):
                        if not re.fullmatch(r"exams/[^/]+/[^/]+\.zarr/\.zattrs", name): continue
                        x = archive.read(name, max_bytes=1_000_000)
                        metadata_bytes += x["bytes"]
                        assert metadata_bytes <= 20_000_000
                        attrs = json.loads(x["body"])
                        if set(attrs.get("content_types", [])) != {"2d_brightness_mode", "1d_pulsed_wave_doppler"}: continue
                        if attrs.get("frame_counts_by_content_type", {}).get("2d_brightness_mode") != 1: continue
                        try:
                            assert attrs["exam_id"] == selected["exam"] == exams_by_record[attrs["recording_id"]]
                            label = acquisition_purpose(attrs, allowed, merged_catalog=merged)
                            candidates.setdefault(label["purpose"], ({"metadata_member": name, "metadata_sha256": x["sha256"], "attrs": attrs}, label))
                        except (ValueError, AssertionError, KeyError) as error:
                            exclusions.append({"metadata_member": name, "reason": str(error)})
                    entry.update(metadata_bytes_read=metadata_bytes, candidate_purposes=sorted(candidates), exclusions=exclusions)
                    assert set(candidates) == {"LVOT", "MITRAL_INFLOW"}, "no unambiguous metadata purpose pair"
                    pair = extract_pair(archive, candidates, slot)
            pairs.append({"fixed_slot": slot, "exam": selected["exam"], "provider_fold": selected["provider_fold"], "images": pair})
            entry["ok"] = True
        except Exception as error:
            entry.update(error_type=type(error).__name__, error=str(error))
            if partial.exists():
                partial.replace(OUT / f"failed_partial_{slot:03d}.tar.part")
            # Excluded pair is not replaced or retried; preserve any decoded bytes privately.
        finally:
            entry.update(wall_seconds=time.time() - start, process_CPU_seconds=time.process_time() - cpu)
            attempts.append(entry)
            dump("cohort_attempts.private.json", attempts)
            dump("cohort_pairs.private.json", pairs)
            archive_path.unlink(missing_ok=True)
            retained = sum(p.stat().st_size for p in (OUT / "cohort_samples").rglob("*") if p.is_file()) if (OUT / "cohort_samples").exists() else 0
            dump("cohort_progress.json", {"phase": "fixed_source_intake_running", "pid": os.getpid(), "finished_slots": len(attempts),
                "total_fixed_slots": len(source["pool"]), "paired_exams": len(pairs), "transferred_bytes": transferred,
                "process_CPU_seconds": time.process_time() - cpu_start, "retained_sample_bytes": retained,
                "namespace_peak_bytes_observed": peak, "model_forwards": 0, "GPU_hours": 0})
            assert retained <= source["retained_source_cap"]
    by_fold = collections.Counter(p["provider_fold"] for p in pairs)
    unique, crossfold = {}, []
    for purpose in ["LVOT", "MITRAL_INFLOW"]:
        fit, heldout, all_hashes, hash_folds = set(), set(), set(), collections.defaultdict(set)
        for pair in pairs:
            image = next(r for r in pair["images"] if r["purpose"] == purpose)
            h = image["raw_sha256"]; all_hashes.add(h); hash_folds[h].add(pair["provider_fold"])
            (heldout if pair["provider_fold"] == 0 else fit).add(h)
        unique[purpose] = {"all": len(all_hashes), "fit": len(fit), "heldout": len(heldout)}
        crossfold.extend(h for h, fs in hash_folds.items() if len(fs) > 1)
    # Check cross-purpose duplicates too, never silently assign them to one side.
    global_folds = collections.defaultdict(set)
    for pair in pairs:
        for image in pair["images"]: global_folds[image["raw_sha256"]].add(pair["provider_fold"])
    crossfold = sorted(h for h, fs in global_folds.items() if len(fs) > 1)
    passed = (len(attempts) == len(source["pool"]) and len(pairs) >= 80 and sum(v for f, v in by_fold.items() if f != 0) >= 64 and by_fold[0] >= 16 and
        all(v["all"] >= 80 and v["fit"] >= 64 and v["heldout"] >= 16 for v in unique.values()) and not crossfold)
    result = {"phase": "cohort_preparation_complete", "source_commit": commit, "all_source_quantity_gates": passed,
        "fixed_slots": len(source["pool"]), "attempted_slots": len(attempts), "successful_exam_pairs": len(pairs),
        "paired_exams_by_provider_fold": dict(by_fold), "distinct_native_images_by_purpose": unique,
        "crossfold_exact_image_hashes": len(crossfold), "transferred_additional_bytes": transferred,
        "namespace_peak_bytes_observed": peak, "process_CPU_seconds": time.process_time() - cpu_start,
        "private_pair_sha256": sha(OUT / "cohort_pairs.private.json"),
        "private_attempt_sha256": sha(OUT / "cohort_attempts.private.json"),
        "model_forwards": 0, "model_updates": 0, "GPU_hours": 0, "new_EV9V_readout": False,
        "labels_are_clinical_acquisition_purpose_not_anatomy_truth": True}
    dump("cohort_result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["freeze", "acquire"], required=True)
    parser.add_argument("--commit", default="")
    args = parser.parse_args()
    start, cpu, ok = time.time(), time.process_time(), False
    try:
        result = freeze() if args.stage == "freeze" else acquire(args.commit, cpu)
        ok = True
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        dump(f"cohort_{args.stage}_failure.private.json", {"error_type": type(error).__name__, "message": str(error)})
        raise
    finally:
        dump(f"cohort_{args.stage}_receipt.json", {"ok": ok, "started_unix": start, "finished_unix": time.time(),
            "wall_seconds": time.time() - start, "process_CPU_seconds": time.process_time() - cpu,
            "GPU_hours": 0, "namespace_bytes": sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())})
