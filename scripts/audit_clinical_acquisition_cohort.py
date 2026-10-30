"""Independent A50 retained-source/label/count audit; imports no admission helper/model."""
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/a50_clinical_acquisition_v1"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def verify_labels(attrs, purpose, allowed, merged):
    assert set(attrs["content_types"]) == {"2d_brightness_mode", "1d_pulsed_wave_doppler"}
    counts = attrs["frame_counts_by_content_type"]
    assert counts["2d_brightness_mode"] == 1 and counts["1d_pulsed_wave_doppler"] > 0
    manifest = attrs["recording_manifest"]
    sectors = manifest["sectors"]
    assert len(sectors) == 1 and sectors[0]["semantic_id"] == "bmode"
    assert sectors[0]["frames"]["zarr_path"] == "data/2d_brightness_mode"
    assert sectors[0]["geometry"]["sample_type"] == "Tissue(8)"
    tracks = [r for r in manifest["tracks"] if r["data"].get("zarr_path") == "data/1d_pulsed_wave_doppler"]
    assert len(tracks) == 1 and tracks[0]["semantic_id"] == "pw_doppler"
    used = []
    for ann in manifest.get("annotations", []):
        if ann.get("kind") == "measurement" and ann.get("target", {}).get("semantic_id") == "pw_doppler":
            assert ann["target"].get("type") == "track"
            assert ann["target_data"]["zarr_path"] == "data/1d_pulsed_wave_doppler"
            links = ann["links"]
            assert links["init_from_cad"] is False and links["source_entry_ref"]
            key = (links["measurement_kind"], links["measure_id"])
            assert allowed.get(key) == purpose
            used.append(key)
        if ann.get("kind") == "measurement_inventory":
            for item in ann.get("links", {}).get("measurements", []):
                if str(item.get("measure_id", "")).startswith("Cardiac/SD/"):
                    assert allowed.get((item.get("measurement_kind"), item.get("measure_id"))) == purpose
    assert used
    for alias in attrs.get("merged_sources", []):
        row = merged.get(alias["recording_id"])
        assert row and row["canonical"] == attrs["recording_id"]
        assert alias["disposition"] == "identical"
        assert row["documents"] == alias["annotation_document_ids"] == []
    return len(attrs.get("merged_sources", []))


def independent_cartesian(raw, geometry, output_shape):
    import numpy as np
    d0, d1 = float(geometry["depth_start_m"]), float(geometry["depth_end_m"])
    tilt, width = float(geometry.get("tilt_rad", 0)), float(geometry["width_rad"])
    a0, a1 = tilt - width / 2, tilt + width / 2
    assert 0 <= d0 < d1 and 0 < width < math.pi
    angles = [a0, a1] + [v * math.pi / 2 for v in range(-4, 5) if a0 <= v * math.pi / 2 <= a1]
    coords = [(d * math.sin(a), d * math.cos(a)) for d in [d0, d1] for a in angles]
    x0, x1 = min(x for x, y in coords), max(x for x, y in coords)
    y0, y1 = min(y for x, y in coords), max(y for x, y in coords)
    height = 224
    expected_width = round(height * max(1e-6, x1 - x0) / max(1e-6, y1 - y0))
    assert tuple(output_shape) == (height, expected_width)
    x, y = np.meshgrid(np.linspace(x0, x1, expected_width), np.linspace(y0, y1, height))
    depth, angle = np.hypot(x, y), np.arctan2(x, y)
    mask = (depth >= d0) & (depth <= d1) & (angle >= a0) & (angle <= a1)
    rr = (depth - d0) * (raw.shape[0] - 1) / (d1 - d0)
    cc = (angle - a0) * (raw.shape[1] - 1) / (a1 - a0)
    ri, ci = np.floor(rr).astype(int), np.floor(cc).astype(int)
    wr, wc = (rr - ri).astype(np.float32), (cc - ci).astype(np.float32)
    r0, r1 = np.clip(ri, 0, raw.shape[0] - 1), np.clip(ri + 1, 0, raw.shape[0] - 1)
    c0, c1 = np.clip(ci, 0, raw.shape[1] - 1), np.clip(ci + 1, 0, raw.shape[1] - 1)
    values = raw.astype(np.float32)
    top = values[r0, c0] * (1 - wc) + values[r0, c1] * wc
    bottom = values[r1, c0] * (1 - wc) + values[r1, c1] * wc
    pixels = np.clip(np.rint(top * (1 - wr) + bottom * wr), 0, 255).astype(np.uint8)
    pixels[~mask] = 0
    return pixels


def audit():
    sys.path[:0] = [str(ROOT / "runs/a49_lvot_evidence_qualification_v1/vendor")]
    import numpy as np
    import numcodecs
    import zarr
    selection = json.loads((OUT / "cohort_selection.private.json").read_text())
    frozen = json.loads((OUT / "cohort_freeze.json").read_text())
    aggregate = json.loads((OUT / "cohort_result.json").read_text())
    assert digest(OUT / "cohort_selection.private.json") == frozen["selection_sha256"]
    assert digest(OUT / "cohort_pairs.private.json") == aggregate["private_pair_sha256"]
    assert digest(OUT / "cohort_attempts.private.json") == aggregate["private_attempt_sha256"]
    catalog_path = ROOT / "runs/a47_public_evidence_intake_v1/croissant.json"
    assert digest(catalog_path) == selection["catalog_sha256"]
    sets = {r["name"]: r for r in json.loads(catalog_path.read_text())["recordSet"]}
    assert sets["merged_recordings"]["description"] == "Recordings that stored a physical acquisition already published by another recording of the same exam; their annotation documents live in that recording under the listed document IDs."
    merged = {}
    for r in sets["merged_recordings"]["data"]:
        alias = r["merged_recordings/merged_recording_id"]
        assert alias not in merged
        merged[alias] = {"canonical": r["merged_recordings/recording_id"], "documents": r["merged_recordings/annotation_document_ids"]}
    split_path = ROOT / "runs/a49_lvot_evidence_qualification_v1/splits.csv"
    assert digest(split_path) == selection["splits_sha256"]
    folds = {r["exam_id"]: int(r["five_fold_split"]) for r in csv.DictReader(split_path.open())}
    eligible, seen = collections.Counter(), set()
    for r in sets["recordings"]["data"]:
        assert r["recordings/recording_id"] not in seen
        seen.add(r["recordings/recording_id"])
        c, n = r["recordings/content_types"], r["recordings/frame_counts_by_content_type"]
        if (set(c) == {"2d_brightness_mode", "1d_pulsed_wave_doppler"} and
            n.get("2d_brightness_mode") == 1 and n.get("1d_pulsed_wave_doppler", 0) > 0):
            eligible[r["recordings/exam_id"]] += 1
    expected = [(e, f) for f in range(5) for e in sorted(e for e in eligible if eligible[e] >= 2 and folds[e] == f)[:40]]
    assert [(r["exam"], r["provider_fold"]) for r in selection["pool"]] == expected
    attempts = json.loads((OUT / "cohort_attempts.private.json").read_text())
    pairs = json.loads((OUT / "cohort_pairs.private.json").read_text())
    assert len(attempts) == len(expected) == 200
    verified_archives, reused, partial, metadata_fail = 0, 0, 0, 0
    transferred = 0
    for i, row in enumerate(attempts):
        source = selection["pool"][i]
        assert row["fixed_slot"] == i and (row["exam"], row["provider_fold"]) == expected[i]
        transferred += row["transferred_bytes"]
        proof = row.get("publisher_full_sha_verified")
        if isinstance(proof, dict):
            assert proof["sha256"] == source["publisher"]["lfs"]["oid"]
            assert proof["size"] == source["publisher"]["size"] == row["transferred_bytes"]
            verified_archives += 1
        elif row.get("reused_completed_initial_pair"):
            assert source["reuse_completed_initial_pair"] and row["transferred_bytes"] == 0
            reused += 1
        else:
            assert row["error_type"] == "BootstrapEnvironmentAbort" and row["ok"] is False
            partial += 1
        if row.get("error") == "no unambiguous metadata purpose pair": metadata_fail += 1
    assert transferred == aggregate["transferred_additional_bytes"]
    assert {r["fixed_slot"] for r in pairs} == {r["fixed_slot"] for r in attempts if r["ok"]}
    assert len(pairs) == len({r["exam"] for r in pairs})
    label_contract = json.loads((ROOT / "docs/superpowers/plans/2026-10-30-clinical-acquisition-label-contract.json").read_text())
    allowed = {(r["measurement_kind"], r["measure_id"]): r["purpose"] for r in label_contract["exact_whitelist"]}
    initial_proof = json.loads((OUT / "contract_proofs.private.json").read_text())
    initial_by_purpose = {r["purpose"]: r for r in initial_proof}
    all_hashes = collections.defaultdict(set)
    hashes = {p: {"all": set(), "fit": set(), "heldout": set()} for p in ["LVOT", "MITRAL_INFLOW"]}
    member_count, alias_count, max_scan_delta = 0, 0, 0
    for pair in pairs:
        assert (pair["exam"], pair["provider_fold"]) == expected[pair["fixed_slot"]]
        assert {r["purpose"] for r in pair["images"]} == set(hashes)
        pair_hashes = []
        for row in pair["images"]:
            purpose = row["purpose"]
            exact_relative = f"cohort_samples/{pair['fixed_slot']:03d}/{purpose}"
            assert row["sample_relative_path"] == exact_relative
            target = OUT / exact_relative
            attrs = json.loads((target / ".zattrs").read_text())
            assert attrs["exam_id"] == pair["exam"]
            alias_count += verify_labels(attrs, purpose, allowed, merged)
            members = row.get("member_proofs")
            if members is None:
                assert row["reused_initial_private_proof_sha256"] == digest(OUT / "contract_proofs.private.json")
                members = initial_by_purpose[purpose]["members"]
            owner_prefix = f"exams/{attrs['exam_id']}/{attrs['recording_id']}.zarr/"
            for item in members:
                assert item["name"].startswith(owner_prefix)
                rel = item["name"][len(owner_prefix):]
                assert rel in [".zattrs", ".zgroup", "data/.zgroup", "timestamps/.zgroup"] or rel.startswith(("data/2d_brightness_mode/", "timestamps/2d_brightness_mode/"))
                path = target / rel
                assert path.stat().st_size == item["bytes"] and digest(path) == item["sha256"]
                member_count += 1
            native_path = target / "data/2d_brightness_mode"
            zm = json.loads((native_path / ".zarray").read_text())
            assert zm["shape"] == zm["chunks"] and len(zm["shape"]) == 3 and zm["shape"][0] == 1
            assert np.dtype(zm["dtype"]) == np.uint8 and zm["order"] == "C" and not zm.get("filters")
            blob = (native_path / zm.get("dimension_separator", ".").join(["0"] * 3)).read_bytes()
            data = numcodecs.get_codec(zm["compressor"]).decode(blob) if zm.get("compressor") else blob
            raw = np.frombuffer(data, dtype=np.uint8).reshape(zm["shape"])
            assert np.array_equal(np.asarray(zarr.open_array(str(native_path), mode="r")[:]), raw)
            h = hashlib.sha256(raw.tobytes()).hexdigest()
            assert h == row["raw_sha256"]
            geometry = attrs["recording_manifest"]["sectors"][0]["geometry"]
            assert list(geometry["grid_size"]) == list(raw.shape[1:])
            cartesian = np.load(target / "cartesian_no_overlay.npy", allow_pickle=False)
            assert hashlib.sha256(cartesian.tobytes()).hexdigest() == row["cartesian_sha256"]
            independent = independent_cartesian(raw[0], geometry, cartesian.shape)
            delta = int(np.max(np.abs(independent.astype(int) - cartesian.astype(int))))
            # Independent FP64 coordinate algebra with FP32 interpolation can differ by1 gray level.
            assert delta <= 1
            max_scan_delta = max(max_scan_delta, delta)
            pair_hashes.append(h); all_hashes[h].add(pair["provider_fold"])
            hashes[purpose]["all"].add(h)
            hashes[purpose]["heldout" if pair["provider_fold"] == 0 else "fit"].add(h)
        assert len(set(pair_hashes)) == 2
    unique = {p: {k: len(v) for k, v in groups.items()} for p, groups in hashes.items()}
    by_fold = dict(collections.Counter(r["provider_fold"] for r in pairs))
    crossfold = sum(len(fs) > 1 for fs in all_hashes.values())
    fit = sum(n for f, n in by_fold.items() if f != 0)
    gate = (len(pairs) >= 80 and fit >= 64 and by_fold.get(0, 0) >= 16 and
            all(r["all"] >= 80 and r["fit"] >= 64 and r["heldout"] >= 16 for r in unique.values()) and not crossfold)
    assert len(pairs) == aggregate["successful_exam_pairs"]
    assert {str(k): v for k, v in by_fold.items()} == aggregate["paired_exams_by_provider_fold"]
    assert unique == aggregate["distinct_native_images_by_purpose"] and crossfold == aggregate["crossfold_exact_image_hashes"]
    assert gate is aggregate["all_source_quantity_gates"] is False
    assert aggregate["model_forwards"] == aggregate["model_updates"] == aggregate["GPU_hours"] == 0
    assert not (OUT / "current_archive.tar").exists() and not (OUT / "current_archive.tar.part").exists()
    return {"all_independent_source_checks_pass": True, "source_quantity_gate": gate,
        "fixed_slots_recomputed": len(expected), "completed_attempts": len(attempts),
        "new_full_archive_proof_receipts_verified": verified_archives, "initial_full_archive_reused": reused,
        "engineering_partial_archive_excluded": partial, "metadata_missing_pair_exclusions": metadata_fail,
        "retained_member_hashes_recomputed": member_count, "native_image_hashes_recomputed": sum(v["all"] for v in unique.values()),
        "canonical_independent_raw_equal": True, "independent_scan_conversion_max_uint8_error": max_scan_delta,
        "scan_diagnostic_tolerance_note": "Independent FP64 coordinate algebra diagnostic bound1; original official single/stack parity was exact and is not relaxed.",
        "merged_aliases_independently_bound": alias_count, "successful_exam_pairs": len(pairs),
        "paired_exams_by_provider_fold": by_fold, "distinct_native_images_by_purpose": unique,
        "crossfold_exact_image_hashes": crossfold, "new_transferred_archive_bytes": transferred,
        "publisher_full_archive_scope": "Publisher SHA/size recomputed during original streaming/seekable admission; full temporary archives deleted. Audit rechecks saved complete-object proof receipts and all retained members, without redownload.",
        "private_pair_sha256": digest(OUT / "cohort_pairs.private.json"), "private_attempt_sha256": digest(OUT / "cohort_attempts.private.json"),
        "model_forwards": 0, "model_updates": 0, "GPU_hours": 0, "new_EV9V_readout": False,
        "scientific_scope": "Initial image/object contract passes; fixed paired-purpose cohort fails quantity before any representation training. No conclusion about learned A5C recovery."}


if __name__ == "__main__":
    start, cpu = time.time(), time.process_time()
    result = audit()
    result.update(finished_unix=time.time(), wall_seconds=time.time() - start,
                  process_CPU_seconds=time.process_time() - cpu, audit_code_sha256=digest(__file__))
    (OUT / "independent_source_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
