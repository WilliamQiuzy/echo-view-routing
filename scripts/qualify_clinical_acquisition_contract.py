"""One fixed A50 exemplar pair: publisher-bound raw images, never anatomy labels."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from echo_routing.temporal.clinical_acquisition_source import (
    VerifiedArchive, acquisition_purpose, validate_static)

OUT = ROOT / "runs/a50_clinical_acquisition_v1"
CONTRACT = "docs/superpowers/plans/2026-10-30-clinical-acquisition-label-contract.json"
CATALOG = ROOT / "runs/a47_public_evidence_intake_v1/croissant.json"
CATALOG_SHA = "8e90e77079cce8804073ff281558d6c4b5f343117ab042e8791427cd0e369454"
DEFINITION = "Recordings that stored a physical acquisition already published by another recording of the same exam; their annotation documents live in that recording under the listed document IDs."
CODE = ["scripts/qualify_clinical_acquisition_contract.py", "echo_routing/temporal/clinical_acquisition_source.py"]


def sha_bytes(body): return hashlib.sha256(body).hexdigest()


def save(name, value):
    path = OUT / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def qualify(commit):
    assert commit and not (OUT / "contract_result.json").exists()
    contract = json.loads((ROOT / CONTRACT).read_text())
    assert all(sha_bytes((ROOT / p).read_bytes()) == digest for p, digest in contract["code_sha256"].items())
    inventory_body = (OUT / "metadata_inventory.private.json").read_bytes()
    assert sha_bytes(inventory_body) == contract["metadata_inventory_private_sha256"]
    catalog_body = CATALOG.read_bytes()
    assert sha_bytes(catalog_body) == CATALOG_SHA == contract["croissant_sha256"]
    recordsets = {row["name"]: row for row in json.loads(catalog_body)["recordSet"]}
    assert recordsets["merged_recordings"]["description"] == DEFINITION
    merged, records = {}, {}
    for row in recordsets["merged_recordings"]["data"]:
        key = row["merged_recordings/merged_recording_id"]
        assert key not in merged
        merged[key] = {"recording_id": row["merged_recordings/recording_id"],
                       "annotation_document_ids": row["merged_recordings/annotation_document_ids"]}
    for row in recordsets["recordings"]["data"]:
        key = row["recordings/recording_id"]
        assert key not in records
        records[key] = row
    allowed = {(row["measurement_kind"], row["measure_id"]): row["purpose"]
               for row in contract["exact_whitelist"]}
    selected = json.loads((OUT / "source_selection.private.json").read_text())
    assert sha_bytes((OUT / "source_selection.private.json").read_bytes()) == contract["selection_sha256"]
    candidates, exclusions = {}, []
    for row in sorted(json.loads(inventory_body), key=lambda r: r["metadata_member"]):
        attrs = row["attrs"]
        try:
            assert attrs["exam_id"] == selected["exam"]
            assert records[attrs["recording_id"]]["recordings/exam_id"] == selected["exam"]
            label = acquisition_purpose(attrs, allowed, merged_catalog=merged)
            candidates.setdefault(label["purpose"], (row, label))
        except (AssertionError, ValueError) as error:
            exclusions.append({"metadata_member": row["metadata_member"], "reason": str(error)})
    save("contract_exclusions.private.json", exclusions)
    assert set(candidates) == {"LVOT", "MITRAL_INFLOW"}, "metadata-only purpose pair admission failed"
    # Exact whitelist and selected metadata are frozen before any image is opened.
    save("contract_selected.private.json", {k: {"metadata_member": v[0]["metadata_member"],
        "metadata_sha256": v[0]["metadata_sha256"], "label": v[1]} for k, v in candidates.items()})
    sys.path[:0] = [str(ROOT / "runs/a49_lvot_evidence_qualification_v1/vendor"),
                    str(ROOT / "runs/a48_anatomical_qualification_v1/upstream/src")]
    import numpy as np
    import zarr
    import numcodecs
    from echoxflow.loading import open_recording
    from echoxflow.scan.geometry import sector_geometry_from_mapping
    from echoxflow.scan.conversion import sector_to_cartesian, sector_stack_to_cartesian_array
    proof, proofs, images, bytes_retained = {}, [], {}, 0
    with VerifiedArchive(OUT / "published_sample.tar", selected["publisher"]) as archive:
        proof = archive.proof
        for purpose, (row, label) in sorted(candidates.items()):
            root = row["metadata_member"].rsplit("/", 1)[0]
            image_path = "data/2d_brightness_mode"
            timestamp_path = "timestamps/2d_brightness_mode"
            exact = {root + "/.zattrs", root + "/.zgroup", root + "/data/.zgroup", root + "/timestamps/.zgroup"}
            for path in [image_path, timestamp_path]:
                exact.update(name for name in archive.members if name.startswith(root + "/" + path + "/"))
            target = OUT / "contract_samples" / purpose
            assert not target.exists(), "do not replace previously opened image samples"
            member_proofs = []
            for name in sorted(exact):
                if name not in archive.members:
                    if name.endswith("/.zgroup"): continue
                    raise ValueError("required selected object absent")
                result = archive.read(name)
                bytes_retained += result["bytes"]
                assert bytes_retained <= 20_000_000
                rel = name[len(root) + 1:]
                output = target / rel
                output.parent.mkdir(exist_ok=True, parents=True)
                output.write_bytes(result.pop("body"))
                member_proofs.append(result)
            assert sha_bytes((target / ".zattrs").read_bytes()) == row["metadata_sha256"]
            zm = json.loads((target / image_path / ".zarray").read_text())
            native = np.asarray(zarr.open_array(str(target / image_path), mode="r")[:])
            chunk_name = zm.get("dimension_separator", ".").join("0" for _ in zm["shape"])
            blob = (target / image_path / chunk_name).read_bytes()
            independently_decoded = numcodecs.get_codec(zm["compressor"]).decode(blob) if zm.get("compressor") else blob
            geometry = row["attrs"]["recording_manifest"]["sectors"][0]["geometry"]
            frame = validate_static(zm, geometry, native, independently_decoded)
            with open_recording(target) as official:
                assert np.array_equal(official.load_array(image_path), frame)
                stamp = official.load_array(timestamp_path)
                assert stamp.size == 1 and np.isfinite(stamp).all()
            geo = sector_geometry_from_mapping(geometry, grid_shape=tuple(frame.shape[1:]))
            image = sector_to_cartesian(frame[0], geo, output_height=224)
            stack = sector_stack_to_cartesian_array(frame, geo, grid=image.grid)
            assert np.array_equal(image.data, stack.data[0])
            assert np.array_equal(image.mask, stack.mask)
            assert image.data.dtype == np.uint8 and np.ptp(image.data[image.mask]) > 0
            images[purpose] = sha_bytes(frame.tobytes())
            np.save(target / "cartesian_no_overlay.npy", image.data)
            proofs.append({"purpose": purpose, "label_unit": label["label_unit"],
                "is_anatomy_or_visibility_truth": False,
                "canonical_independent_official_raw_equal": True,
                "official_single_stack_scan_conversion_equal": True,
                "raw_shape": list(frame.shape), "cartesian_shape": list(image.data.shape),
                "raw_sha256": images[purpose], "cartesian_sha256": sha_bytes(image.data.tobytes()),
                "merged_aliases_counted_once": len(row["attrs"].get("merged_sources", [])),
                "members": member_proofs})
    assert len(set(images.values())) == 2, "pair contains the exact same source image"
    save("contract_proofs.private.json", proofs)
    result = {"phase": "initial_contract_pass_not_training_qualification", "source_commit": commit,
        "contract_sha256": sha_bytes((ROOT / CONTRACT).read_bytes()),
        "publisher_archive_full_sha_verified": True, "publisher_archive_bytes": proof["size"],
        "canonical_independent_official_raw_equal": True, "official_scan_conversion_parity": True,
        "fit_exam_pairs": 1, "admitted_images_by_purpose": {"LVOT": 1, "MITRAL_INFLOW": 1},
        "excluded_unlabeled_candidates": len(exclusions), "selected_member_bytes": bytes_retained,
        "private_proof_sha256": sha_bytes((OUT / "contract_proofs.private.json").read_bytes()),
        "recording_identifiers_or_images_copied_to_Mac": False,
        "native_image_decodes": 2, "spectral_value_decodes": 0, "model_forwards": 0,
        "model_updates": 0, "GPU_hours": 0, "new_EV9V_readout": False,
        "array_data_sha256_semantics": "undefined; no claim that the old array-digest ambiguity is repaired"}
    save("contract_result.json", result)
    (OUT / "published_sample.tar").unlink()
    return result


if __name__ == "__main__":
    start, cpu, ok = time.time(), time.process_time(), False
    try:
        result = qualify(sys.argv[1] if len(sys.argv) > 1 else "")
        ok = True
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        save("contract_failure.private.json", {"error": type(error).__name__, "message": str(error)})
        raise
    finally:
        save("contract_receipt.json", {"ok": ok, "started_unix": start, "finished_unix": time.time(),
            "wall_seconds": time.time() - start, "process_CPU_seconds": time.process_time() - cpu,
            "GPU_hours": 0, "namespace_bytes": sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())})
