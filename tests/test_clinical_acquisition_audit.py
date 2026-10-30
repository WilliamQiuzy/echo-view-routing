import copy
import importlib.util
from pathlib import Path

import pytest


def load_auditor():
    path = Path(__file__).resolve().parents[1] / "scripts/audit_clinical_acquisition_cohort.py"
    spec = importlib.util.spec_from_file_location("a50_independent_auditor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def attrs():
    return {"recording_id": "canonical", "exam_id": "exam", "content_types": ["2d_brightness_mode", "1d_pulsed_wave_doppler"],
        "frame_counts_by_content_type": {"2d_brightness_mode": 1, "1d_pulsed_wave_doppler": 20},
        "recording_manifest": {"sectors": [{"semantic_id": "bmode", "frames": {"zarr_path": "data/2d_brightness_mode"}, "geometry": {"sample_type": "Tissue(8)"}}],
            "tracks": [{"semantic_id": "pw_doppler", "data": {"zarr_path": "data/1d_pulsed_wave_doppler"}}],
            "annotations": [{"kind": "measurement", "target": {"type": "track", "semantic_id": "pw_doppler"},
                "target_data": {"zarr_path": "data/1d_pulsed_wave_doppler"},
                "links": {"measurement_kind": "lvot_manual", "measure_id": "Cardiac/SD/LVOT", "init_from_cad": False, "source_entry_ref": {"index": 1}}}]}}


def test_independent_auditor_rejects_wrong_target_and_unverified_merge():
    audit = load_auditor()
    allowed = {("lvot_manual", "Cardiac/SD/LVOT"): "LVOT"}
    assert audit.verify_labels(attrs(), "LVOT", allowed, {}) == 0
    other = attrs()
    other["recording_manifest"]["annotations"][0]["target_data"]["zarr_path"] = "data/other"
    with pytest.raises(AssertionError): audit.verify_labels(other, "LVOT", allowed, {})
    other = attrs(); other["merged_sources"] = [{"recording_id": "alias", "disposition": "identical", "annotation_document_ids": []}]
    with pytest.raises(AssertionError): audit.verify_labels(other, "LVOT", allowed, {})
    assert audit.verify_labels(other, "LVOT", allowed, {"alias": {"canonical": "canonical", "documents": []}}) == 1


def test_independent_auditor_detects_unlocalized_inventory_conflict():
    audit = load_auditor()
    a = attrs()
    a["recording_manifest"]["annotations"].append({"kind": "measurement_inventory", "links": {
        "measurements": [{"measurement_kind": "mitral", "measure_id": "Cardiac/SD/Mitral"}]}})
    with pytest.raises(AssertionError):
        audit.verify_labels(a, "LVOT", {("lvot_manual", "Cardiac/SD/LVOT"): "LVOT"}, {})
