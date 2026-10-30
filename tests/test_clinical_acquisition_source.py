import hashlib
import io
import tarfile

import pytest


def archive(tmp_path, members):
    path = tmp_path / "source.tar"
    with tarfile.open(path, "w") as output:
        for name, kind, value in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(value)
                output.addfile(info, io.BytesIO(value))
            else:
                info.type = tarfile.LNKTYPE if kind == "hardlink" else tarfile.SYMTYPE
                info.linkname = value
                output.addfile(info)
    return path, {"size": path.stat().st_size,
                  "lfs": {"oid": hashlib.sha256(path.read_bytes()).hexdigest()}}


def test_published_archive_proof_and_hardlink_resolve_exact_bytes(tmp_path):
    from echo_routing.temporal.clinical_acquisition_source import VerifiedArchive
    path, meta = archive(tmp_path, [("other/shared", "file", b"published"),
        ("picked/frame", "hardlink", "other/shared"), ("unused", "file", b"discard")])
    with VerifiedArchive(path, meta) as source:
        result = source.read("picked/frame")
        assert result["body"] == b"published"
        assert result["resolved_regular_member"] == "other/shared"
        assert source.proof["sha256"] == meta["lfs"]["oid"]


@pytest.mark.parametrize("change", ["hash", "size"])
def test_reject_publisher_mismatch_before_members(tmp_path, change):
    from echo_routing.temporal.clinical_acquisition_source import VerifiedArchive
    path, meta = archive(tmp_path, [("picked", "file", b"x")])
    if change == "hash": meta["lfs"]["oid"] = "0" * 64
    else: meta["size"] += 1
    with pytest.raises(ValueError, match=change):
        VerifiedArchive(path, meta)


@pytest.mark.parametrize("members,target", [
    ([("picked", "symlink", "other")], "picked"),
    ([("picked", "hardlink", "../escape")], "picked"),
    ([("picked", "hardlink", "second"), ("second", "hardlink", "picked")], "picked"),
    ([("picked", "file", b"a"), ("picked", "file", b"b")], "picked"),
    ([("picked/../escape", "file", b"a")], "picked/../escape"),
])
def test_unsafe_or_ambiguous_archive_member_is_rejected(tmp_path, members, target):
    from echo_routing.temporal.clinical_acquisition_source import VerifiedArchive
    path, meta = archive(tmp_path, members)
    with VerifiedArchive(path, meta) as source:
        with pytest.raises(ValueError): source.read(target)


def metadata():
    return {"content_types": ["2d_brightness_mode", "1d_pulsed_wave_doppler"],
        "frame_counts_by_content_type": {"2d_brightness_mode": 1, "1d_pulsed_wave_doppler": 10},
        "recording_manifest": {"sectors": [{"semantic_id": "bmode",
            "frames": {"zarr_path": "data/2d_brightness_mode"},
            "geometry": {"sample_type": "Tissue(8)"}}],
            "tracks": [{"semantic_id": "pw_doppler", "data": {"zarr_path": "data/1d_pulsed_wave_doppler"}}],
            "annotations": [{"kind": "measurement", "target": {"semantic_id": "pw_doppler"},
                "target_data": {"zarr_path": "data/1d_pulsed_wave_doppler"},
                "links": {"measurement_kind": "lvot_manual", "measure_id": "LVOT manual",
                    "init_from_cad": False, "source_entry_ref": {"element_index": 2}}}]}}


def test_acquisition_label_keeps_weak_purpose_semantics():
    from echo_routing.temporal.clinical_acquisition_source import acquisition_purpose
    result = acquisition_purpose(metadata(), {("lvot_manual", "LVOT manual"): "LVOT"})
    assert result["purpose"] == "LVOT"
    assert result["label_unit"] == "acquisition_purpose"
    assert not result["is_visibility_or_view_truth"]


@pytest.mark.parametrize("change", ["cad", "other_target", "no_source", "multiple_frames", "mixed_purpose", "unknown"])
def test_no_silent_purpose_label_inference(change):
    from echo_routing.temporal.clinical_acquisition_source import acquisition_purpose
    attrs = metadata(); ann = attrs["recording_manifest"]["annotations"][0]
    if change == "cad": ann["links"]["init_from_cad"] = True
    if change == "other_target": ann["target_data"]["zarr_path"] = "data/other"
    if change == "no_source": ann["links"].pop("source_entry_ref")
    if change == "multiple_frames": attrs["frame_counts_by_content_type"]["2d_brightness_mode"] = 2
    if change == "mixed_purpose":
        import copy
        other = copy.deepcopy(ann); other["links"].update(measurement_kind="mitral_manual", measure_id="Mitral manual")
        attrs["recording_manifest"]["annotations"].append(other)
    if change == "unknown": ann["links"]["measurement_kind"] = "unrecognised"
    with pytest.raises(ValueError):
        acquisition_purpose(attrs, {("lvot_manual", "LVOT manual"): "LVOT", ("mitral_manual", "Mitral manual"): "Mitral"})
