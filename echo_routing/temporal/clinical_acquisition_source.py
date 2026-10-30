"""Publisher-bound source objects and weak acquisition-purpose labels for A50.

No image decoding, model labels, anatomical truth, filesystem tar extraction or
changes to closed A48/A49 checks are provided by this module.
"""
import hashlib
from pathlib import Path, PurePosixPath
import tarfile


class VerifiedArchive:
    def __init__(self, path, publisher):
        self.path = Path(path)
        if self.path.stat().st_size != publisher["size"]:
            raise ValueError("publisher archive size mismatch")
        expected = publisher.get("lfs", {}).get("oid")
        if not expected or len(expected) != 64:
            raise ValueError("complete publisher LFS SHA256 required")
        digest = hashlib.sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError("publisher archive hash mismatch")
        self.proof = {"size": self.path.stat().st_size, "sha256": digest.hexdigest()}
        self.archive = tarfile.open(self.path, mode="r:")
        self.members = {}
        for member in self.archive.getmembers():
            self.members.setdefault(member.name, []).append(member)

    def __enter__(self):
        return self

    def close(self):
        self.archive.close()

    def __exit__(self, *args):
        self.close()

    @staticmethod
    def safe_name(name):
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError("unsafe archive path")
        return name

    def _regular(self, name, seen):
        self.safe_name(name)
        if name in seen or len(seen) >= 32:
            raise ValueError("cyclic/deep archive hardlink")
        matches = self.members.get(name, [])
        if len(matches) != 1:
            raise ValueError("missing or duplicate archive member")
        member = matches[0]
        if member.isfile():
            return member
        if member.islnk():
            return self._regular(member.linkname, seen | {name})
        raise ValueError("symlinks/non-regular member refused")

    def read(self, name, max_bytes=20_000_000):
        regular = self._regular(name, set())
        if regular.size > max_bytes:
            raise ValueError("selected member byte cap exceeded")
        body = self.archive.extractfile(regular).read()
        if len(body) != regular.size:
            raise ValueError("truncated selected member")
        original = self.members[name][0]
        return {"name": name, "body": body, "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "header_type": original.type.decode("ascii"),
                "resolved_regular_member": regular.name}


def acquisition_purpose(attrs, allowed, *, merged_catalog=None):
    """Return only a declared weak acquisition purpose under an exact whitelist.

    Does not identify which structure is present/absent in any individual frame.
    """
    for source in attrs.get("merged_sources", []):
        if merged_catalog is None:
            raise ValueError("version-matched merged identity catalog required")
        entry = merged_catalog.get(source.get("recording_id"))
        if (not entry or source.get("disposition") != "identical" or
            entry.get("recording_id") != attrs.get("recording_id") or
            entry.get("annotation_document_ids") != source.get("annotation_document_ids") or
            entry.get("annotation_document_ids")):
            raise ValueError("unresolved merged acquisition identity")
    if set(attrs.get("content_types", [])) != {"2d_brightness_mode", "1d_pulsed_wave_doppler"}:
        raise ValueError("exclusive B-mode plus blood PW acquisition required")
    counts = attrs.get("frame_counts_by_content_type", {})
    if counts.get("2d_brightness_mode") != 1 or counts.get("1d_pulsed_wave_doppler", 0) < 1:
        raise ValueError("one static reference and nonempty PW required")
    manifest = attrs["recording_manifest"]
    sectors, tracks = manifest.get("sectors", []), manifest.get("tracks", [])
    if (len(sectors) != 1 or sectors[0].get("semantic_id") != "bmode" or
        sectors[0].get("frames", {}).get("zarr_path") != "data/2d_brightness_mode" or
        sectors[0].get("geometry", {}).get("sample_type") != "Tissue(8)"):
        raise ValueError("unambiguous raw tissue B-mode object membership required")
    pw = [v for v in tracks if v.get("data", {}).get("zarr_path") == "data/1d_pulsed_wave_doppler"]
    if len(pw) != 1 or pw[0].get("semantic_id") != "pw_doppler":
        raise ValueError("one explicit PW acquisition track required")
    purposes, evidence = set(), []
    for annotation in manifest.get("annotations", []):
        if annotation.get("kind") != "measurement":
            continue
        if annotation.get("target", {}).get("semantic_id") != "pw_doppler":
            continue
        if annotation.get("target_data", {}).get("zarr_path") != "data/1d_pulsed_wave_doppler":
            raise ValueError("clinical measurement targets another stream")
        links = annotation.get("links", {})
        key = (links.get("measurement_kind"), links.get("measure_id"))
        if key not in allowed:
            raise ValueError("clinical purpose not in frozen exact whitelist")
        if links.get("init_from_cad") is not False or not links.get("source_entry_ref"):
            raise ValueError("manual/adjusted source provenance not established")
        purposes.add(allowed[key])
        evidence.append({"measurement_kind": key[0], "measure_id": key[1],
                         "target_data": annotation["target_data"],
                         "source_entry_ref": links["source_entry_ref"]})
    if len(purposes) != 1:
        raise ValueError("absent or conflicting clinical acquisition purpose")
    for annotation in manifest.get("annotations", []):
        if annotation.get("kind") != "measurement_inventory":
            continue
        for item in annotation.get("links", {}).get("measurements", []):
            key = (item.get("measurement_kind"), item.get("measure_id"))
            if str(key[1]).startswith("Cardiac/SD/") and (key not in allowed or allowed[key] not in purposes):
                raise ValueError("conflicting or unknown spectral clinical inventory")
    return {"purpose": purposes.pop(), "label_unit": "acquisition_purpose",
            "is_visibility_or_view_truth": False, "evidence": evidence}


def validate_static(zarray, geometry, canonical, independent_bytes):
    """Verify native byte/layout/grid parity, without treating it as anatomy truth."""
    import numpy as np
    shape = zarray["shape"]
    if (len(shape) != 3 or shape[0] != 1 or shape != zarray["chunks"] or
        np.dtype(zarray["dtype"]) != np.dtype("uint8") or
        zarray.get("order") != "C" or zarray.get("filters")):
        raise ValueError("one unfiltered single-chunk uint8 static image required")
    if list(geometry.get("grid_size", [])) != shape[1:]:
        raise ValueError("native image grid mismatch")
    raw = np.asarray(canonical)
    if list(raw.shape) != shape or raw.dtype != np.uint8:
        raise ValueError("canonical native shape/dtype mismatch")
    body = bytes(independent_bytes)
    if len(body) != raw.size or raw.tobytes(order="C") != body:
        raise ValueError("canonical and independent native bytes differ")
    return raw.copy()


def fixed_exam_pool(recordings, provider_folds, per_fold=40):
    """Metadata-only stable slots; failures do not create replacement exams."""
    counts, seen = {}, set()
    for row in recordings:
        identity = row["recordings/recording_id"]
        if identity in seen:
            raise ValueError("duplicate canonical catalog recording")
        seen.add(identity)
        modes = set(row.get("recordings/content_types", []))
        frames = row.get("recordings/frame_counts_by_content_type", {})
        if (modes == {"2d_brightness_mode", "1d_pulsed_wave_doppler"} and
            frames.get("2d_brightness_mode") == 1 and frames.get("1d_pulsed_wave_doppler", 0) > 0):
            exam = row["recordings/exam_id"]
            counts[exam] = counts.get(exam, 0) + 1
    return [{"exam": exam, "provider_fold": fold}
            for fold in range(5)
            for exam in sorted(e for e, count in counts.items()
                               if count >= 2 and provider_folds.get(e) == fold)[:per_fold]]


def validate_resume_prefix(attempts, pairs, pool):
    if len(attempts) > len(pool):
        raise ValueError("attempt prefix exceeds frozen pool")
    for slot, attempt in enumerate(attempts):
        selected = pool[slot]
        if (attempt.get("fixed_slot") != slot or attempt.get("exam") != selected["exam"] or
            attempt.get("provider_fold") != selected["provider_fold"]):
            raise ValueError("attempt prefix changed original frozen slots")
    successful = {r["fixed_slot"] for r in attempts if r.get("ok")}
    if {r["fixed_slot"] for r in pairs} != successful or len(pairs) != len(successful):
        raise ValueError("retained pair membership differs from completed attempts")
    return len(attempts), sum(r["transferred_bytes"] for r in attempts)
