"""a5c_expert_pilot_v1: record schema, blind / unblind state machine, CSV round trip, overlay invariants and the private
server, all on synthetic data (no real media or identifiers)."""
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import a5c_pilot_records as rec  # noqa: E402
import a5c_pilot_server as srv  # noqa: E402

AXIS = {"sha256": "abc", "n_frames": 10, "time_base": "1/15360", "source": "synthetic"}


def full(**kw):
    f = {"overall_view": "A5C", "overall_confidence": "medium", "view_intervals": [{"start": 0, "end": 4, "state": "A5C"}, {"start": 5, "end": 9, "state": "transitional_or_off_axis"}],
         "lvot": {"status": "intervals", "intervals": [{"start": 2, "end": 3, "state": "visible"}]}, "aortic_valve": {"status": "not_visible_anywhere", "intervals": []},
         "image_quality": "poor", "quality_reason": "dropout, ünïcode, \"quotes\", commas", "comments": "line1\nline2"}
    f.update(kw); return f


@pytest.mark.parametrize("bad", [
    {"view_intervals": [{"start": 0, "end": 10, "state": "A4C"}]},                                            # beyond the last frame
    {"view_intervals": [{"start": 5, "end": 2, "state": "A4C"}]},                                             # reversed
    {"view_intervals": [{"start": 0, "end": 4, "state": "A4C"}, {"start": 4, "end": 6, "state": "A5C"}]},     # overlap
    {"lvot": {"status": "intervals", "intervals": []}},                                                      # intervals status without intervals
    {"lvot": {"status": "not_visible_anywhere", "intervals": [{"start": 1, "end": 2, "state": "visible"}]}},  # contradiction
    {"overall_view": "LVOT"}, {"lvot": {"status": None, "intervals": [{"start": 1, "end": 2, "state": "not_visible"}]}},
    {"view_intervals": [{"start": 1.0, "end": 2, "state": "A4C"}]}, {"label_fit": "yes"},                   # float frame; unblinded field in the blind phase
])
def test_validation_refuses(bad):
    with pytest.raises(rec.RecordError):
        rec.validate_fields(full(**bad), 10, "blind")


def test_unanswered_is_not_negative():
    g = rec.validate_fields({}, 10, "blind")
    assert g["lvot"]["status"] is None and g["overall_view"] is None and g["image_quality"] is None
    codes = rec.frame_codes(g, 10)
    assert set(codes["view"]) == {rec.UNKNOWN} and set(codes["lvot"]) == {rec.UNKNOWN}


def test_blind_lock_unblind_and_versions():
    r = rec.new_record("P01", "R1", AXIS, True)
    with pytest.raises(rec.RecordError):
        rec.unblind(r)
    r = rec.add_version(r, {"comments": "draft"}, "blind", False, 10); assert r["state"] == "blind_draft"
    r = rec.add_version(r, full(), "blind", True, 10); assert r["state"] == "blind_submitted"
    with pytest.raises(rec.RecordError):
        rec.add_version(r, full(overall_view="A4C"), "blind", False, 10)
    with pytest.raises(rec.RecordError):
        rec.add_version(r, full(label_fit="no"), "unblinded", True, 10)
    r = rec.unblind(r); r = rec.add_version(r, full(overall_view="A4C", label_fit="partly", review_comment="after"), "unblinded", True, 10)
    assert r["state"] == "unblinded_submitted" and rec.blind_version(r)["fields"]["overall_view"] == "A5C" and rec.current(r)["fields"]["overall_view"] == "A4C"
    assert [v["version"] for v in r["versions"]] == [1, 2, 3]


def test_csv_round_trip_keeps_every_field():
    r = rec.add_version(rec.new_record("P01", "R1", AXIS, True), full(), "blind", True, 10); r = rec.unblind(r, now=0)
    r = rec.add_version(r, full(label_fit="yes", review_comment="ok, \"fine\""), "unblinded", False, 10)
    e = rec.new_record("P02", "R1", AXIS, True)
    assert rec.canonical(rec.from_csv(rec.to_csv([r, e]))) == rec.canonical([r, e])


def test_overlay_invariants():
    axes = {"P01": AXIS}; links = {"P01": {"fold": 1, "sample_frames": [0, 3, 6, 9, 12]}}
    a = rec.add_version(rec.new_record("P01", "R1", AXIS, True), full(), "blind", True, 10)
    b = rec.add_version(rec.new_record("P01", "R2", AXIS, True), full(view_intervals=[{"start": 0, "end": 4, "state": "A4C"}], lvot={"status": "not_visible_anywhere", "intervals": []}), "blind", True, 10)
    ov = rec.build_overlay([a, b], axes, links); r = ov["recordings"]["P01"]
    assert r["frames"]["view"][:5] == [rec.DISAGREE] * 5 and r["frames"]["view"][5:] == [rec.VIEW_CODE["transitional_or_off_axis"]] * 5
    assert r["frames"]["lvot"][2] == rec.DISAGREE and r["frames"]["lvot"][0] == rec.STRUCT_CODE["not_visible"]           # R1 asserts nothing at frame 0; R2 asserts not visible
    assert r["frames"]["aortic_valve"] == [rec.STRUCT_CODE["not_visible"]] * 10 and rec.VIEW_CODE["A4C"] not in r["frames"]["view"][5:]   # not visible never becomes A4C
    assert r["samples_10hz"]["view"][-1] == rec.UNKNOWN and r["fold"] == 1 and r["overall_view"] == "A5C"
    assert "public" not in json.dumps(ov) and rec.check_overlay(ov, axes)["P01"]["view"]["disagree"] == 5
    real = rec.add_version(rec.new_record("P01", "R3", AXIS, False), full(), "blind", True, 10)
    with pytest.raises(rec.RecordError):
        rec.build_overlay([a, real], axes, links)
    with pytest.raises(rec.RecordError):
        rec.build_overlay([a], {"P01": {**AXIS, "sha256": "other"}}, links)
    draft = rec.add_version(rec.new_record("P01", "R1", AXIS, True), full(), "blind", False, 10)
    assert rec.build_overlay([draft], axes, links)["recordings"] == {}                                                    # drafts never reach training


def test_disagreement_rows_format():
    a = rec.add_version(rec.new_record("P01", "R1", AXIS, True), full(), "blind", True, 10)
    b = rec.add_version(rec.new_record("P01", "R2", AXIS, True), full(overall_view="A4C"), "blind", True, 10)
    d = rec.add_version(rec.new_record("P01", "R3", AXIS, True), full(), "blind", False, 10)                     # a draft is not compared
    rows = rec.disagreement_rows([a, b, d], {"P01": AXIS}); by = {r["field"]: r for r in rows}
    assert {r["reader_b"] for r in rows} == {"R2"} and by["overall_view"]["equal"] == 0 and by["view.frames"]["frames_disagree"] == 0 and by["lvot.frames"]["frames_both_asserted"] == 2
    assert rec.rows_csv(rows, rec.DISAGREE_COLUMNS).count("\n") == len(rows) + 1


@pytest.fixture()
def server(tmp_path):
    from PIL import Image
    pkg = tmp_path / "package"; man = {"synthetic": True, "order": ["Z01", "Z02"], "cases": {}}
    for pid, n in (("Z01", 5), ("Z02", 3)):
        (pkg / "media" / pid).mkdir(parents=True)
        for k in range(n):
            Image.new("RGB", (8, 8), (k, 0, 0)).save(pkg / "media" / pid / f"{k:06d}.png")
        man["cases"][pid] = {"pid": pid, "n_frames": n, "time_base": "1/30", "pts_time": [k / 30 for k in range(n)], "fps_nominal": "30/1", "fps_from_pts": 30.0, "constant_rate": True,
                             "source": "synthetic", "sha256": f"sha{pid}", "resolution_note": "one step = one frame", "frame_ext": "png", "SECRET_FIELD": "S1-03 A5C"}
    (pkg / "cases.json").write_text(json.dumps(man)); (pkg / "reveal.json").write_text(json.dumps({"Z01": {"public_clip_code": "A5C"}}))
    (tmp_path / "readers.json").write_text(json.dumps({"tokens": {"tok1": "R1", "tok2": "R2"}}))
    app = srv.App(pkg, tmp_path / "ann", tmp_path / "readers.json", True); httpd = srv.serve(app, 0); threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], tmp_path
    httpd.shutdown()


def call(port, path, token, body=None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}{'&' if '?' in path else '?'}t={token}", data=None if body is None else json.dumps(body).encode(), method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_blind_flow_and_isolation(server):
    port, tmp = server
    assert call(port, "/api/me", "nope")[0] == 403
    st, b = call(port, "/api/case/Z01", "tok1"); j = json.loads(b)
    assert st == 200 and "SECRET_FIELD" not in j["case"] and b"S1-03" not in b and b"public_clip_code" not in b
    assert call(port, "/api/reveal/Z01", "tok1")[0] == 403
    assert call(port, "/media/Z01/000005.png", "tok1")[0] == 404 and call(port, "/media/Z01/../../readers.json", "tok1")[0] == 404 and call(port, "/media/Z01/000000.png", "")[0] == 403
    f = {"overall_view": "A4C", "lvot": {"status": "intervals", "intervals": [{"start": 0, "end": 4, "state": "uncertain"}]}}
    assert call(port, "/api/save/Z01", "tok1", {"phase": "blind", "submit": True, "fields": f})[0] == 200
    assert call(port, "/api/save/Z01", "tok1", {"phase": "blind", "submit": False, "fields": f})[0] == 400
    assert json.loads(call(port, "/api/case/Z01", "tok2")[1])["record"]["versions"] == []                  # reader 2 sees nothing of reader 1
    assert call(port, "/api/unblind/Z01", "tok2", {})[0] == 400 and call(port, "/api/reveal/Z01", "tok2")[0] == 403
    assert call(port, "/api/unblind/Z01", "tok1", {})[0] == 200 and json.loads(call(port, "/api/reveal/Z01", "tok1")[1])["reveal"]["public_clip_code"] == "A5C"
    rj = json.loads(call(port, "/api/export?fmt=json", "tok1")[1])["records"]; rc = call(port, "/api/export?fmt=csv", "tok1")[1].decode()
    assert len(rj) == 1 and rj[0]["synthetic"] is True and rec.canonical(rec.from_csv(rc)) == rec.canonical(rj)
    assert sorted(p.relative_to(tmp / "ann").as_posix() for p in (tmp / "ann").rglob("*.json")) == ["R1/Z01.json"]
