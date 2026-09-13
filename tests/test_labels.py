from __future__ import annotations

import pytest

from echo_routing.audit.label_map import LabelError, load_ontology


def test_ontology_loads_and_covers_all_nine_codes(ontology):
    assert len(ontology.raw_codes) == 9
    assert ontology.mapping_version == "family5_v1"
    assert ontology.family_classes == ("PLAX", "PSAX", "A4C", "A5C", "SC4C")


@pytest.mark.parametrize(
    "code,family",
    [("PLHLA", "PLAX"), ("PASA", "PSAX"), ("PMVLSA", "PSAX"), ("PPMLSA", "PSAX"), ("PMASA", "PSAX"),
     ("A4C", "A4C"), ("A5C", "A5C"), ("SC4C", "SC4C"), ("PMPALA", None)],
)
def test_family_mapping(ontology, code, family):
    assert ontology.family(code) == family


def test_family_index_none_for_excluded(ontology):
    assert ontology.family_index("PMPALA") is None
    assert ontology.family_index("A4C") == 2


def test_raw9_index_matches_stfm_order(ontology):
    assert ontology.raw9_index("PLHLA") == 0
    assert ontology.raw9_index("SC4C") == 8


def test_unknown_code_raises(ontology):
    with pytest.raises(LabelError):
        ontology.family("XYZ")


def test_task_dispatch(ontology):
    assert ontology.classes("raw9")[4] == "A4C"
    assert ontology.index("raw9", "A4C") == 4
    with pytest.raises(LabelError):
        ontology.classes("nope")


def test_load_ontology_rejects_bad_mapping(tmp_path):
    bad = tmp_path / "labels.yaml"
    bad.write_text(
        "mapping_version: x\nraw_codes: [A4C, A5C]\nfamily5: {A4C: A4C}\n"
        "family5_classes: [A4C]\nraw9_classes: [A4C, A5C]\n"
    )
    with pytest.raises(LabelError):
        load_ontology(bad)
