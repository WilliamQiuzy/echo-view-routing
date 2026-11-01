from echo_routing.temporal import direct_view_transfer as q


def example_metrics():
    base = {"balanced_accuracy": .70, "a4c_recall": .90, "a5c_recall": .50, "a4c_fpr": .10, "AUC": .80}
    good = {"balanced_accuracy": .90, "a4c_recall": .92, "a5c_recall": .88, "a4c_fpr": .08, "AUC": .93}
    return {"G": base, "L": good, "W": good.copy(), "geometry": {"balanced_accuracy": .65}, "constant": {"balanced_accuracy": .5}}


def test_common_source_signal_is_separate_from_label_increment():
    out = q.transfer_qualification(example_metrics(), original_label_gate=False)
    assert out["absolute_source_pass"] and out["target_transfer_eligible"]
    assert not out["original_A51P_label_increment_pass"]


def test_calibration_only_gain_cannot_pass_absolute_ranking_condition():
    m = example_metrics(); m["G"]["AUC"] = .93
    out = q.transfer_qualification(m, original_label_gate=False)
    assert not out["absolute_source_pass"] and not out["target_transfer_eligible"]


def test_original_preregistered_advancement_verdict_is_preserved():
    m = example_metrics(); m["G"]["AUC"] = .99
    out = q.transfer_qualification(m, original_label_gate=True)
    assert out["target_transfer_eligible"] and not out["absolute_source_pass"]


def test_source_a4c_damage_and_geometry_guards_are_required():
    m = example_metrics(); m["L"]["a4c_fpr"] = .13; m["W"]["a4c_fpr"] = .13
    assert not q.transfer_qualification(m, False)["absolute_source_pass"]
    m = example_metrics(); m["geometry"]["balanced_accuracy"] = .85
    assert not q.transfer_qualification(m, False)["absolute_source_pass"]
