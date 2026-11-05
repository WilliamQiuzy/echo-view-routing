"""Pure-logic tests for EX2. They need no data, label table or model."""
import numpy as np
import pytest

from open_routing import action_budget as ab, openset as os_, quality as ql, reading as rd, retune as rt, safe as sf


class TestAuroc:
    def test_perfect_separation(self):
        assert os_.auroc([3., 4.], [1., 2.]) == 1.

    def test_reverse_separation(self):
        assert os_.auroc([1., 2.], [3., 4.]) == 0.

    def test_all_tied_is_half(self):
        assert os_.auroc([1., 1.], [1., 1.]) == .5

    def test_matches_pair_count(self):
        a, b = np.array([1., 3., 5., 2.]), np.array([2., 4.])
        wins = (a[:, None] > b[None]) + .5 * (a[:, None] == b[None])
        assert os_.auroc(a, b) == pytest.approx(wins.mean())


class TestScores:
    def test_msp_takes_largest_probability(self):
        assert os_.msp(np.array([[.7, .2, .1, 0, 0]])).tolist() == [.7]

    def test_merged_msp_adds_a4c_and_a5c(self):
        assert os_.merged_msp(np.array([[.3, .1, .3, .3, 0]]))[0] == pytest.approx(.6)

    def test_energy_is_mean_logsumexp_over_windows(self):
        logits = np.log(np.array([[.5, .5, 1e-12, 1e-12, 1e-12], [1., 1e-12, 1e-12, 1e-12, 1e-12]]))
        assert os_.energy_score(logits) == pytest.approx(0., abs=1e-9)

    def test_energy_higher_for_larger_logits(self):
        assert os_.energy_score(np.full((2, 5), 3.)) > os_.energy_score(np.full((2, 5), 1.))

    def test_knn_prefers_the_nearby_query(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(size=(50, 8)) * .01 + np.eye(8)[0]
        near, far = reference[0] + .001, np.eye(8)[1]
        scores = os_.knn_score(reference, np.stack([near, far]))
        assert scores[0] > scores[1]

    def test_knn_rejects_k_larger_than_reference(self):
        with pytest.raises(ValueError):
            os_.knn_score(np.eye(3), np.eye(3), k=4)

    def test_mahalanobis_prefers_a_class_mean(self):
        rng = np.random.default_rng(0)
        centres = np.eye(5, 20) * 5
        labels = np.repeat(np.arange(5), 40)
        features = centres[labels] + rng.normal(size=(200, 20)) * .3
        model = os_.fit_mahalanobis(features, labels, dims=6)
        assert os_.mahalanobis_score(model, centres[:1])[0] > os_.mahalanobis_score(model, np.full((1, 20), 20.))[0]


class TestThreshold:
    def test_retention_is_at_least_requested(self):
        scores = np.arange(100.)
        threshold = os_.threshold_for_retention(scores, .95)
        assert (scores >= threshold).mean() >= .95

    def test_retention_is_tight(self):
        scores = np.arange(100.)
        threshold = os_.threshold_for_retention(scores, .95)
        assert (scores >= threshold).mean() == .95

    def test_rejects_bad_retention(self):
        with pytest.raises(ValueError):
            os_.threshold_for_retention(np.arange(3.), 1.2)

    def test_paired_bootstrap_uses_whole_patients(self):
        inside, outside = np.arange(20.) + 100, np.arange(20.)
        ci = os_.paired_auroc_ci(inside, outside, n_boot=50, seed=0)
        assert ci == [1., 1.]

    def test_paired_bootstrap_requires_pairs(self):
        with pytest.raises(ValueError):
            os_.paired_auroc_ci(np.arange(3.), np.arange(4.))

    def test_usable_rule_requires_every_condition(self):
        good = {'auroc': .95, 'camus_2ch_accept': .1, 'camus_4ch_retain': .8, 'dynamic_retain': .9, 'lvh_retain': .9}
        assert os_.usable([good, good, good])
        assert not os_.usable([good, good, {**good, 'lvh_retain': .5}])
        assert not os_.usable([good, good, {**good, 'camus_2ch_accept': .3}])


class TestRetune:
    def test_grid_has_declared_size(self):
        assert len(rt.settings()) == 40

    def test_feasible_setting_wins_over_higher_infeasible(self):
        rows = [{'setting': 0, 'segment_f1': .99, 'native_macro_f1': .9, 'change_recall': .99},
                {'setting': 1, 'segment_f1': .95, 'native_macro_f1': .99, 'change_recall': .99}]
        assert rt.select(rows)['setting'] == 1

    def test_tie_goes_to_the_earlier_entry(self):
        rows = [{'setting': i, 'segment_f1': .95, 'native_macro_f1': .99, 'change_recall': .99} for i in (0, 1)]
        assert rt.select(rows)['setting'] == 0

    def test_no_feasible_setting_is_marked(self):
        rows = [{'setting': 0, 'segment_f1': .9, 'native_macro_f1': .5, 'change_recall': .5},
                {'setting': 1, 'segment_f1': .8, 'native_macro_f1': .5, 'change_recall': .5}]
        chosen = rt.select(rows)
        assert chosen['setting'] == 0 and chosen['infeasible']


class TestSafeFilter:
    @pytest.mark.parametrize('name,value,expected', [
        ('none', (0, 5, 5), True), ('frame', (0, 0, 5), True), ('frame', (0, 1, 0), False),
        ('frame_and_fp', (0, 0, 0), True), ('frame_and_fp', (0, -2, 1), False), ('frame_and_fp', (0, 1, -1), False)])
    def test_keep(self, name, value, expected):
        assert sf.keep(name, value) is expected

    def test_unknown_filter_refused(self):
        with pytest.raises(ValueError):
            sf.keep('other', (0, 0, 0))

    def test_rule_needs_both_halves(self):
        assert sf.little_room(.009, .001)
        assert not sf.little_room(.012, .001)


class TestActionBudget:
    def test_chunks_keep_whole_clips_and_cover_everything(self):
        spans = [('a', 0, 60), ('b', 60, 140), ('c', 140, 270), ('d', 270, 290)]
        assert ab.chunk_edges(spans, 124) == [(0, 140), (140, 290)]

    def test_short_stream_is_one_chunk(self):
        assert ab.chunk_edges([('a', 0, 50)], 124) == [(0, 50)]

    def test_trailing_remainder_joins_the_last_chunk(self):
        assert ab.chunk_edges([('a', 0, 130), ('b', 130, 140)], 124) == [(0, 140)]

    def test_survives_rule(self):
        cell = lambda d, lo, f: {'delta': d, 'ci_low': lo, 'frame_errors': f}
        assert ab.survives([cell(.01, .001, 0), cell(.01, .001, -1), cell(.01, -.1, 0), cell(.01, -.1, 0)])
        assert not ab.survives([cell(.01, .001, 0), cell(.01, .001, 1), cell(.01, -.1, 0), cell(.01, -.1, 0)])
        assert not ab.survives([cell(.01, .001, 0), cell(.01, -.1, 0), cell(.01, -.1, 0), cell(.01, -.1, 0)])
        assert not ab.survives([cell(.01, .001, 0), cell(.01, .001, 0), cell(.01, -.1, 0), cell(0., -.1, 0)])


class TestQuality:
    def test_dhash_is_64_bits(self):
        assert isinstance(ql.dhash(np.random.default_rng(0).integers(0, 255, (240, 320), np.uint8)), int)
        assert ql.dhash(np.zeros((240, 320), np.uint8)) < 2 ** 64

    def test_identical_images_have_zero_distance(self):
        image = np.random.default_rng(0).integers(0, 255, (240, 320), np.uint8)
        assert ql.hamming(np.array([ql.dhash(image)], np.uint64), np.array([ql.dhash(image)], np.uint64))[0, 0] == 0

    def test_hamming_counts_bits(self):
        assert ql.hamming(np.array([0b1011], np.uint64), np.array([0], np.uint64))[0, 0] == 3

    def test_ncc_of_identical_is_one(self):
        image = np.random.default_rng(0).integers(0, 255, (240, 320), np.uint8)
        assert ql.ncc(image, image) == pytest.approx(1.)

    def test_ncc_of_inverse_is_minus_one(self):
        image = np.random.default_rng(0).integers(0, 255, (240, 320), np.uint8).astype(float)
        assert ql.ncc(image, 255 - image) == pytest.approx(-1.)

    def test_static_flag(self):
        frames = np.zeros((4, 24, 32), np.uint8)
        assert ql.is_static(frames)
        frames[1::2] = 200
        assert not ql.is_static(frames)

    def test_colour_share_counts_saturated_pixels(self):
        frames = np.zeros((1, 10, 10, 3), np.uint8)
        frames[0, 0, 0] = (200, 0, 0)
        assert ql.colour_share(frames) == pytest.approx(.01)

    def test_flag_thresholds(self):
        assert ql.is_colour(.02) and not ql.is_colour(.019)
        assert ql.is_short(15) and not ql.is_short(16)
        assert ql.static_from_differences([.5, .5, 9.]) and not ql.static_from_differences([.5, 2., 9.])

    OWNERS = [{'split': 'train', 'half': None, 'stamp': 's1', 'code': 'A4C', 'family': 'A4C'},
              {'split': 'validation', 'half': 'selection', 'stamp': 's2', 'code': 'A5C', 'family': 'A5C'},
              {'split': 'validation', 'half': 'calibration', 'stamp': 's2', 'code': 'PASA', 'family': 'PSAX'},
              {'split': 'validation', 'half': 'calibration', 'stamp': 's3', 'code': 'PMASA', 'family': 'PSAX'},
              {'split': 'validation', 'half': 'calibration', 'stamp': 's4', 'code': 'PMPALA', 'family': None}]

    def test_describe_handles_a_recording_without_a_half(self):
        row = ql.describe_pairs([(0, 1, .99)], self.OWNERS)
        assert row['cross_split_train_validation'] == 1 and row['cross_half'] == 0
        assert row['family_conflict'] == 1 and row['a4c_a5c_conflict'] == 1

    def test_codes_in_one_family_are_not_a_family_conflict(self):
        row = ql.describe_pairs([(2, 3)], self.OWNERS)
        assert row['raw_code_conflict'] == 1 and row['family_conflict'] == 0

    def test_excluded_code_is_not_a_family_conflict(self):
        assert not ql.family_conflict(self.OWNERS[3], self.OWNERS[4])

    def test_cross_half_counted(self):
        assert ql.describe_pairs([(1, 2)], self.OWNERS)['cross_half'] == 1

    def test_candidates_cross_sessions_only_and_once(self):
        values = np.array([0, 0, 0, 2 ** 40 - 1], np.uint64)
        owners = np.array([0, 0, 1, 2])
        found = ql.near_duplicate_candidates(values, owners, np.array(['a', 'b', 'c']), block=2)
        assert set(found) == {(0, 1)} and sorted(found[(0, 1)]) == [(0, 2), (1, 2)]

    def test_candidates_skip_the_same_session(self):
        values = np.array([0, 0], np.uint64)
        assert not ql.near_duplicate_candidates(values, np.array([0, 1]), np.array(['a', 'a']))


class TestReading:
    def test_blind_ids_are_unique_and_carry_no_position(self):
        ids = rd.blind_ids(100, seed=1)
        assert len(set(ids)) == 100 and all(len(i) == 8 for i in ids)
        assert ids != rd.blind_ids(100, seed=2)

    def test_frame_choice_is_ordered_and_inside(self):
        picks = rd.frame_picks(100)
        assert len(picks) == 4 and picks == sorted(picks) and 0 <= picks[0] and picks[-1] < 100

    def test_frame_choice_on_short_clip(self):
        assert all(0 <= p < 3 for p in rd.frame_picks(3))

    def test_kappa_perfect_and_chance(self):
        assert rd.cohen_kappa(['a', 'b', 'a', 'b'], ['a', 'b', 'a', 'b']) == 1.
        assert rd.cohen_kappa(['a', 'a', 'b', 'b'], ['a', 'b', 'a', 'b']) == pytest.approx(0.)

    def test_kappa_undefined_when_one_category(self):
        assert rd.cohen_kappa(['a', 'a'], ['a', 'a']) is None

    def test_wilson_contains_rate(self):
        low, high = rd.wilson(3, 10)
        assert low < .3 < high

    def test_wilson_empty(self):
        assert rd.wilson(0, 0) == [None, None]

    def test_flagged_needs_both_readers_against_label(self):
        assert rd.both_disagree('A5C', 'A4C', 'A4C')
        assert not rd.both_disagree('A5C', 'A4C', 'A5C')
        assert not rd.both_disagree('A5C', 'uncertain', 'A4C')
        assert rd.both_disagree('A5C', 'other', 'A4C')

    def test_signal_rule(self):
        assert rd.label_signal(30, 90, 1, 30)
        assert not rd.label_signal(3, 90, 1, 30)

    def test_repeat_pairs_survive_any_key_order(self):
        entries = {'zz': {'video_id': 'b', 'repeat_of': None}, 'aa': {'video_id': 'a', 'repeat_of': None},
                   'mm': {'video_id': 'a', 'repeat_of': 1}, 'cc': {'video_id': 'b', 'repeat_of': 0}}
        assert sorted(rd.repeat_pairs(entries)) == [('aa', 'mm'), ('zz', 'cc')]
        assert sorted(rd.repeat_pairs(dict(sorted(entries.items())))) == [('aa', 'mm'), ('zz', 'cc')]

    def test_unpack_answers(self):
        raw = {'x': {'answer': 'A5C', 'lvot_frames': 2, 'text_seen': False}, 'y': {'answer': 'A4C', 'lvot_frames': 0, 'text_seen': True}}
        answers, frames, text = rd.unpack_answers(raw)
        assert answers == {'x': 'A5C', 'y': 'A4C'} and frames == {'x': 2, 'y': 0} and text == ['y']

    @pytest.mark.parametrize('record', [{'answer': 'A4C', 'lvot_frames': 5, 'text_seen': False}, {'answer': 'A4C', 'lvot_frames': 0, 'text_seen': 'no'}])
    def test_unpack_refuses_bad_records(self, record):
        with pytest.raises(ValueError):
            rd.unpack_answers({'x': record})

    def test_inconsistent_records(self):
        answers = {'a': 'A5C', 'b': 'A4C', 'c': 'A5C', 'd': 'A4C', 'e': 'other'}
        frames = {'a': 0, 'b': 1, 'c': 2, 'd': 0, 'e': 3}
        assert rd.inconsistent_records(answers, frames) == ['a', 'b']

    def test_answers_are_validated(self):
        with pytest.raises(ValueError):
            rd.check_answers({'x': 'maybe'}, ['x'])
        with pytest.raises(ValueError):
            rd.check_answers({'x': 'A4C'}, ['x', 'y'])
        assert rd.check_answers({'x': 'A4C'}, ['x']) == {'x': 'A4C'}
