"""Pure-logic tests for EX3. They need no dataset, label table or model."""
import numpy as np
import pytest
from PIL import Image

from echo_routing.temporal.joint_visual import crop_uint8
from helena_input import data, geometry as gm, pixels as px, verdict as vd


def noise(height, width, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (height, width, 3), np.uint8)


class TestZoomCrop:
    @pytest.mark.parametrize('height,width', [(240, 320), (112, 112), (768, 1024), (487, 591)])
    def test_reproduces_a08_at_a08_centre(self, height, width):
        image = noise(height, width)
        cx, cy = gm.a08_centre(width, height)
        out = gm.zoom_crop(image, cx, cy, gm.a08_side(width, height))
        assert np.array_equal(out, crop_uint8(Image.fromarray(image)))

    def test_a08_side_on_ev9v(self):
        assert gm.a08_side(320, 240) == 210.

    def test_centre_shift_moves_content(self):
        image = noise(240, 320)
        base = gm.zoom_crop(image, 160, 120, 210)
        moved = gm.zoom_crop(image, 160, 130, 210)
        assert not np.array_equal(base, moved)

    def test_zero_padding_outside_the_frame(self):
        out = gm.zoom_crop(np.full((100, 100, 3), 200, np.uint8), 0, 0, 100, size=100)
        assert out[0, 0].tolist() == [0, 0, 0] and out[99, 99].tolist() == [200, 200, 200]

    def test_a_crop_off_the_frame_is_black(self):
        assert gm.zoom_crop(noise(50, 50), 1000, 1000, 50, size=50).sum() == 0

    def test_refuses_nonpositive_side(self):
        with pytest.raises(ValueError):
            gm.zoom_crop(noise(10, 10), 5, 5, 0)


class TestMasks:
    def test_round_trip(self):
        mask = np.random.default_rng(1).random((37, 53)) > .5
        assert np.array_equal(gm.unpack_mask(gm.pack_mask(mask)), mask)

    def test_apply_keeps_inside_and_zeroes_outside(self):
        image = np.full((2, 2, 3), 9, np.uint8)
        out = gm.apply_mask(image, np.array([[True, False], [False, True]]))
        assert out[0, 0, 0] == 9 and out[0, 1, 0] == 0

    def test_apply_refuses_size_mismatch(self):
        with pytest.raises(ValueError):
            gm.apply_mask(noise(2, 2), np.ones((3, 3), bool))

    def test_all_ones_mask_gives_a08_pixels(self):
        image = noise(240, 320)
        segments = [{'start': 0, 'end': 10, 'mask': gm.pack_mask(np.ones((240, 320), bool))}]
        cx, cy = gm.a08_centre(320, 240)
        out = gm.z_frame(image, segments, cx, cy, 210., 3)
        assert np.array_equal(out, crop_uint8(Image.fromarray(image)))


class TestSegments:
    SEGMENTS = [{'start': 0, 'end': 5, 'mask': 'a'}, {'start': 5, 'end': 9, 'mask': 'b'}]

    def test_picks_the_right_segment(self):
        assert gm.segment_at(self.SEGMENTS, 4)['mask'] == 'a' and gm.segment_at(self.SEGMENTS, 5)['mask'] == 'b'

    def test_past_the_end_belongs_to_the_last(self):
        assert gm.segment_at(self.SEGMENTS, 50)['mask'] == 'b'

    def test_a_gap_is_an_error(self):
        with pytest.raises(ValueError):
            gm.segment_at([{'start': 3, 'end': 5, 'mask': 'a'}], 1)

    def test_empty_is_an_error(self):
        with pytest.raises(ValueError):
            gm.segment_at([], 0)


class TestRatio:
    def test_uses_the_median(self):
        assert gm.side_ratio(210., [100., 300., 420.]) == 210. / 300.

    def test_refuses_empty(self):
        with pytest.raises(ValueError):
            gm.side_ratio(210., [])


class TestFullFrame:
    def test_output_is_square_and_keeps_the_frame(self):
        out = gm.full_frame_square(np.full((100, 200, 3), 200, np.uint8), size=100)
        assert out.shape == (100, 100, 3) and out[50, 50, 0] == 200 and out[5, 50, 0] == 0


class TestDirection:
    def test_all_above(self):
        assert vd.direction([(.1, .2), (.01, .3)]) == 1

    def test_all_below(self):
        assert vd.direction([(-.2, -.1), (-.3, -.01)]) == -1

    def test_mixed_or_crossing(self):
        assert vd.direction([(.1, .2), (-.1, .3)]) == 0

    def test_touching_zero_is_not_consistent(self):
        assert vd.direction([(0., .2)]) == 0


class TestReadings:
    UP, DOWN, CROSS = [(.1, .2)] * 3, [(-.2, -.1)] * 3, [(-.1, .1)] * 3

    def test_ev9v_better_needs_one_positive_half(self):
        assert vd.ev9v_better({'selection': self.UP, 'calibration': self.CROSS}, True)

    def test_ev9v_better_fails_when_the_other_half_is_negative(self):
        assert not vd.ev9v_better({'selection': self.UP, 'calibration': self.DOWN}, True)

    def test_ev9v_better_fails_on_a_floor(self):
        assert not vd.ev9v_better({'selection': self.UP, 'calibration': self.UP}, False)

    def test_floors(self):
        row = {'half': 'selection', 'boundary_recall': .96, 'native_macro_f1': .973}
        assert vd.floors_hold([row]) and not vd.floors_hold([{**row, 'boundary_recall': .94}])
        assert not vd.floors_hold([{**row, 'half': 'calibration'}])

    def test_external_better(self):
        assert vd.external_better({'camus': self.UP, 'dynamic': self.CROSS})
        assert not vd.external_better({'camus': self.UP, 'dynamic': self.DOWN})
        assert not vd.external_better({'camus': self.CROSS})

    def test_reading_order(self):
        assert vd.reading([1, 0], True, False, False) == 'ev9v_better'
        assert vd.reading([-1, -1], False, True, False) == 'trade_off'
        assert vd.reading([0, 0], False, True, False) == 'external_better'
        assert vd.reading([0, 0], False, False, True) == 'open_set_improved'
        assert vd.reading([-1, 0], False, False, False) == 'no_gain'


class TestStop:
    BELOW = [(-.05, -.01), (-.04, -.02)]

    def test_stops_only_when_all_three_fail(self):
        assert vd.stop_after_first_seed(self.BELOW, [(-.1, .1)], False)

    def test_continues_on_a_crossing_half(self):
        assert not vd.stop_after_first_seed([(-.05, .01), (-.04, -.02)], [(-.1, .1)], False)

    def test_continues_on_external_gain(self):
        assert not vd.stop_after_first_seed(self.BELOW, [(.01, .1)], False)

    def test_continues_on_a_usable_score(self):
        assert not vd.stop_after_first_seed(self.BELOW, [(-.1, .1)], True)


class TestFallbackKind:
    def test_failed_outranks_everything(self):
        assert vd.fallback_kind({'failed': 'ValueError: x', 'segments': [{'fallback': 'geometry_fallback'}]}) == 'failed'

    def test_envelope_outranks_geometry_across_segments(self):
        assert vd.fallback_kind({'failed': None, 'segments': [{'fallback': 'geometry_fallback'}, {'fallback': 'dynamic_envelope_fallback'}]}) == 'dynamic_envelope_fallback'

    def test_one_segment_is_enough(self):
        assert vd.fallback_kind({'failed': None, 'segments': [{'fallback': ''}, {'fallback': 'geometry_fallback'}]}) == 'geometry_fallback'

    def test_no_fallback(self):
        assert vd.fallback_kind({'failed': None, 'segments': [{'fallback': ''}]}) == 'none'


class TestStillPossible:
    BELOW, CROSSES, ABOVE = (-.05, -.01), (-.01, .01), (.01, .05)

    def possible(self, dev, floors, external, usable):
        return vd.still_possible_after_first_seed(dev, floors, external, usable)

    def test_an_interval_crossing_zero_already_rules_out_consistently_positive(self):
        out = self.possible({'selection': self.CROSSES, 'calibration': self.CROSSES}, True, {'e': self.CROSSES}, False)
        assert out == {'ev9v_better': False, 'external_better': False, 'open_set_improved': False}

    def test_ev9v_needs_one_half_above_zero_and_the_floors(self):
        dev = {'selection': self.ABOVE, 'calibration': self.BELOW}
        assert self.possible(dev, True, {'e': self.BELOW}, False)['ev9v_better'] is True
        assert self.possible(dev, False, {'e': self.BELOW}, False)['ev9v_better'] is False

    def test_external_needs_one_endpoint_above_zero(self):
        dev = {'selection': self.BELOW, 'calibration': self.BELOW}
        assert self.possible(dev, True, {'a': self.BELOW, 'b': self.ABOVE}, False)['external_better'] is True

    def test_open_set_follows_the_first_seed(self):
        dev = {'selection': self.BELOW, 'calibration': self.BELOW}
        assert self.possible(dev, True, {'e': self.BELOW}, True)['open_set_improved'] is True


class TestMergeStored:
    ONE = {'results': {'selection': {'H|0': 1, 'Z|0': 2}, 'calibration': {'H|0': 3, 'Z|0': 4}}}
    TWO = {'results': {'selection': {'Z|1': 5}, 'calibration': {'Z|1': 6}}}

    def test_dev_files_combine_within_each_half(self):
        out = vd.merge_dev([self.ONE, self.TWO])['results']
        assert out == {'selection': {'H|0': 1, 'Z|0': 2, 'Z|1': 5}, 'calibration': {'H|0': 3, 'Z|0': 4, 'Z|1': 6}}

    def test_dev_merge_leaves_its_inputs_alone(self):
        vd.merge_dev([self.ONE, self.TWO])
        assert self.ONE['results']['selection'] == {'H|0': 1, 'Z|0': 2}

    def test_the_same_dev_row_twice_is_an_error(self):
        with pytest.raises(ValueError):
            vd.merge_dev([self.ONE, self.ONE])

    def test_a_missing_half_is_an_error(self):
        with pytest.raises(KeyError):
            vd.merge_dev([{'results': {'selection': {'Z|0': 1}}}])

    def test_external_files_combine_by_tag(self):
        out = vd.merge_external([{'tags': {'Z_s0': 1}}, {'tags': {'Z_s1': 2}}])
        assert out == {'tags': {'Z_s0': 1, 'Z_s1': 2}}

    def test_the_same_tag_twice_is_an_error(self):
        with pytest.raises(ValueError):
            vd.merge_external([{'tags': {'Z_s0': 1}}, {'tags': {'Z_s0': 2}}])

    def test_no_files_is_an_error(self):
        with pytest.raises(ValueError):
            vd.merge_dev([])
        with pytest.raises(ValueError):
            vd.merge_external([])


class TestAllSeedPairs:
    SYSTEM = np.ones(200)
    HALF = np.r_[np.ones(100), np.zeros(100)]

    def test_every_system_seed_meets_every_reference_seed(self):
        out = vd.all_seed_pairs({0: self.SYSTEM, 1: self.SYSTEM}, {0: self.HALF, 1: self.HALF, 2: self.HALF}, np.arange(200))
        assert [(p['system_seed'], p['reference_seed']) for p in out['pairs']] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    def test_one_strong_reference_seed_breaks_the_count(self):
        out = vd.all_seed_pairs({0: self.SYSTEM}, {0: self.HALF, 1: np.ones(200), 2: np.zeros(200)}, np.arange(200))
        assert out['above_zero'] == 2 and out['below_zero'] == 0 and len(out['pairs']) == 3

    def test_a_system_below_every_reference_counts_below(self):
        out = vd.all_seed_pairs({0: np.zeros(200)}, {0: self.HALF, 1: np.ones(200)}, np.arange(200))
        assert out['above_zero'] == 0 and out['below_zero'] == 2

    def test_empty_seeds_are_an_error(self):
        with pytest.raises(ValueError):
            vd.all_seed_pairs({}, {0: self.HALF}, np.arange(200))


class TestCpuLog:
    def test_user_and_system_time_add_up_by_stage(self):
        out = vd.parse_cpu_log('train\t3600\t1800\t100\ndecide\t0\t3600\t10\n')
        assert out['stages']['train']['cpu_hours'] == 1.5 and out['stages']['train']['wall_hours'] == 100 / 3600
        assert out['cpu_hours'] == 2.5

    def test_a_retried_stage_keeps_both_runs(self):
        out = vd.parse_cpu_log('Command exited with non-zero status 1\ntrain\t3600\t0\t10\ntrain\t3600\t0\t10\n')
        assert out['stages']['train']['cpu_hours'] == 2. and out['cpu_hours'] == 2.

    def test_a_signal_line_is_skipped(self):
        assert vd.parse_cpu_log('Command terminated by signal 9\ntrain\t3600\t0\t10\n')['cpu_hours'] == 1.

    @pytest.mark.parametrize('text', ['noise line\n', 'train\t1\t2\n', 'stage\tuser\tsystem\twall\n', 'train\t1\t2\t3\n\n', ''])
    def test_anything_else_is_an_error(self, text):
        with pytest.raises(ValueError):
            vd.parse_cpu_log(text)


class TestDevResultName:
    def test_all_arms_keep_the_original_name(self):
        assert data.dev_result_name('trained', data.ARMS, [0]) == 'dev_trained_s0.json'
        assert data.dev_result_name('swap', data.ARMS, [0, 1, 2]) == 'dev_swap_s0_s1_s2.json'

    def test_one_arm_is_named(self):
        assert data.dev_result_name('trained', ['Z'], [1, 2]) == 'dev_trained_Z_s1_s2.json'


class TestPairedRecall:
    def test_identical_systems_have_zero_difference(self):
        a = [1, 0, 1, 1]
        result = vd.paired_recall_delta(a, a, [0, 1, 2, 3])
        assert result['delta'] == 0. and result['ci95'] == [0., 0.]

    def test_clear_gain_has_interval_above_zero(self):
        a, b = np.ones(200), np.r_[np.ones(100), np.zeros(100)]
        result = vd.paired_recall_delta(a, b, np.arange(200))
        assert result['delta'] == .5 and result['ci95'][0] > 0

    def test_clusters_are_kept_whole(self):
        a, b = np.array([1, 1, 0, 0.]), np.zeros(4)
        result = vd.paired_recall_delta(a, b, ['x', 'x', 'y', 'y'], n_boot=2000)
        assert result['clusters'] == 2 and set(np.round(result['ci95'], 6)) <= {0., 1.}

    def test_refuses_unequal_lengths(self):
        with pytest.raises(ValueError):
            vd.paired_recall_delta([1], [1, 0], [0])


class TestGuards:
    def test_empty_inputs_are_errors(self):
        with pytest.raises(ValueError):
            vd.floors_hold([])
        with pytest.raises(ValueError):
            vd.external_better({})
        with pytest.raises(ValueError):
            vd.stop_after_first_seed([(-.1, -.01)], [(-.1, .1)], False)

    def test_identifier_scan_finds_nested_names(self):
        assert data.find_identifier({'a': [{'b': 'x_2022-03-30_y'}]}, {'2022-03-30'}) == '2022-03-30'

    def test_identifier_scan_passes_clean_values(self):
        assert data.find_identifier({'a': 1.5, 'b': 'selection'}, {'2022-03-30', ''}) is None


class TestPixelFallbacks:
    @staticmethod
    def h_source(tmp_path, videos):
        import json
        (tmp_path / 'index.json').write_text(json.dumps({'videos': videos}))
        return px.HelenaPixels(tmp_path, lambda _: [])

    def test_unknown_recording_is_an_error_in_h(self, tmp_path):
        with pytest.raises(KeyError):
            self.h_source(tmp_path, {}).frame('never_prepared', 0, 1.)

    def test_unknown_recording_is_an_error_in_z(self, tmp_path):
        import pickle
        path = tmp_path / 'index.pkl'
        path.write_bytes(pickle.dumps({}))
        with pytest.raises(KeyError):
            px.ZoomPixels(path, lambda _: []).frame('never_prepared', 0, 1.)

    def test_declared_fallback_uses_the_a08_crop_in_both(self, tmp_path):
        import pickle
        image = noise(240, 320)
        jpeg = tmp_path / '0.png'
        Image.fromarray(image).save(jpeg)
        expected = crop_uint8(Image.fromarray(image))
        h = self.h_source(tmp_path, {'v': {'fallback': 'no_provider_row'}})
        h.raw.frame_paths = lambda _: [jpeg]
        assert np.array_equal(h.frame('v', 0, 1.), expected)
        path = tmp_path / 'index.pkl'
        path.write_bytes(pickle.dumps({'v': {'fallback': 'no_provider_row'}}))
        z = px.ZoomPixels(path, lambda _: [jpeg])
        assert np.array_equal(z.frame('v', 0, 1.), expected)

    def test_z_with_neutral_entry_equals_a08_on_a_real_file(self, tmp_path):
        import pickle
        image = noise(240, 320)
        png = tmp_path / '0.png'
        Image.fromarray(image).save(png)
        cx, cy = gm.a08_centre(320, 240)
        entry = {'cx': cx, 'cy': cy, 'segments': [{'start': 0, 'end': 1, 'mask': gm.pack_mask(np.ones((240, 320), bool))}]}
        path = tmp_path / 'index.pkl'
        path.write_bytes(pickle.dumps({'v': entry}))
        assert np.array_equal(px.ZoomPixels(path, lambda _: [png]).frame('v', 0, 1.), crop_uint8(Image.fromarray(image)))


def test_distinguished_counts_a_patient_only_when_the_4ch_is_read_and_the_2ch_is_not():
    assert vd.distinguished([1, 1, 0, 0], [0, 1, 0, 1]).tolist() == [1., 0., 0., 0.]


def test_distinguished_refuses_unequal_lengths():
    with pytest.raises(ValueError):
        vd.distinguished([1, 0], [1])


def test_auroc_of_matches_counting_every_pair_with_ties_as_half():
    positive, negative = [.9, .5, .2], [.5, .1]
    pairs = [1. if p > n else .5 if p == n else 0. for p in positive for n in negative]
    assert vd.auroc_of(positive, negative) == pytest.approx(sum(pairs) / len(pairs))


def test_auroc_of_refuses_an_empty_group():
    with pytest.raises(ValueError):
        vd.auroc_of([], [.1])


def test_paired_auroc_delta_is_exactly_zero_for_one_system_against_itself():
    positive, negative = np.array([.9, .8, .4, .7]), np.array([.2, .6, .5, .1])
    row = vd.paired_auroc_delta(positive, negative, positive, negative, n_boot=200)
    assert row['delta'] == 0. and row['ci95'] == [0., 0.]


def test_paired_auroc_delta_is_above_zero_when_one_system_separates_the_views():
    rng = np.random.default_rng(0)
    good = rng.normal(2., .3, 80), rng.normal(0., .3, 80)
    poor = rng.normal(0., 1., 80), rng.normal(0., 1., 80)
    row = vd.paired_auroc_delta(*good, *poor, n_boot=500)
    assert row['delta'] > .2 and row['ci95'][0] > 0


def test_paired_auroc_delta_refuses_patients_that_do_not_line_up():
    with pytest.raises(ValueError):
        vd.paired_auroc_delta([.1, .2], [.3], [.1, .2], [.3, .4])
