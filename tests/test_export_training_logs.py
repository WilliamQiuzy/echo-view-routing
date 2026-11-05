"""Tests for the public training-log export. Synthetic files only."""
import csv
import json

import pytest

from scripts.export_training_logs import clean_record, export, window_records


def write_log(root, relative, records):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))
    return path


class TestCleanRecord:
    def test_numbers_and_nested_numbers_stay(self):
        kept, dropped = clean_record({'epoch': 3, 'loss': {'ce': 1.5, 'parts': {'a': 2}}, 'partial': True, 'edge_acc': None})
        assert kept == {'epoch': 3, 'loss': {'ce': 1.5, 'parts': {'a': 2}}, 'partial': True, 'edge_acc': None} and dropped == []

    def test_digests_and_other_text_go(self):
        kept, dropped = clean_record({'loss': 1., 'input_digest': 'abc', 'note': 'x', 'monitor': {'path': '/home/a'}})
        assert kept == {'loss': 1., 'monitor': {}} and sorted(dropped) == ['input_digest', 'monitor.path', 'note']

    def test_lists_go(self):
        kept, dropped = clean_record({'loss': 1., 'ids': ['a', 'b']})
        assert kept == {'loss': 1.} and dropped == ['ids']

    def test_the_head_label_stays(self):
        assert clean_record({'head': 'switch'})[0] == {'head': 'switch'}

    @pytest.mark.parametrize('name', ['test', 'test_acc', 'acc_test', 'TEST_loss', 'macro_test_f1'])
    def test_fields_naming_the_test_split_go(self, name):
        kept, dropped = clean_record({name: .9, 'loss': 1.})
        assert kept == {'loss': 1.} and dropped == [name]

    def test_a_nested_test_block_goes_whole(self):
        kept, dropped = clean_record({'monitor': {'test': {'f1': .9}, 'f1': .8}})
        assert kept == {'monitor': {'f1': .8}} and dropped == ['monitor.test']

    @pytest.mark.parametrize('name', ['fastest_step', 'latest_loss', 'contest'])
    def test_words_that_only_contain_test_stay(self, name):
        assert clean_record({name: 1.})[0] == {name: 1.}

    @pytest.mark.parametrize('value', [float('nan'), float('inf'), float('-inf')])
    def test_non_finite_becomes_null(self, value):
        assert clean_record({'loss': value})[0] == {'loss': None}

    def test_the_input_is_not_changed(self):
        record = {'loss': {'ce': 1.}, 'input_digest': 'abc'}
        clean_record(record)
        assert record == {'loss': {'ce': 1.}, 'input_digest': 'abc'}


class TestWindowRecords:
    def test_a_long_log_becomes_about_fifty_means(self):
        records = [{'u': i, 'loss': float(i)} for i in range(120)]
        out, dropped = window_records(records, points=50)
        assert len(out) == 40 and dropped == []
        assert out[0] == {'first_record': 0, 'records': 3, 'loss': 1.0, 'u': 2}
        assert out[-1] == {'first_record': 117, 'records': 3, 'loss': 118.0, 'u': 119}

    def test_a_short_log_keeps_every_record(self):
        out, _ = window_records([{'step': i, 'loss': 1.5} for i in range(4)], points=50)
        assert [row['records'] for row in out] == [1, 1, 1, 1] and [row['step'] for row in out] == [0, 1, 2, 3]

    @pytest.mark.parametrize('name', ['u', 'step', 'update', 'updates', 'epoch'])
    def test_position_fields_give_the_last_value(self, name):
        out, _ = window_records([{name: 10}, {name: 20}], points=1)
        assert out[0][name] == 20 and isinstance(out[0][name], int)

    def test_only_top_level_numbers_pass(self):
        record = {'loss': 1., 'device': 'cuda', 'lr': [.1], 'groups': {'some_recording_name': 1}, 'plan_row_sha256': 'abc', 'pid': 7, 'edge_acc': None}
        out, dropped = window_records([record], points=50)
        assert out == [{'first_record': 0, 'records': 1, 'loss': 1.0}]
        assert sorted(dropped) == ['device', 'groups', 'lr', 'pid', 'plan_row_sha256']

    def test_a_field_naming_the_test_split_goes(self):
        out, dropped = window_records([{'test_acc': .9, 'loss': 1.}], points=50)
        assert 'test_acc' not in out[0] and dropped == ['test_acc']

    def test_a_field_missing_from_part_of_a_window_averages_what_is_there(self):
        out, _ = window_records([{'loss': 1.}, {'loss': 3., 'aux': 5.}], points=1)
        assert out[0]['loss'] == 2. and out[0]['aux'] == 5.

    def test_non_finite_values_are_counted_not_averaged(self):
        out, _ = window_records([{'loss': float('nan')}, {'loss': 2.}], points=1)
        assert out[0]['loss'] == 2. and out[0]['non_finite'] == 1

    def test_booleans_count_as_numbers(self):
        assert window_records([{'ok': True}, {'ok': False}], points=1)[0][0]['ok'] == .5

    def test_values_keep_six_significant_digits(self):
        assert window_records([{'loss': 1.23456789}], points=50)[0][0]['loss'] == 1.23457

    def test_an_empty_log_gives_nothing(self):
        assert window_records([], points=50) == ([], [])


class TestExport:
    def test_update_logs_export_as_windows(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, 'a57/seed0/F/physical_updates.jsonl', [{'u': i, 'loss': 1.} for i in range(200)])
        summary = export(runs, out, r'pediatric')
        assert len((out / 'a57/seed0/F/physical_updates.windows.jsonl').read_text().splitlines()) == 50
        with (out / 'index.csv').open(newline='') as handle:
            row = next(csv.DictReader(handle))
        assert row['kind'] == 'windows' and row['source_records'] == '200' and row['records'] == '50' and row['last_updates'] == '199'
        assert summary['source_records'] == 200 and summary['records'] == 50

    @pytest.mark.parametrize('source,target', [('a51/train/L/updates.private.jsonl', 'a51/train/L/updates.windows.jsonl'), ('pw/S.train.private.jsonl', 'pw/S.train.windows.jsonl'), ('v/train/U/updates.jsonl', 'v/train/U/updates.windows.jsonl')])
    def test_exported_names_do_not_say_private(self, tmp_path, source, target):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, source, [{'loss': 1.}])
        export(runs, out, r'pediatric')
        assert (out / target).is_file()

    def test_committed_records_are_not_training_logs(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, 'a57/committed.jsonl', [{'index': 1, 'positions': 2}])
        assert export(runs, out, r'pediatric')['logs'] == 0

    def test_a_manifest_supplies_epochs_when_nothing_else_does(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        (runs / 'v/train').mkdir(parents=True)
        (runs / 'v/train/run_manifest.json').write_text(json.dumps({'identity': 'x', 'final_checkpoint': '/home/a', 'epochs': [{'updates': 375, 'loss': 1., 'pixels_digest': 'abc'}]}))
        summary = export(runs, out, r'pediatric')
        assert json.loads((out / 'v/train/train_log.jsonl').read_text()) == {'updates': 375, 'loss': 1.} and summary['dropped_fields'] == {'pixels_digest': 1}

    def test_a_train_log_wins_over_a_manifest_in_the_same_folder(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, 'v/train_log.jsonl', [{'updates': 1, 'loss': 2.}])
        (runs / 'v/run_manifest.json').write_text(json.dumps({'epochs': [{'updates': 9, 'loss': 9.}]}))
        assert export(runs, out, r'pediatric')['logs'] == 1 and json.loads((out / 'v/train_log.jsonl').read_text())['updates'] == 1

    def test_a_manifest_without_epochs_is_ignored(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        (runs / 'v').mkdir(parents=True)
        (runs / 'v/run_manifest.json').write_text(json.dumps({'identity': 'x'}))
        assert export(runs, out, r'pediatric')['logs'] == 0

    def test_select_limits_the_runs(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, 'ex3_a/train_log.jsonl', [{'loss': 1.}])
        write_log(runs, 'video_a/train_log.jsonl', [{'loss': 1.}])
        summary = export(runs, out, r'pediatric', select=r'^(?!ex3_)')
        assert summary['logs'] == 1 and (out / 'video_a').exists() and not (out / 'ex3_a').exists()

    def test_mirrors_the_run_path_and_writes_an_index(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, 'video_a/train/s0/train_log.jsonl', [{'epoch': 1, 'updates': 10, 'loss': 2.}, {'epoch': 2, 'updates': 20, 'loss': 1., 'input_digest': 'x'}])
        summary = export(runs, out, r'pediatric')
        lines = [json.loads(line) for line in (out / 'video_a/train/s0/train_log.jsonl').read_text().splitlines()]
        assert lines == [{'epoch': 1, 'updates': 10, 'loss': 2.}, {'epoch': 2, 'updates': 20, 'loss': 1.}]
        with (out / 'index.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        assert rows == [{'run': 'video_a', 'log': 'video_a/train/s0/train_log.jsonl', 'kind': 'per_epoch', 'source_records': '2', 'records': '2', 'last_epoch': '2', 'last_updates': '20'}]
        assert summary['logs'] == 1 and summary['records'] == 2 and summary['dropped_fields'] == {'input_digest': 1}
        assert json.loads((out / 'export_summary.json').read_text()) == summary

    def test_a_log_without_positions_leaves_blank_cells(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, 'r/train_log.jsonl', [{'loss': 2.}])
        export(runs, out, r'pediatric')
        with (out / 'index.csv').open(newline='') as handle:
            row = next(csv.DictReader(handle))
        assert row['last_epoch'] == '' and row['last_updates'] == ''

    def test_a_step_counts_as_the_position(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, 'r/train_log.jsonl', [{'step': 7, 'loss': 2.}])
        export(runs, out, r'pediatric')
        with (out / 'index.csv').open(newline='') as handle:
            assert next(csv.DictReader(handle))['last_updates'] == '7'

    def test_withheld_runs_are_left_out_and_counted(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, 'video_a/train_log.jsonl', [{'loss': 1.}])
        write_log(runs, 'pediatric_adapt/s0/train_log.jsonl', [{'loss': 1.}])
        summary = export(runs, out, r'pediatric')
        assert not (out / 'pediatric_adapt').exists() and summary['logs'] == 1 and summary['withheld_logs'] == 1

    def test_a_subfolder_name_does_not_withhold_a_run(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, 'video_a/pediatric_note/train_log.jsonl', [{'loss': 1.}])
        assert export(runs, out, r'pediatric')['logs'] == 1

    def test_a_run_config_can_withhold_a_run_with_a_plain_name(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, '20261005-md_M1-s0/train_log.jsonl', [{'loss': 1.}])
        (runs / '20261005-md_M1-s0/config.resolved.yaml').write_text('data:\n  root: data/Pediatric\n')
        summary = export(runs, out, r'pediatric')
        assert summary['logs'] == 0 and summary['withheld_logs'] == 1 and not (out / '20261005-md_M1-s0').exists()

    def test_a_clean_config_keeps_the_run(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, 'video_a/train_log.jsonl', [{'loss': 1.}])
        (runs / 'video_a/config.resolved.yaml').write_text('data:\n  root: data/ev9v\n')
        assert export(runs, out, r'pediatric')['logs'] == 1

    @pytest.mark.parametrize('folder', ['cache', 'code', 'forward', 'evaluation', '__pycache__', 'engineering', 'smoke', 'smoke_s0'])
    def test_data_and_code_folders_are_skipped(self, tmp_path, folder):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, f'video_a/{folder}/train_log.jsonl', [{'loss': 1.}])
        assert export(runs, out, r'pediatric')['logs'] == 0

    def test_an_unparsable_line_is_an_error(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        path = write_log(runs, 'video_a/train_log.jsonl', [])
        path.write_text('not json\n')
        with pytest.raises(ValueError, match='video_a/train_log.jsonl'):
            export(runs, out, r'pediatric')

    def test_the_source_files_are_not_changed(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        path = write_log(runs, 'video_a/train_log.jsonl', [{'loss': 1., 'input_digest': 'x'}])
        before = path.read_bytes()
        export(runs, out, r'pediatric')
        assert path.read_bytes() == before

    def test_a_used_output_folder_is_refused(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, 'video_a/train_log.jsonl', [{'loss': 1.}])
        out.mkdir()
        (out / 'old.txt').write_text('x')
        with pytest.raises(ValueError, match='not empty'):
            export(runs, out, r'pediatric')

    def test_nothing_private_reaches_the_summary(self, tmp_path):
        runs, out = tmp_path / 'runs', tmp_path / 'out'
        write_log(runs, 'video_a/train_log.jsonl', [{'loss': 1.}])
        assert str(tmp_path) not in (out.parent / 'runs' / 'video_a' / 'train_log.jsonl').read_text()
        summary = export(runs, out, r'pediatric')
        assert str(tmp_path) not in json.dumps(summary)
