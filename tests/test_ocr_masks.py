from pathlib import Path
import importlib.util
import numpy as np


def api():
    p = Path(__file__).parents[1] / "echo_routing/temporal/ocr_masks.py"
    assert p.exists(), "OCR coordinate implementation missing"
    spec = importlib.util.spec_from_file_location("ocr_masks", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_tsv_filter_is_fixed_and_does_not_promote_single_symbols():
    m = api()
    s = 'level\tleft\ttop\twidth\theight\tconf\ttext\n5\t10\t20\t30\t12\t91\tA5C\n5\t1\t2\t10\t10\t99\t+\n5\t1\t2\t10\t10\t99\tA\n5\t1\t2\t10\t10\t59\t72BPM\n'
    b = m.boxes_from_tsv(s)
    assert len(b) == 1 and b[0]['text'] == 'A5C'
    assert not m.boxes_from_tsv('left\ttop\twidth\theight\tconf\ttext\n1\t2\t3\t4\tnan\tA5C\n')


def test_native_boxes_map_through_the_actual_original_crop():
    m = api()
    center = [{'left': 300, 'top': 220, 'width': 40, 'height': 40}]
    corner = [{'left': 0, 'top': 0, 'width': 30, 'height': 30}]
    mask = m.native_to_input_mask((480, 640), center)
    assert mask.shape == (224, 224) and mask.dtype == bool
    assert mask[112, 112]
    assert not m.native_to_input_mask((480, 640), corner).any()


def test_scaled_boxes_are_clipped_and_empty_masks_stay_empty():
    m = api()
    b = [{'left': 300, 'top': 300, 'width': 60, 'height': 60}]
    mask = m.scaled_to_input_mask(b)
    assert mask[100, 100] and mask[119, 119]
    assert mask.sum() == 22 * 22
    assert not m.native_to_input_mask((480, 640), []).any()
    assert not m.scaled_to_input_mask([]).any()
