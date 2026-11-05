"""True-effect filters for the label-assisted oracle. These use reference labels and can never be deployed."""
FILTERS = ('none', 'frame', 'frame_and_fp')
LITTLE_ROOM = .01


def keep(name, value):
    """value is A60's (utility, frame-error change, false-short change) of one action."""
    if name not in FILTERS:
        raise ValueError('declared safety filter required')
    if name == 'none':
        return True
    if name == 'frame':
        return bool(value[1] <= 0)
    return bool(value[1] <= 0 and value[2] <= 0)


def little_room(selection_gain, calibration_gain):
    """The pre-written rule: both real-order halves gain under .01 at the safe oracle."""
    return selection_gain < LITTLE_ROOM and calibration_gain < LITTLE_ROOM
