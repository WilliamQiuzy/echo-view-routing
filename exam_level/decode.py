"""The original B3 decoder applied unchanged, using the parameters frozen for each A08 seed."""
from echo_routing.temporal.baselines.b3_js_divergence import detect_boundaries, labels_from_boundaries


def decode_b3(prob, params):
    cuts = detect_boundaries(prob, int(params['half_window']), float(params['threshold']), int(params['nms_radius']))
    return labels_from_boundaries(prob, cuts)
