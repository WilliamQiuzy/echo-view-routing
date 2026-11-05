"""Selective prediction on clip-level mean probabilities. Empirical curves, not finite-sample guarantees."""
import numpy as np
from scipy.stats import beta

A4C, A5C = 2, 3


def merged_view(prob):
    """Five classes to PLAX, PSAX, apical (A4C + A5C), SC4C."""
    p = np.asarray(prob, float)
    return np.stack([p[..., 0], p[..., 1], p[..., A4C] + p[..., A5C], p[..., 4]], axis=-1)


def merged_labels(labels):
    return np.array([0, 1, 2, 2, 3])[np.asarray(labels)]


def clip_decision(mean_prob, merged):
    p = merged_view(mean_prob) if merged else np.asarray(mean_prob, float)
    return {'label': p.argmax(-1), 'confidence': p.max(-1)}


def threshold_for_risk(confidence, correct, alpha):
    """Lowest threshold (largest coverage) whose accepted set has empirical risk <= alpha. Ties decide together."""
    conf, ok = np.asarray(confidence, float), np.asarray(correct, bool)
    order = np.argsort(-conf, kind='stable')
    errors = np.cumsum(~ok[order])
    sizes = np.arange(1, conf.size + 1)
    last_of_tie = np.append(conf[order][1:] != conf[order][:-1], True)
    feasible = last_of_tie & (errors <= alpha * sizes)
    return float(conf[order][np.flatnonzero(feasible)[-1]]) if feasible.any() else float('inf')


def apply_threshold(confidence, correct, threshold):
    conf, ok = np.asarray(confidence, float), np.asarray(correct, bool)
    keep = conf >= threshold
    accepted, errors = int(keep.sum()), int((~ok[keep]).sum())
    return {'n': int(conf.size), 'accepted': accepted, 'errors': errors, 'coverage': accepted / conf.size,
            'risk': errors / accepted if accepted else None}


def risk_upper_bound(errors, accepted, confidence=.95):
    """One-sided Clopper-Pearson upper bound. Clips are treated as independent, so read it as a rough guide."""
    if not accepted:
        return None
    return 1.0 if errors >= accepted else float(beta.ppf(confidence, errors + 1, accepted - errors))


def destination_errors(pred, truth, accepted, n_classes):
    """Per destination class: how many accepted clips were routed there, and how many of those were wrong."""
    pred, truth, accepted = np.asarray(pred), np.asarray(truth), np.asarray(accepted, bool)
    return {c: {'routed': int((accepted & (pred == c)).sum()), 'errors': int((accepted & (pred == c) & (truth != c)).sum())}
            for c in range(n_classes)}


def risk_coverage_curve(confidence, correct, points=21):
    """Risk at evenly spaced coverage levels, for plotting."""
    conf, ok = np.asarray(confidence, float), np.asarray(correct, bool)
    order = np.argsort(-conf, kind='stable')
    errors = np.cumsum(~ok[order])
    out = []
    for coverage in np.linspace(.05, 1., points):
        size = max(1, int(round(coverage * conf.size)))
        out.append({'coverage': size / conf.size, 'risk': float(errors[size - 1] / size)})
    return out
