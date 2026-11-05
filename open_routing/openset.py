"""Open-set scores, thresholds and the pre-written usability rule. A higher score means more like the five classes."""
import numpy as np
from scipy.special import logsumexp
from scipy.stats import rankdata

KNN_K = 5
PCA_DIMS = 64
RIDGE = 1e-6
A4C, A5C = 2, 3
MIN_AUROC = .90
MAX_2CH_ACCEPT = .20
MIN_4CH_RETAIN = .70
MIN_EXTERNAL_RETAIN = .80


def auroc(inside, outside):
    """Chance that a five-class recording scores above an out-of-set one; ties count one half."""
    inside, outside = np.asarray(inside, float), np.asarray(outside, float)
    ranks = rankdata(np.concatenate([inside, outside]))
    n, m = len(inside), len(outside)
    return float((ranks[:n].sum() - n * (n + 1) / 2) / (n * m))


def msp(prob):
    return np.asarray(prob, float).max(1)


def merged_msp(prob):
    """The largest probability after adding A4C and A5C."""
    p = np.asarray(prob, float)
    merged = np.delete(p, A5C, axis=1)
    merged[:, A4C] += p[:, A5C]
    return merged.max(1)


def energy_score(window_logits):
    """Mean over windows of logsumexp of the logits, for one recording."""
    return float(logsumexp(np.asarray(window_logits, float), axis=1).mean())


def _unit(x):
    x = np.asarray(x, float)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def knn_score(reference, query, k=KNN_K):
    """Negative mean distance to the k nearest reference recordings, on unit-length embeddings."""
    if k > len(reference):
        raise ValueError('k exceeds the reference size')
    reference, query = _unit(reference), _unit(query)
    distance = np.sqrt(np.maximum(2 - 2 * query @ reference.T, 0.))
    return -np.sort(distance, axis=1)[:, :k].mean(1)


def fit_mahalanobis(features, labels, dims=PCA_DIMS):
    """PCA on the reference, then class means and one shared covariance."""
    x, y = np.asarray(features, float), np.asarray(labels)
    centre = x.mean(0)
    _, _, vt = np.linalg.svd(x - centre, full_matrices=False)
    components = vt[:dims]
    z = (x - centre) @ components.T
    classes = np.unique(y)
    means = np.stack([z[y == c].mean(0) for c in classes])
    within = z - means[np.searchsorted(classes, y)]
    covariance = within.T @ within / len(z) + RIDGE * np.eye(z.shape[1])
    return {'centre': centre, 'components': components, 'means': means, 'precision': np.linalg.inv(covariance)}


def mahalanobis_score(model, query):
    """Negative squared distance to the nearest class mean."""
    z = (np.asarray(query, float) - model['centre']) @ model['components'].T
    gaps = z[:, None, :] - model['means'][None, :, :]
    return -np.einsum('nci,ij,ncj->nc', gaps, model['precision'], gaps).min(1)


def threshold_for_retention(scores, retain):
    """The largest threshold that keeps at least this share of the given recordings."""
    if not 0 < retain <= 1:
        raise ValueError('retention must be in (0, 1]')
    ordered = np.sort(np.asarray(scores, float))
    keep = int(np.ceil(retain * len(ordered) - 1e-12))
    return float(ordered[len(ordered) - keep])


def accept_share(scores, threshold):
    return float((np.asarray(scores, float) >= threshold).mean())


def paired_auroc_ci(inside, outside, n_boot=5000, seed=2026110402):
    """Percentile interval of AUROC when one patient contributes one inside and one outside recording."""
    inside, outside = np.asarray(inside, float), np.asarray(outside, float)
    if len(inside) != len(outside):
        raise ValueError('one inside and one outside recording per patient required')
    rng = np.random.default_rng(seed)
    n = len(inside)
    draws = [auroc(inside[i], outside[i]) for i in rng.integers(0, n, (n_boot, n))]
    return np.quantile(draws, [.025, .975]).tolist()


def usable(seed_rows):
    """The pre-written rule: every condition must hold in every seed."""
    return all(r['auroc'] >= MIN_AUROC and r['camus_2ch_accept'] <= MAX_2CH_ACCEPT and r['camus_4ch_retain'] >= MIN_4CH_RETAIN
               and r['dynamic_retain'] >= MIN_EXTERNAL_RETAIN and r['lvh_retain'] >= MIN_EXTERNAL_RETAIN for r in seed_rows)
