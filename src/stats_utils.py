import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from sklearn.metrics import confusion_matrix

def bootstrap_auc_ci(y_true, y_score, n_boot=2000, seed=42):
    """
    Compute a bootstrap 95% confidence interval for the AUC.

    Returns
    ci : tuple (lower, upper) — 2.5th and 97.5th percentiles
    mean_auc : mean AUC across bootstrap resamples
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue  # skip resamples without both classes present
        aucs.append(roc_auc_score(y_true[idx], y_score[idx]))

    ci = np.percentile(aucs, [2.5, 97.5])
    mean_auc = np.mean(aucs)
    return ci, mean_auc


def _compute_midrank(x):
    """Auxiliary function for delong_test: computes midranks handling ties."""
    J = np.argsort(x)
    Z = x[J]
    n = len(x)
    T = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(n)
    T2[J] = T
    return T2


def delong_test(y_true, score_a, score_b):
    """
    DeLong test comparing two correlated AUCs (same samples, two models).

    Returns:
    z : z-statistic
    p_value : one-tailed p-value
    auc_a, auc_b : point estimates of each AUC
    """
    y_true = np.asarray(y_true)
    score_a = np.asarray(score_a)
    score_b = np.asarray(score_b)

    pos = y_true == 1
    neg = y_true == 0
    m = pos.sum()
    n = neg.sum()

    scores = np.vstack([score_a, score_b])
    aucs = []
    v10 = []
    v01 = []

    for s in scores:
        tx = _compute_midrank(s[pos])
        ty = _compute_midrank(s[neg])
        tz = _compute_midrank(s)

        aucs.append((tz[pos].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[pos] - tx) / n)
        v01.append(1.0 - (tz[neg] - ty) / m)

    v10 = np.array(v10)
    v01 = np.array(v01)

    s10 = np.cov(v10)
    s01 = np.cov(v01)
    var = s10 / m + s01 / n

    auc_diff = aucs[1] - aucs[0]
    var_diff = var[0, 0] + var[1, 1] - 2 * var[0, 1]
    z = auc_diff / np.sqrt(var_diff)

    p_value = 1 - norm.cdf(z)

    return z, p_value, aucs[0], aucs[1]


from sklearn.metrics import cohen_kappa_score


def bootstrap_kappa_diff(y_true, pred_a, pred_b, n_boot=2000, seed=42):
    """
    Paired bootstrap for the difference in Cohen's Kappa
    (pred_b - pred_a).

    Returns:
        diffs: bootstrap differences
        ci: 95% percentile confidence interval
        n_valid: number of valid bootstrap resamples
    """
    rng = np.random.default_rng(seed)

    n = len(y_true)
    y_true = np.asarray(y_true)
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)

    diffs = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, n)

        k_a = cohen_kappa_score(y_true[idx], pred_a[idx])
        k_b = cohen_kappa_score(y_true[idx], pred_b[idx])

        diff = k_b - k_a

        if np.isfinite(diff):
            diffs.append(diff)

    diffs = np.asarray(diffs)

    if len(diffs) == 0:
        raise ValueError("No valid bootstrap resamples were obtained.")

    ci = np.percentile(diffs, [2.5, 97.5])

    return diffs, ci, len(diffs)

def classification_metrics_by_class(y_true, y_pred, labels):
    """
    Compute one-vs-rest classification metrics for each class.

    Returns a DataFrame with sensitivity, specificity,
    precision, F1-score, and support.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()

    rows = []

    for i, label in enumerate(labels):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp

        support = tp + fn

        sensitivity = tp / support if support > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan

        f1_denominator = 2 * tp + fp + fn
        f1 = (
            2 * tp / f1_denominator
            if f1_denominator > 0 else np.nan
        )

        rows.append({
            'Category': label,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'Precision': precision,
            'F1-score': f1,
            'Support': support
        })

    return pd.DataFrame(rows)