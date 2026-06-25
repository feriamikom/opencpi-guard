from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score


def precision_recall_at_top_fraction(y_true, y_score, frac=0.10):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    n = len(y_true)
    k = max(1, int(np.ceil(frac * n)))
    idx = np.argsort(-y_score)[:k]
    selected_true = y_true[idx]
    precision_at_k = selected_true.mean() if k > 0 else np.nan
    positives = y_true.sum()
    recall_at_k = selected_true.sum() / positives if positives > 0 else np.nan
    return precision_at_k, recall_at_k


def binary_metrics(y_true, y_score, threshold=0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    out = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'prevalence': float(y_true.mean()),
    }
    if len(np.unique(y_true)) > 1:
        out['pr_auc'] = average_precision_score(y_true, y_score)
        out['roc_auc'] = roc_auc_score(y_true, y_score)
    else:
        out['pr_auc'] = np.nan
        out['roc_auc'] = np.nan
    p10, r10 = precision_recall_at_top_fraction(y_true, y_score, 0.10)
    p20, r20 = precision_recall_at_top_fraction(y_true, y_score, 0.20)
    out.update({
        'precision_at_top10pct': p10,
        'recall_at_top10pct': r10,
        'precision_at_top20pct': p20,
        'recall_at_top20pct': r20,
    })
    return out
