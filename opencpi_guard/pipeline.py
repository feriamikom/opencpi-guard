from __future__ import annotations

import os
import pandas as pd
from joblib import dump
from tqdm.auto import tqdm

from opencpi_guard.adapters.generic_csv_adapter import load_generic_csv
from opencpi_guard.labels import add_default_labels
from opencpi_guard.features import build_leakage_safe_features
from opencpi_guard.split import temporal_split
from opencpi_guard.metrics import binary_metrics
from opencpi_guard.models import make_model_zoo, build_pipeline

TARGETS = [
    'target_high_mtm_top20_next',
    'target_high_yoy_top20_next',
    'target_high_mtm_top10_next',
    'target_high_yoy_top10_next',
]


def run_pipeline(input_csv: str, output_dir: str, valid_start='2025-06-01', test_start='2025-12-01', country=None):
    os.makedirs(output_dir, exist_ok=True)
    df = load_generic_csv(input_csv)
    if country is not None:
        df = df[df['country'].astype(str).str.lower() == str(country).lower()].copy()

    if not any(c.startswith('target_') for c in df.columns):
        df = add_default_labels(df)
    feat = build_leakage_safe_features(df)

    # Drop rows with unavailable labels
    target_cols = [t for t in TARGETS if t in feat.columns]
    feat = feat.dropna(subset=target_cols, how='all').copy()

    train, valid, test = temporal_split(feat, valid_start, test_start)
    id_cols = ['country', 'region_id', 'region_name', 'month']
    drop_cols = id_cols + [c for c in feat.columns if c.startswith('target_') or c.endswith('_next1')]
    feature_cols = [c for c in feat.columns if c not in drop_cols]
    categorical_cols = [c for c in feature_cols if feat[c].dtype == 'object' or c in ['region_id', 'region_name']]
    numeric_cols = [c for c in feature_cols if c not in categorical_cols]

    rows = []
    models = make_model_zoo()
    for target in tqdm(target_cols, desc='Targets'):
        tr = train.dropna(subset=[target])
        va = valid.dropna(subset=[target])
        te = test.dropna(subset=[target])
        if len(tr) == 0 or len(va) == 0 or len(te) == 0:
            continue
        X_train, y_train = tr[feature_cols], tr[target].astype(int)
        X_valid, y_valid = va[feature_cols], va[target].astype(int)
        X_test, y_test = te[feature_cols], te[target].astype(int)
        best_name, best_score, best_pipe = None, -1, None
        for name, model in tqdm(models.items(), desc=f'Models {target}', leave=False):
            pipe = build_pipeline(model, numeric_cols, categorical_cols)
            pipe.fit(X_train, y_train)
            if hasattr(pipe, 'predict_proba'):
                va_score = pipe.predict_proba(X_valid)[:, 1]
            else:
                va_score = pipe.decision_function(X_valid)
            m_val = binary_metrics(y_valid, va_score)
            if m_val.get('pr_auc', -1) > best_score:
                best_name, best_score, best_pipe = name, m_val.get('pr_auc', -1), pipe
        if hasattr(best_pipe, 'predict_proba'):
            test_score = best_pipe.predict_proba(X_test)[:, 1]
        else:
            test_score = best_pipe.decision_function(X_test)
        m_test = binary_metrics(y_test, test_score)
        row = {'target': target, 'best_model_by_validation': best_name, 'valid_pr_auc': best_score, **{f'test_{k}': v for k, v in m_test.items()}}
        rows.append(row)
        dump(best_pipe, os.path.join(output_dir, f'best_model_{target}_{best_name}.joblib'))

        pred = te[id_cols].copy()
        pred[target] = y_test.values
        pred['risk_score'] = test_score
        pred.to_csv(os.path.join(output_dir, f'predictions_{target}.csv'), index=False)

    metrics = pd.DataFrame(rows)
    metrics.to_csv(os.path.join(output_dir, 'model_metrics.csv'), index=False)
    return metrics
