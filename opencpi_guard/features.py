from __future__ import annotations

import pandas as pd
import numpy as np

ID_COLS = ['country', 'region_id', 'region_name', 'month']
TARGET_PREFIX = 'target_'


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        c for c in df.columns
        if c not in ID_COLS and not c.startswith(TARGET_PREFIX)
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def add_lag_features(df: pd.DataFrame, cols: list[str], lags=(1, 2, 3), region_col='region_id') -> pd.DataFrame:
    out = df.sort_values([region_col, 'month']).copy()
    for col in cols:
        for lag in lags:
            out[f'{col}_lag{lag}'] = out.groupby(region_col)[col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, cols: list[str], windows=(3, 6), region_col='region_id') -> pd.DataFrame:
    out = df.sort_values([region_col, 'month']).copy()
    g = out.groupby(region_col, group_keys=False)
    for col in cols:
        shifted = g[col].shift(1)
        for w in windows:
            out[f'{col}_roll{w}_mean'] = shifted.groupby(out[region_col]).rolling(w, min_periods=2).mean().reset_index(level=0, drop=True)
            out[f'{col}_roll{w}_std'] = shifted.groupby(out[region_col]).rolling(w, min_periods=2).std().reset_index(level=0, drop=True)
    return out


def add_cross_sectional_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f'{col}_cs_rank_pct'] = out.groupby('month')[col].rank(pct=True, method='average')
        mean = out.groupby('month')[col].transform('mean')
        std = out.groupby('month')[col].transform('std').replace(0, np.nan)
        out[f'{col}_cs_z'] = (out[col] - mean) / std
    return out


def build_leakage_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build generic leakage-safe features.

    Note: current-month features are allowed because the task predicts t+1.
    Rolling/lag features use prior months only.
    """
    out = df.copy()
    base_cols = numeric_feature_columns(out)
    core_cols = [c for c in base_cols if not c.endswith('_next1') and '_rank_pct' not in c]
    out = add_lag_features(out, core_cols, lags=(1, 2, 3))
    out = add_rolling_features(out, core_cols, windows=(3, 6))
    out = add_cross_sectional_features(out, core_cols)

    if 'inflation_yoy' in out.columns:
        out['current_inflation_yoy_top20'] = (out.groupby('month')['inflation_yoy'].rank(pct=True) >= 0.80).astype(int)
        out['current_inflation_yoy_top10'] = (out.groupby('month')['inflation_yoy'].rank(pct=True) >= 0.90).astype(int)
    if 'inflation_mtm' in out.columns:
        out['current_inflation_mtm_top20'] = (out.groupby('month')['inflation_mtm'].rank(pct=True) >= 0.80).astype(int)
        out['current_inflation_mtm_top10'] = (out.groupby('month')['inflation_mtm'].rank(pct=True) >= 0.90).astype(int)
    return out
