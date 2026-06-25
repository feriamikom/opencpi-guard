from __future__ import annotations

import pandas as pd


def add_top_pressure_labels(
    df: pd.DataFrame,
    value_col: str,
    top_frac: float = 0.20,
    horizon: int = 1,
    label_name: str | None = None,
    group_col: str = 'month',
    region_col: str = 'region_id',
) -> pd.DataFrame:
    """Create next-period top-k inflation pressure labels.

    Label for row (region i, month t) is 1 if region i is in the top fraction of
    `value_col` at month t+horizon, computed cross-sectionally within each month.
    """
    out = df.sort_values([region_col, group_col]).copy()
    next_col = f'{value_col}_next{horizon}'
    out[next_col] = out.groupby(region_col)[value_col].shift(-horizon)

    rank_col = f'{next_col}_rank_pct'
    out[rank_col] = out.groupby(group_col)[next_col].rank(pct=True, method='average')

    if label_name is None:
        pct = int(round(top_frac * 100))
        label_name = f'target_high_{value_col}_top{pct}_next'

    out[label_name] = (out[rank_col] >= (1.0 - top_frac)).astype('Int64')
    return out


def add_default_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = add_top_pressure_labels(out, 'inflation_mtm', 0.20, label_name='target_high_mtm_top20_next')
    out = add_top_pressure_labels(out, 'inflation_yoy', 0.20, label_name='target_high_yoy_top20_next')
    out = add_top_pressure_labels(out, 'inflation_mtm', 0.10, label_name='target_high_mtm_top10_next')
    out = add_top_pressure_labels(out, 'inflation_yoy', 0.10, label_name='target_high_yoy_top10_next')
    return out
