from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = [
    'country', 'region_id', 'region_name', 'month',
    'cpi_all_items', 'inflation_mtm', 'inflation_yoy'
]

OPTIONAL_PREFIXES = ['cpi_g', 'inflation_mtm_g', 'inflation_yoy_g']


def validate_schema(df: pd.DataFrame, strict: bool = False) -> dict:
    """Validate the common CPI panel schema.

    Parameters
    ----------
    df : pandas.DataFrame
        CPI panel in long region-month format.
    strict : bool
        If True, require all 11 CPI group columns for CPI, m-to-m, and y-o-y.

    Returns
    -------
    dict
        Validation report.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    report = {
        'is_valid': len(missing) == 0,
        'missing_required_columns': missing,
        'n_rows': int(len(df)),
        'n_regions': int(df['region_id'].nunique()) if 'region_id' in df.columns else None,
        'optional_group_columns_found': [c for c in df.columns if any(c.startswith(p) for p in OPTIONAL_PREFIXES)],
    }
    if strict:
        strict_required = []
        for prefix in OPTIONAL_PREFIXES:
            strict_required.extend([f'{prefix}{i:02d}' for i in range(1, 12)])
        strict_missing = [c for c in strict_required if c not in df.columns]
        report['strict_missing_group_columns'] = strict_missing
        report['is_valid'] = report['is_valid'] and len(strict_missing) == 0
    return report


def coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if 'month' in df.columns:
        df['month'] = pd.to_datetime(df['month'])
    for c in df.columns:
        if c not in ['country', 'region_id', 'region_name', 'month']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df
