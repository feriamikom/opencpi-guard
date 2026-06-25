from __future__ import annotations

import pandas as pd
from opencpi_guard.schema import coerce_schema, validate_schema


def load_bps_final_wide(path: str, include_national: bool = False) -> pd.DataFrame:
    """Convert the prepared BPS wide panel into the OpenCPI common schema.

    This adapter expects the cleaned dataset generated in this project, with
    columns such as `region_key`, `region_name`, `date`, `cpi_all_items`,
    `inflation_mtm`, and `inflation_yoy`.
    """
    df = pd.read_csv(path)
    if not include_national and 'is_national' in df.columns:
        df = df[df['is_national'] == False].copy()
    out = df.copy()
    out['country'] = 'Indonesia'
    out['region_id'] = out['region_key'].astype(str)
    out['month'] = pd.to_datetime(out['date'])
    keep = ['country', 'region_id', 'region_name', 'month', 'cpi_all_items', 'inflation_mtm', 'inflation_yoy']
    keep += [c for c in out.columns if c.startswith('cpi_g') or c.startswith('inflation_mtm_g') or c.startswith('inflation_yoy_g')]
    out = out[keep]
    out = coerce_schema(out)
    report = validate_schema(out, strict=False)
    if not report['is_valid']:
        raise ValueError(f'Converted BPS dataset does not satisfy schema: {report}')
    return out
