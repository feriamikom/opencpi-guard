from __future__ import annotations

import pandas as pd
from opencpi_guard.schema import coerce_schema, validate_schema


def load_generic_csv(path: str, strict: bool = False) -> pd.DataFrame:
    """Load any country CPI panel formatted with the OpenCPI-Guard schema."""
    df = pd.read_csv(path)
    df = coerce_schema(df)
    report = validate_schema(df, strict=strict)
    if not report['is_valid']:
        raise ValueError(f'Invalid CPI schema: {report}')
    return df
