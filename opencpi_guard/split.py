from __future__ import annotations

import pandas as pd


def temporal_split(df: pd.DataFrame, valid_start: str, test_start: str):
    valid_start = pd.Timestamp(valid_start)
    test_start = pd.Timestamp(test_start)
    train = df[df['month'] < valid_start].copy()
    valid = df[(df['month'] >= valid_start) & (df['month'] < test_start)].copy()
    test = df[df['month'] >= test_start].copy()
    return train, valid, test
