# OpenCPI-Guard Reproducibility Protocol

## Step 1 — Prepare official CPI tables
Collect official CPI data from a national statistical office, IMF CPI, Eurostat HICP, or other official sources.

## Step 2 — Map into common CPI schema
At minimum, provide `country`, `region_id`, `region_name`, `month`, `cpi_all_items`, `inflation_mtm`, and `inflation_yoy`.

## Step 3 — Validate schema

```bash
python scripts/validate_schema.py --input your_country_cpi.csv
```

## Step 4 — Generate leakage-safe features
OpenCPI-Guard creates lags, rolling means/standard deviations, cross-sectional ranks, and current-month pressure indicators using only information available up to month t.

## Step 5 — Create relative pressure labels
Labels are based on within-country regional ranking at month t+1, such as top-20% or top-10% inflation pressure.

## Step 6 — Temporal validation
Use chronological train/validation/test split. Select models using validation PR-AUC only.

## Step 7 — Report final test metrics
Report PR-AUC, ROC-AUC, Macro-F1, Precision@TopK, and Recall@TopK.

## Step 8 — Export risk ranking and model artifacts
The output directory includes metrics, predictions, risk scores, and trained models.
