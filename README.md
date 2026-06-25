# OpenCPI-Guard

**OpenCPI-Guard** is a reproducible machine-learning framework for regional inflation pressure detection from official Consumer Price Index (CPI) tables.

The framework is designed to be **country-portable**: each country only needs a data adapter that maps official CPI tables into the common CPI panel schema. The Indonesia reference implementation uses BPS CPI data for 150 CPI observation areas.

## Main idea

Given CPI information available up to month `t`, OpenCPI-Guard predicts whether a region will enter a high inflation-pressure group at month `t+1`, using relative within-country labels such as top-20% or top-10% inflation pressure.

## Repository layout

```text
opencpi_guard/
  adapters/
    generic_csv_adapter.py      # load CPI data from any country using common schema
    bps_indonesia_note.py       # notes for BPS-specific data acquisition
  schema.py                     # schema validation
  features.py                   # leakage-safe feature builder
  labels.py                     # top-k pressure labels
  split.py                      # temporal train/valid/test split
  metrics.py                    # PR-AUC, ROC-AUC, Recall@K, Precision@K
  models.py                     # baseline and ML model zoo
  pipeline.py                   # end-to-end runner helpers
scripts/
  run_opencpi_guard.py          # CLI runner
  validate_schema.py            # validate generic CSV input
config/
  default_config.yml
data/
  schema_template.csv           # template for other countries
  indonesia_reference_dataset.zip
notebooks/
  Colab GPU notebook
results/
  hardcore v2.1 output ZIP and summary
```

## Quick start: generic country CSV

Prepare a CSV with at least:

```text
country,region_id,region_name,month,cpi_all_items,inflation_mtm,inflation_yoy
```

Recommended optional columns:

```text
cpi_g01...cpi_g11
inflation_mtm_g01...inflation_mtm_g11
inflation_yoy_g01...inflation_yoy_g11
```

Run:

```bash
pip install -r requirements.txt
python scripts/validate_schema.py --input data/schema_template.csv
python scripts/run_opencpi_guard.py --input data/schema_template.csv --output outputs_demo --country DEMO
```

For the Indonesia reference implementation, use the Colab notebook in `notebooks/`.

## Paper positioning

Recommended title:

> OpenCPI-Guard: A Reproducible Machine Learning Framework for Regional Inflation Pressure Detection from Official CPI Tables

## Indonesia reference result summary

The Indonesia implementation shows strong performance for annual/y-o-y regional inflation pressure detection:

| Target | Test PR-AUC | Test ROC-AUC | Test Macro-F1 |
|---|---:|---:|---:|
| y-o-y top-20 next month | 0.695 | 0.893 | 0.820 |
| y-o-y top-10 next month | 0.636 | 0.933 | 0.790 |
| m-to-m top-20 next month | 0.340 | 0.637 | 0.566 |
| m-to-m top-10 next month | 0.193 | 0.687 | 0.561 |

## Important reproducibility rule

All feature generation uses information available up to month `t` only. The labels refer to month `t+1`, and model selection is based on validation PR-AUC, not test results.
"# opencpi-guard" 
