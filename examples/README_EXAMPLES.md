# Examples

## Validate generic schema

```bash
python scripts/validate_schema.py --input data/schema_template.csv
```

## Convert Indonesia reference wide panel to common schema

```bash
python scripts/prepare_indonesia_reference.py \
  --input data/indonesia_reference_unzipped/bps_cpi_wide_region_month_2024_2026.csv \
  --output data/indonesia_common_schema.csv
```

## Run generic pipeline

For full paper-grade results, use the Colab GPU notebook in `notebooks/colab_gpu_local_first_package.zip`.

For a quick CLI demonstration:

```bash
python scripts/run_opencpi_guard.py \
  --input data/indonesia_common_schema.csv \
  --output outputs_indonesia_cli \
  --valid-start 2025-06-01 \
  --test-start 2025-12-01
```

The CLI is intentionally simpler than the hardcore Colab notebook. The paper-grade output is in `results/BPS_CPI_hardcore_v21_outputs.zip`.
