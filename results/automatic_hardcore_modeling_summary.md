# BPS CPI Inflation Pressure Detection — Hardcore v2.1 Summary

## Split

```json
{
  "train_rows": 2550,
  "valid_rows": 900,
  "test_rows": 750,
  "train_date_min": "2024-01-01",
  "train_date_max": "2025-05-01",
  "valid_date_min": "2025-06-01",
  "valid_date_max": "2025-11-01",
  "test_date_min": "2025-12-01",
  "test_date_max": "2026-04-01"
}
```

## Feature audit

- Numeric features: 428

- Categorical features: ['region_name', 'month_str']

- Total features: 430

- Leakage audit: passed. No target/next/y_next columns used as features.

## Best validation-selected models

| target                     | selected_by       | top_models                                                                    | final_model                       |   validation_threshold |   validation_pr_auc |   test_pr_auc |   test_roc_auc |   test_macro_f1 |   test_recall_at_top20pct |   test_recall_at_top10pct |
|:---------------------------|:------------------|:------------------------------------------------------------------------------|:----------------------------------|-----------------------:|--------------------:|--------------:|---------------:|----------------:|--------------------------:|--------------------------:|
| target_high_mtm_top20_next | validation_pr_auc | random_forest_balanced; xgboost_gpu_or_cpu_rank; extra_trees_balanced         | ensemble_rankavg_top3_by_valid_pr |               0.842748 |            0.31543  |      0.340038 |       0.636604 |        0.566298 |                  0.343949 |                  0.203822 |
| target_high_yoy_top20_next | validation_pr_auc | extra_trees_balanced; random_forest_balanced; xgboost_gpu_or_cpu_rank         | ensemble_rankavg_top3_by_valid_pr |               0.822659 |            0.811522 |      0.694917 |       0.892658 |        0.819513 |                  0.701987 |                  0.384106 |
| target_high_mtm_top10_next | validation_pr_auc | catboost_gpu_or_cpu_native_cat; xgboost_gpu_or_cpu_rank; extra_trees_balanced | ensemble_rankavg_top3_by_valid_pr |               0.816667 |            0.173644 |      0.193084 |       0.686622 |        0.560861 |                  0.384615 |                  0.205128 |
| target_high_yoy_top10_next | validation_pr_auc | xgboost_gpu_or_cpu_rank; lightgbm_balanced; extra_trees_balanced              | ensemble_rankavg_top3_by_valid_pr |               0.877496 |            0.681378 |      0.636317 |       0.932836 |        0.789943 |                  0.87013  |                  0.584416 |


## Interpretation note

Model selection is based on validation PR-AUC, then selected models are refit on train+validation and evaluated once on the temporal test set. This avoids selecting the model directly from the test set.

Rank-average ensemble is used to stabilize risk ranking across several high-performing models.
