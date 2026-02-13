from train import (
    REQUIRED_COMPARE_COLUMNS,
    TRAINING_STATS_FIELDS,
    BASELINE_STATS_FIELDS,
)


def test_required_columns_in_training_and_baseline_schemas():
    required = set(REQUIRED_COMPARE_COLUMNS)
    train_cols = set(TRAINING_STATS_FIELDS)
    baseline_cols = set(BASELINE_STATS_FIELDS)
    assert required.issubset(train_cols), f"training schema missing: {sorted(required - train_cols)}"
    assert required.issubset(baseline_cols), f"baseline schema missing: {sorted(required - baseline_cols)}"
