# mlops-core

The shared bits that all three pipelines use. Extracted this so I wouldn't copy-paste feature engineering and training logic everywhere.

## What's in it

| Module | Does what |
|--------|-----------|
| `mlops_core.features` | Time encoding, holidays, lags, rolling stats |
| `mlops_core.training` | XGBoost training with chronological split and bias correction |
| `mlops_core.storage` | IBM COS client |
| `mlops_core.inference` | Prediction with bias correction applied |

## Installing it

From any of the sibling project directories:

```bash
uv add --editable ../shared

# or with COS support
uv add --editable "../shared[cos]"
```

## Using it

```python
from mlops_core.features import (
    add_cyclical_datetime_features,
    add_holiday_weekend_features,
    add_lag_features,
    add_rolling_statistics,
)
from mlops_core.training import train_model, TrainedModel, evaluate_model
from mlops_core.storage import COSClient
from mlops_core.inference import predict_with_model
```
