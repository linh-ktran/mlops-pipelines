# mlops-core

Shared ML utilities for energy price forecasting pipelines.

This package extracts common functionality used across all three pipeline implementations:
- **AWS SageMaker Pipeline**
- **IBM Code Engine Pipeline**
- **IBM Watsonx Pipeline**

## Shared Modules

| Module | Description |
|--------|-------------|
| `mlops_core.features` | Cyclical datetime, holiday, lag, and rolling statistics features |
| `mlops_core.training` | XGBoost training with chronological split, bias correction, and evaluation |
| `mlops_core.storage` | IBM Cloud Object Storage client (COS) |
| `mlops_core.inference` | Base prediction with bias correction |

## Installation

From a sibling project directory:

```bash
uv add --editable ../shared
```

Or with COS storage support:

```bash
uv add --editable "../shared[cos]"
```

## Usage

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

