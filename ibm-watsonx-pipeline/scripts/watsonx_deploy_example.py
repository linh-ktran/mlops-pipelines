"""
Example: Register and deploy a model to watsonx.ai.

This script demonstrates the full flow:
    1. Train a model locally
    2. Store it in the watsonx.ai model registry
    3. Deploy as an online endpoint
    4. Score with sample data
    5. Clean up

Usage:
    uv run python scripts/watsonx_deploy_example.py
"""

from src.pipeline.config import PipelineConfig
from src.storage.watsonx_client import WatsonxClient
from src.training.trainer import train_model, TrainedModel
from scripts.generate_sample_data import generate_sample_data
from src.training.feature_engineering import generate_features

import numpy as np


def main():
    print("=" * 60)
    print("  watsonx.ai Model Deployment Example")
    print("=" * 60)
    print()

    # 1. Generate sample data and train
    print("→ Step 1: Generate features and train model")
    config = PipelineConfig.from_env()
    raw_df = generate_sample_data(days=60)
    features_df = generate_features(raw_df, config)
    trained = train_model(features_df, config)
    print(f"  Model trained. MAE={trained.metrics['mae']:.3f}, R²={trained.metrics['r2']:.3f}")
    print()

    # 2. Connect to watsonx.ai
    print("→ Step 2: Connect to watsonx.ai")
    watsonx = WatsonxClient(config)
    print("  Connected!")
    print()

    # 3. Store model
    print("→ Step 3: Store model in watsonx.ai registry")
    model_id = watsonx.store_model(trained, config)
    print(f"  Model ID: {model_id}")
    print()

    # 4. Deploy
    print("→ Step 4: Deploy model as online endpoint")
    deployment_id = watsonx.deploy_model(model_id, config)
    print(f"  Deployment ID: {deployment_id}")
    print()

    # 5. Score
    print("→ Step 5: Score with sample data")
    sample_features = features_df[trained.feature_names].head(5).values.tolist()
    payload = {
        "input_data": [{
            "fields": trained.feature_names,
            "values": sample_features,
        }]
    }
    result = watsonx.score(deployment_id, payload)
    print(f"  Predictions: {result}")
    print()

    # 6. Cleanup
    print("→ Step 6: Cleanup")
    watsonx.delete_deployment(deployment_id)
    watsonx.delete_model(model_id)
    print("  Cleaned up deployment and model.")
    print()

    print("✓ Done!")


if __name__ == "__main__":
    main()

