"""IBM Cloud Object Storage client — replaces boto3/S3 in the AWS pipeline."""

from __future__ import annotations

import io
import pickle
from datetime import datetime, timezone

import ibm_boto3
from ibm_botocore.client import Config
import pandas as pd
import structlog

from src.orchestrator.config import PipelineConfig

logger = structlog.get_logger(__name__)


class COSClient:
    """Wrapper around IBM Cloud Object Storage (S3-compatible API)."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.bucket = config.cos_bucket
        self._client = ibm_boto3.client(
            "s3",
            ibm_api_key_id=config.cos_api_key,
            ibm_service_instance_id=config.cos_instance_crn,
            config=Config(signature_version="oauth"),
            endpoint_url=config.cos_endpoint,
        )
        logger.info("cos.connected", endpoint=config.cos_endpoint, bucket=self.bucket)

    # ─── Read / Write Parquet ────────────────────────────────────────────

    def read_parquet(self, key: str) -> pd.DataFrame:
        """Read a parquet file from COS into a DataFrame."""
        logger.info("cos.read_parquet", key=key)
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        data = response["Body"].read()
        return pd.read_parquet(io.BytesIO(data))

    def write_parquet(self, df: pd.DataFrame, key: str) -> None:
        """Write a DataFrame to COS as parquet."""
        logger.info("cos.write_parquet", key=key, rows=len(df))
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=False, engine="pyarrow")
        buffer.seek(0)
        self._client.put_object(Bucket=self.bucket, Key=key, Body=buffer.getvalue())

    # ─── Read / Write Pickle (models) ───────────────────────────────────

    def read_pickle(self, key: str):
        """Load a pickled object from COS."""
        logger.info("cos.read_pickle", key=key)
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        return pickle.loads(response["Body"].read())

    def write_pickle(self, obj, key: str) -> None:
        """Write a pickled object to COS."""
        logger.info("cos.write_pickle", key=key)
        data = pickle.dumps(obj)
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

    # ─── Utilities ───────────────────────────────────────────────────────

    def list_keys(self, prefix: str) -> list[str]:
        """List object keys under a prefix."""
        response = self._client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        if "Contents" not in response:
            return []
        return [obj["Key"] for obj in response["Contents"]]

    def get_latest_key(self, prefix: str, suffix: str = ".pkl") -> str | None:
        """Get the most recently modified key matching prefix and suffix."""
        keys = [k for k in self.list_keys(prefix) if k.endswith(suffix)]
        if not keys:
            return None
        # Keys are named with dates, so lexicographic sort works
        return sorted(keys)[-1]

    def generate_dated_key(self, prefix: str, suffix: str) -> str:
        """Generate a key with today's date, e.g. models/model_2026_06_01.pkl"""
        date_str = datetime.now(timezone.utc).strftime("%Y_%m_%d")
        return f"{prefix}{date_str}{suffix}"

