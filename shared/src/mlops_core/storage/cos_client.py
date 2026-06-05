"""IBM Cloud Object Storage client (S3-compatible API).

Provides a unified interface for reading/writing parquet, pickle, and raw bytes
to IBM COS. Used by both IBM Code Engine and IBM Watsonx pipelines.
"""

from __future__ import annotations

import io
import pickle
from datetime import datetime, timezone

import ibm_boto3
from ibm_botocore.client import Config
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)


class COSClient:
    """Wrapper around IBM Cloud Object Storage (S3-compatible API).

    Args:
        endpoint: COS endpoint URL.
        api_key: IBM Cloud API key.
        instance_crn: COS service instance CRN.
        bucket: Target bucket name.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        instance_crn: str,
        bucket: str,
    ):
        self.bucket = bucket
        self._client = ibm_boto3.client(
            "s3",
            ibm_api_key_id=api_key,
            ibm_service_instance_id=instance_crn,
            config=Config(signature_version="oauth"),
            endpoint_url=endpoint,
        )
        logger.info("cos.connected", endpoint=endpoint, bucket=bucket)

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

    # ─── Read / Write Bytes ──────────────────────────────────────────────

    def read_bytes(self, key: str) -> bytes:
        """Read raw bytes from COS."""
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def write_bytes(self, data: bytes, key: str) -> None:
        """Write raw bytes to COS."""
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)

    # ─── Utilities ───────────────────────────────────────────────────────

    def list_keys(self, prefix: str) -> list[str]:
        """List object keys under a prefix."""
        response = self._client.list_objects_v2(Bucket=self.bucket, Prefix=prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def get_latest_key(self, prefix: str, suffix: str = ".pkl") -> str | None:
        """Get the most recently modified key matching prefix and suffix."""
        keys = [k for k in self.list_keys(prefix) if k.endswith(suffix)]
        if not keys:
            return None
        # Keys are named with dates, so lexicographic sort works
        return sorted(keys)[-1]

    def generate_dated_key(self, prefix: str, suffix: str) -> str:
        """Generate a key with today's UTC date, e.g. models/2026_06_05.pkl"""
        date_str = datetime.now(timezone.utc).strftime("%Y_%m_%d")
        return f"{prefix}{date_str}{suffix}"

