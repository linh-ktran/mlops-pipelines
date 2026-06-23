"""Local filesystem storage — drop-in replacement for COSClient (testing)."""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import structlog

from src.pipeline.config import PipelineConfig

log = structlog.get_logger(__name__)

LOCAL_DIR = Path("local_storage")


class LocalStorageClient:
    """Same interface as COSClient but writes to ./local_storage/."""

    def __init__(self, config: PipelineConfig):
        self.bucket = config.cos_bucket
        self.base_dir = LOCAL_DIR / self.bucket
        self.base_dir.mkdir(parents=True, exist_ok=True)
        log.info("local storage ready", path=str(self.base_dir))

    def _path(self, key: str) -> Path:
        p = self.base_dir / key
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def read_parquet(self, key: str) -> pd.DataFrame:
        return pd.read_parquet(self._path(key))

    def write_parquet(self, df: pd.DataFrame, key: str):
        df.to_parquet(self._path(key), index=False, engine="pyarrow")

    def read_pickle(self, key: str):
        with open(self._path(key), "rb") as f:
            return pickle.load(f)

    def write_pickle(self, obj, key: str):
        with open(self._path(key), "wb") as f:
            pickle.dump(obj, f)

    def write_bytes(self, data: bytes, key: str):
        self._path(key).write_bytes(data)

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        prefix_path = self.base_dir / prefix
        if not prefix_path.exists():
            return []
        return sorted(str(p.relative_to(self.base_dir)) for p in prefix_path.rglob("*") if p.is_file())

    def get_latest_key(self, prefix: str, suffix: str = ".pkl") -> str | None:
        keys = [k for k in self.list_keys(prefix) if k.endswith(suffix)]
        return sorted(keys)[-1] if keys else None

    def generate_dated_key(self, prefix: str, suffix: str) -> str:
        date_str = datetime.now(timezone.utc).strftime("%Y_%m_%d")
        return f"{prefix}{date_str}{suffix}"
