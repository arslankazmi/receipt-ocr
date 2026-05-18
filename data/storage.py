from abc import ABC, abstractmethod
from typing import BinaryIO, Iterator
import os
from pathlib import Path
import functools

class StorageBackend(ABC):
    @abstractmethod
    def open(self, path: str) -> BinaryIO: ...
    @abstractmethod
    def list(self, prefix: str) -> Iterator[str]: ...
    @abstractmethod
    def exists(self, path: str) -> bool: ...

class LocalBackend(StorageBackend):
    def __init__(self, base: str = ""):
        self.base = Path(base) if base else Path(".")
    def open(self, path: str) -> BinaryIO:
        return open(self.base / path, "rb")
    def list(self, prefix: str) -> Iterator[str]:
        p = self.base / prefix
        if p.is_dir():
            for f in sorted(p.iterdir()):
                yield str(f.relative_to(self.base))
    def exists(self, path: str) -> bool:
        return (self.base / path).exists()

class S3Backend(StorageBackend):
    """S3 backend using boto3. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION."""
    def __init__(self, bucket: str, prefix: str = ""):
        import boto3
        self._s3 = boto3.client("s3")
        self.bucket = bucket
        self.prefix = prefix.rstrip("/")
        self._cache: dict[str, bytes] = {}  # LRU via functools.lru_cache not usable on methods; simple dict

    def _key(self, path: str) -> str:
        return f"{self.prefix}/{path}".lstrip("/") if self.prefix else path

    def open(self, path: str) -> BinaryIO:
        import io
        key = self._key(path)
        if key not in self._cache:
            obj = self._s3.get_object(Bucket=self.bucket, Key=key)
            self._cache[key] = obj["Body"].read()
            if len(self._cache) > 200:  # simple eviction
                oldest = next(iter(self._cache))
                del self._cache[oldest]
        return io.BytesIO(self._cache[key])

    def list(self, prefix: str) -> Iterator[str]:
        paginator = self._s3.get_paginator("list_objects_v2")
        full_prefix = self._key(prefix)
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full_prefix):
            for obj in page.get("Contents", []):
                yield obj["Key"]

    def exists(self, path: str) -> bool:
        import botocore.exceptions
        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._key(path))
            return True
        except botocore.exceptions.ClientError:
            return False

def get_backend(cfg: dict) -> StorageBackend:
    """Load backend from config dict (the 'storage' section of config.yaml)."""
    backend = cfg.get("backend", "local")
    if backend == "local":
        return LocalBackend(base=cfg.get("local_base", ""))
    elif backend == "s3":
        return S3Backend(bucket=cfg["bucket"], prefix=cfg.get("prefix", ""))
    raise ValueError(f"Unknown backend: {backend}")
