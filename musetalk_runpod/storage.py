"""Cloudflare R2 helpers used by the RunPod handler.

Configured entirely via environment variables so the same module works for
local smoke tests and inside the container.
"""

from __future__ import annotations

import os
from pathlib import Path

import boto3
import requests
from botocore.config import Config


class R2ConfigError(RuntimeError):
    pass


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise R2ConfigError(
            f"Environment variable {name} is required but not set. "
            f"Configure R2_ENDPOINT, R2_BUCKET, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY."
        )
    return value


def _client():
    return boto3.client(
        "s3",
        endpoint_url=_env("R2_ENDPOINT"),
        aws_access_key_id=_env("R2_ACCESS_KEY_ID"),
        aws_secret_access_key=_env("R2_SECRET_ACCESS_KEY"),
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def download_to_local(url_or_key: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if url_or_key.startswith(("http://", "https://")):
        with requests.get(url_or_key, stream=True, timeout=600) as r:
            r.raise_for_status()
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    else:
        _client().download_file(_env("R2_BUCKET"), url_or_key, str(dest))
    return dest


def upload_from_local(src: Path, key: str, presign_expires: int = 3600) -> str:
    s3 = _client()
    s3.upload_file(str(src), _env("R2_BUCKET"), key)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": _env("R2_BUCKET"), "Key": key},
        ExpiresIn=presign_expires,
    )
