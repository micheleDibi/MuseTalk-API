"""HTTP client for the MuseTalk lipsync service deployed on RunPod Serverless.

Handles:
- Cloudflare R2 (S3-compatible) uploads/downloads of input video + audio
  and the resulting MP4.
- POST /v2/{endpoint_id}/run to submit a job.
- GET /v2/{endpoint_id}/status/{request_id} polling with backoff until
  the job reaches a terminal state.

All configuration is read from environment variables. ``ensure_env()`` fails
fast with a single explicit error listing what is missing.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import boto3
import requests
from botocore.config import Config


REQUIRED_ENV_VARS = (
    "RUNPOD_API_KEY",
    "RUNPOD_ENDPOINT_ID",
    "R2_ENDPOINT",
    "R2_BUCKET",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
)

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"}

_DOTENV_FILES = ("runpod.env", ".env")


class RunPodClientError(RuntimeError):
    pass


def _load_dotenv_files() -> None:
    """Auto-load ``KEY=VALUE`` pairs from ``runpod.env`` / ``.env`` if present.

    Searches the current working directory and up to 3 parent levels, so the
    script works whether you run it from the repo root or a subfolder.

    Variables already defined in ``os.environ`` are NOT overwritten — the
    shell environment always wins.
    """
    here = Path.cwd().resolve()
    candidates: list[Path] = []
    for parent in [here, *list(here.parents)[:3]]:
        for fname in _DOTENV_FILES:
            candidates.append(parent / fname)

    seen: set[Path] = set()
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            if line.lower().startswith("export "):
                line = line[7:].lstrip()
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv_files()


def ensure_env() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.environ.get(name)]
    if missing:
        raise RunPodClientError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ".\nSet them before running the script. Example (PowerShell):\n"
            + "  $env:RUNPOD_API_KEY = '...'\n"
            + "  $env:RUNPOD_ENDPOINT_ID = '...'\n"
            + "  $env:R2_ENDPOINT = 'https://<account>.r2.cloudflarestorage.com'\n"
            + "  $env:R2_BUCKET = 'musetalk-io'\n"
            + "  $env:R2_ACCESS_KEY_ID = '...'\n"
            + "  $env:R2_SECRET_ACCESS_KEY = '...'\n"
        )


def _s3():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["R2_ENDPOINT"],
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
    )


def r2_upload(local_path: Path, key: str, presign_expires: int = 7200) -> str:
    if not local_path.is_file():
        raise RunPodClientError(f"r2_upload: file not found: {local_path}")
    s3 = _s3()
    bucket = os.environ["R2_BUCKET"]
    s3.upload_file(str(local_path), bucket, key)
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key},
        ExpiresIn=presign_expires,
    )


def r2_download(url_or_key: str, dest: Path, timeout: float = 600.0) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if url_or_key.startswith(("http://", "https://")):
        with requests.get(url_or_key, stream=True, timeout=timeout) as r:
            if r.status_code != 200:
                raise RunPodClientError(
                    f"r2_download: GET {url_or_key} -> HTTP {r.status_code}: {r.text[:200]}"
                )
            with dest.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    else:
        _s3().download_file(os.environ["R2_BUCKET"], url_or_key, str(dest))
    return dest


def r2_delete(keys: list[str]) -> None:
    if not keys:
        return
    _s3().delete_objects(
        Bucket=os.environ["R2_BUCKET"],
        Delete={"Objects": [{"Key": k} for k in keys]},
    )


def r2_presigned_url(key: str, expires_in: int = 7200) -> str:
    """Generate a fresh presigned GET URL for an existing R2 key."""
    s3 = _s3()
    return s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": os.environ["R2_BUCKET"], "Key": key},
        ExpiresIn=expires_in,
    )


def r2_exists(key: str) -> bool:
    """True iff the object at ``key`` exists in the R2 bucket."""
    s3 = _s3()
    try:
        s3.head_object(Bucket=os.environ["R2_BUCKET"], Key=key)
        return True
    except s3.exceptions.ClientError:
        return False
    except Exception:
        return False


def _api_base() -> str:
    return f"https://api.runpod.ai/v2/{os.environ['RUNPOD_ENDPOINT_ID']}"


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
        "Content-Type": "application/json",
    }


def submit_job(input_payload: dict, execution_timeout_ms: int = 3_600_000) -> str:
    body = {
        "input": input_payload,
        "policy": {"executionTimeout": execution_timeout_ms},
    }
    r = requests.post(f"{_api_base()}/run", headers=_headers(), json=body, timeout=30)
    if r.status_code != 200:
        raise RunPodClientError(
            f"submit_job: POST /run -> HTTP {r.status_code}: {r.text[:500]}"
        )
    data = r.json()
    job_id = data.get("id")
    if not job_id:
        raise RunPodClientError(f"submit_job: missing 'id' in response: {data}")
    return job_id


def _next_delay(prev: float) -> float:
    if prev < 5:
        return 5.0
    if prev < 15:
        return 15.0
    return 30.0


def poll_job(
    job_id: str,
    timeout_seconds: float | None = None,
    progress: bool = True,
    heartbeat_seconds: float = 60.0,
) -> dict:
    url = f"{_api_base()}/status/{job_id}"
    start = time.perf_counter()
    deadline = start + timeout_seconds if timeout_seconds else None
    delay = 3.0
    last_status = None
    last_print = start

    while True:
        r = requests.get(url, headers=_headers(), timeout=30)
        if r.status_code != 200:
            raise RunPodClientError(
                f"poll_job: GET /status/{job_id} -> HTTP {r.status_code}: {r.text[:500]}"
            )
        data = r.json()
        status = data.get("status")
        now = time.perf_counter()

        if progress:
            status_changed = status != last_status
            heartbeat_due = (now - last_print) >= heartbeat_seconds
            if status_changed or heartbeat_due:
                elapsed_min = (now - start) / 60.0
                print(f"[poll  ]  job={job_id}  status={status}  elapsed={elapsed_min:.1f} min")
                last_status = status
                last_print = now

        if status in TERMINAL_STATUSES:
            return data

        if deadline is not None and now >= deadline:
            raise RunPodClientError(
                f"poll_job: timeout of {timeout_seconds}s reached "
                f"(job still {status}, id={job_id})"
            )

        time.sleep(delay)
        delay = _next_delay(delay)
