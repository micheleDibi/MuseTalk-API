"""HTTP client for MuseTalk-API: health check, /generate/json (path-based),
/generate (multipart upload fallback), /download streaming."""

from __future__ import annotations

import shutil
from pathlib import Path

import requests


class MuseTalkAPIError(RuntimeError):
    pass


def health_check(api_url: str, timeout: float = 10.0) -> dict:
    url = f"{api_url.rstrip('/')}/health"
    try:
        r = requests.get(url, timeout=timeout)
    except requests.exceptions.RequestException as e:
        raise MuseTalkAPIError(
            f"Could not reach MuseTalk server at {url}. "
            f"Start it with: `uvicorn api.main:app --host 0.0.0.0 --port 8000` "
            f"from the repo root. ({e})"
        ) from e

    if r.status_code != 200:
        raise MuseTalkAPIError(f"Health endpoint returned HTTP {r.status_code}: {r.text}")

    data = r.json()
    if not data.get("models_loaded"):
        raise MuseTalkAPIError(
            f"Server is up but models are not loaded yet: {data}. "
            f"Wait ~30-60s after starting uvicorn for first-time model loading."
        )
    return data


def generate_lipsync_json(
    api_url: str,
    video_path: Path,
    audio_path: Path,
    output_name: str,
    enhance: bool = False,
    bbox_shift: int = 0,
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    left_cheek_width: int = 90,
    right_cheek_width: int = 90,
    fps: int = 25,
    batch_size: int = 8,
    gfpgan_weight: float = 0.5,
    timeout: float | None = None,
) -> dict:
    payload = {
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "enhance": enhance,
        "bbox_shift": bbox_shift,
        "extra_margin": extra_margin,
        "parsing_mode": parsing_mode,
        "left_cheek_width": left_cheek_width,
        "right_cheek_width": right_cheek_width,
        "fps": fps,
        "batch_size": batch_size,
        "output_name": output_name,
        "gfpgan_weight": gfpgan_weight,
    }
    url = f"{api_url.rstrip('/')}/generate/json"
    r = requests.post(url, json=payload, timeout=timeout)
    if r.status_code != 200:
        raise MuseTalkAPIError(
            f"POST {url} -> HTTP {r.status_code}: {r.text}"
        )
    return r.json()


def generate_lipsync_upload(
    api_url: str,
    video_path: Path,
    audio_path: Path,
    output_name: str,
    enhance: bool = False,
    bbox_shift: int = 0,
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    left_cheek_width: int = 90,
    right_cheek_width: int = 90,
    fps: int = 25,
    batch_size: int = 8,
    gfpgan_weight: float = 0.5,
    timeout: float | None = None,
) -> dict:
    url = f"{api_url.rstrip('/')}/generate"
    data = {
        "enhance": str(enhance).lower(),
        "bbox_shift": str(bbox_shift),
        "extra_margin": str(extra_margin),
        "parsing_mode": parsing_mode,
        "left_cheek_width": str(left_cheek_width),
        "right_cheek_width": str(right_cheek_width),
        "fps": str(fps),
        "batch_size": str(batch_size),
        "output_name": output_name,
        "gfpgan_weight": str(gfpgan_weight),
    }
    with video_path.open("rb") as fv, audio_path.open("rb") as fa:
        files = {
            "source": (video_path.name, fv, "video/mp4"),
            "audio": (audio_path.name, fa, "application/octet-stream"),
        }
        r = requests.post(url, data=data, files=files, timeout=timeout)
    if r.status_code != 200:
        raise MuseTalkAPIError(f"POST {url} -> HTTP {r.status_code}: {r.text}")
    return r.json()


def download_result(api_url: str, filename: str, dest_path: Path, chunk_size: int = 1 << 20) -> None:
    url = f"{api_url.rstrip('/')}/download/{filename}"
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300) as r:
        if r.status_code != 200:
            raise MuseTalkAPIError(f"GET {url} -> HTTP {r.status_code}: {r.text}")
        with dest_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)


def copy_local_output(server_output_path: Path, dest_path: Path) -> None:
    if not server_output_path.exists():
        raise MuseTalkAPIError(
            f"Server reported output at {server_output_path} but file is missing. "
            f"This typically means uvicorn was launched from a different CWD than the repo root."
        )
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(server_output_path, dest_path)
