"""Cache and computation of face bboxes for the source clip set.

The MuseTalk pipeline normally runs s3fd + DWPose face detection on every
frame (23.5k for 15 min of video) — that step dominates the wall clock.

Since the source clips in this project always show the same subject in a
fixed framing, a single bbox per *unique* clip is enough. This module:

1. Hashes the source clip set deterministically (paths + size + mtime + bbox_shift).
2. Looks up a local JSON cache under ``data/manifests/``.
3. On cache miss, extracts one probe PNG per unique clip via ffmpeg, uploads
   them to R2, submits a lightweight "preprocess_clips" job to the same
   RunPod endpoint that runs the lipsync, and saves the returned bboxes.
4. Expands the per-clip bbox dictionary into a per-frame array matching the
   final video_completo.mp4 frame count, which the handler passes straight
   to ``MuseTalk.generate(..., precomputed_bboxes=...)``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from scripts.client.runpod_client import (
    RunPodClientError,
    poll_job,
    r2_delete,
    r2_exists,
    r2_presigned_url,
    r2_upload,
    submit_job,
)

DEFAULT_CACHE_DIR = Path("data/manifests")
PROBE_FRAME_INDEX = 0


def compute_set_hash(clips: list[Path], bbox_shift: int) -> str:
    """Deterministic hash of a clip set (resolved path + size + mtime) + bbox_shift.

    Truncated to 32 hex chars — collision risk negligible for any realistic
    project, and short enough to use as a filename.
    """
    h = hashlib.sha256()
    h.update(f"bbox_shift={bbox_shift}\n".encode())
    for c in sorted(clips, key=lambda p: str(p)):
        st = c.stat()
        h.update(f"{c.resolve()}\t{st.st_size}\t{int(st.st_mtime)}\n".encode())
    return h.hexdigest()[:32]


def compute_full_set_hash(
    clips: list[Path],
    bbox_shift: int,
    extra_margin: int,
    parsing_mode: str,
    left_cheek_width: int,
    right_cheek_width: int,
) -> str:
    """Hash that includes everything the cached blobs depend on.

    v3 cache stores **per-frame** bboxes, latents and 512x512 parsing masks
    so head motion within a clip is tracked. Blend masks are rebuilt at
    lipsync time from the parsing + bbox + extra_margin, so extra_margin
    and parsing parameters are still part of the hash (different extra_margin
    yields different blend masks even from identical parsing).
    """
    h = hashlib.sha256()
    h.update(
        f"v3|bbox_shift={bbox_shift}|extra_margin={extra_margin}|"
        f"parsing_mode={parsing_mode}|left_cw={left_cheek_width}|"
        f"right_cw={right_cheek_width}\n".encode()
    )
    for c in sorted(clips, key=lambda p: str(p)):
        st = c.stat()
        h.update(f"{c.resolve()}\t{st.st_size}\t{int(st.st_mtime)}\n".encode())
    return h.hexdigest()[:32]


def extract_probe_frame(clip: Path, dest_dir: Path, frame_idx: int = PROBE_FRAME_INDEX) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{clip.stem}.png"
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(clip),
        "-vf", f"select=eq(n\\,{frame_idx})",
        "-frames:v", "1",
        str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if not dest.is_file():
        raise RuntimeError(f"ffmpeg failed to extract probe frame from {clip}")
    return dest


def cache_load(set_hash: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> dict | None:
    f = cache_dir / f"{set_hash}.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def cache_save(set_hash: str, payload: dict, cache_dir: Path = DEFAULT_CACHE_DIR) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    f = cache_dir / f"{set_hash}.json"
    f.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def request_bboxes_runpod(
    unique_clips: list[Path],
    bbox_shift: int,
    r2_prefix: str,
    timeout_seconds: float | None = None,
) -> dict[str, tuple[int, int, int, int]]:
    """Upload one probe frame per clip to R2, then submit a preprocess job."""
    probe_dir = Path(tempfile.mkdtemp(prefix="musetalk_probes_"))
    uploaded_keys: list[str] = []
    try:
        probe_urls: dict[str, str] = {}
        for clip in unique_clips:
            probe_png = extract_probe_frame(clip, probe_dir)
            key = f"{r2_prefix}/{clip.name}.png"
            url = r2_upload(probe_png, key, presign_expires=7200)
            probe_urls[clip.name] = url
            uploaded_keys.append(key)

        job_id = submit_job(
            {
                "action": "preprocess_clips",
                "probe_frames": probe_urls,
                "bbox_shift": bbox_shift,
            },
            execution_timeout_ms=600_000,
        )
        result = poll_job(job_id, timeout_seconds=timeout_seconds, heartbeat_seconds=30.0)

        if result.get("status") != "COMPLETED":
            detail = (result.get("output") or {}).get("detail")
            raise RunPodClientError(
                f"preprocess job ended with status={result.get('status')} detail={detail}"
            )

        handler_output = result.get("output") or {}
        if handler_output.get("status") != "success":
            raise RunPodClientError(
                f"preprocess handler returned non-success: {handler_output}"
            )

        bboxes_raw = handler_output.get("bboxes") or {}
        out: dict[str, tuple[int, int, int, int]] = {}
        for name, b in bboxes_raw.items():
            if b is None or len(b) != 4:
                raise RunPodClientError(
                    f"preprocess returned invalid bbox for {name}: {b}"
                )
            out[name] = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        missing = {c.name for c in unique_clips} - set(out.keys())
        if missing:
            raise RunPodClientError(
                f"preprocess result missing bboxes for clips: {sorted(missing)}"
            )
        return out
    finally:
        try:
            if uploaded_keys:
                r2_delete(uploaded_keys)
        except Exception:
            pass
        shutil.rmtree(probe_dir, ignore_errors=True)


def build_per_frame_bboxes(
    frame_to_clip: list[str],
    clip_bboxes: dict[str, tuple[int, int, int, int]],
) -> list[list[int]]:
    return [list(clip_bboxes[Path(p).name]) for p in frame_to_clip]


def get_or_compute_manifest(
    unique_clips: list[Path],
    frame_to_clip: list[str],
    bbox_shift: int,
    r2_prefix: str,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    poll_timeout: float | None = None,
) -> tuple[list[list[int]], dict]:
    """High-level entrypoint. Returns ``(per_frame_bboxes, meta)``.

    Cache hit: meta has ``cache_hit=True`` and ``set_hash``.
    Cache miss: runs the preprocess job, persists the result, and reports
    elapsed time in ``meta``.
    """
    set_hash = compute_set_hash(unique_clips, bbox_shift)
    expected_names = {c.name for c in unique_clips}

    cached = cache_load(set_hash, cache_dir)
    clip_bboxes: dict[str, tuple[int, int, int, int]] | None = None
    cache_hit = False
    preprocess_elapsed: float | None = None

    if cached and set(cached.get("clip_bboxes", {}).keys()) == expected_names:
        clip_bboxes = {
            k: tuple(int(x) for x in v) for k, v in cached["clip_bboxes"].items()
        }
        cache_hit = True

    if clip_bboxes is None:
        t0 = time.perf_counter()
        clip_bboxes = request_bboxes_runpod(
            unique_clips, bbox_shift, r2_prefix, timeout_seconds=poll_timeout
        )
        preprocess_elapsed = time.perf_counter() - t0
        cache_save(
            set_hash,
            {
                "set_hash": set_hash,
                "bbox_shift": bbox_shift,
                "clip_bboxes": {k: list(v) for k, v in clip_bboxes.items()},
                "preprocess_elapsed_s": preprocess_elapsed,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            cache_dir,
        )

    per_frame_bboxes = build_per_frame_bboxes(frame_to_clip, clip_bboxes)
    meta = {
        "set_hash": set_hash,
        "cache_hit": cache_hit,
        "preprocess_elapsed_s": preprocess_elapsed,
        "unique_clip_count": len(unique_clips),
        "clip_bboxes": {k: list(v) for k, v in clip_bboxes.items()},
    }
    return per_frame_bboxes, meta


def request_full_preprocess_runpod(
    unique_clips: list[Path],
    full_set_hash: str,
    bbox_shift: int,
    extra_margin: int,
    parsing_mode: str,
    left_cheek_width: int,
    right_cheek_width: int,
    timeout_seconds: float | None = None,
) -> dict[str, dict]:
    """Upload each unique clip to R2 (under cache/<hash>/clips/<name>),
    submit a ``preprocess_full`` job and return per-clip metadata:

    ``{clip_name: {bboxes_key, latents_key, parsing_key, n_frames, frame_shape}}``

    The handler does **per-frame** face detection inside preprocess_full so we
    do NOT pre-compute bboxes on the client. Presigned URLs are NOT persisted
    (they expire). The caller regenerates fresh URLs via
    :func:`r2_presigned_url` on each lipsync submission.
    """
    r2_prefix = f"cache/{full_set_hash}"
    uploaded_clip_keys: list[str] = []

    clip_payload: list[dict] = []
    t_up = time.perf_counter()
    for clip in unique_clips:
        name = clip.name
        clip_key = f"{r2_prefix}/clips/{name}"
        url = r2_upload(clip, clip_key, presign_expires=7200)
        uploaded_clip_keys.append(clip_key)
        clip_payload.append({"name": name, "url": url})
    print(
        f"[preprocess_full] uploaded {len(unique_clips)} clips to R2 in "
        f"{time.perf_counter() - t_up:.1f}s"
    )

    try:
        job_id = submit_job(
            {
                "action": "preprocess_full",
                "clips": clip_payload,
                "bbox_shift": bbox_shift,
                "extra_margin": extra_margin,
                "parsing_mode": parsing_mode,
                "left_cheek_width": left_cheek_width,
                "right_cheek_width": right_cheek_width,
                "r2_prefix": r2_prefix,
                "presign_expires": 86400,
            },
            execution_timeout_ms=3_600_000,
        )
        result = poll_job(job_id, timeout_seconds=timeout_seconds, heartbeat_seconds=30.0)

        if result.get("status") != "COMPLETED":
            detail = (result.get("output") or {}).get("detail")
            raise RunPodClientError(
                f"preprocess_full ended status={result.get('status')} detail={detail}"
            )
        handler_output = result.get("output") or {}
        if handler_output.get("status") != "success":
            raise RunPodClientError(
                f"preprocess_full handler non-success: {handler_output}"
            )
        clips_result = handler_output.get("clips") or {}
        if set(clips_result.keys()) != {c.name for c in unique_clips}:
            raise RunPodClientError(
                f"preprocess_full returned clips do not match request: "
                f"got {sorted(clips_result.keys())}, "
                f"expected {sorted(c.name for c in unique_clips)}"
            )
        return clips_result
    finally:
        if uploaded_clip_keys:
            try:
                r2_delete(uploaded_clip_keys)
            except Exception:
                pass


def get_or_compute_full_manifest(
    unique_clips: list[Path],
    frame_to_clip: list[str],
    frame_to_clip_idx: list[int],
    bbox_shift: int,
    extra_margin: int,
    parsing_mode: str,
    left_cheek_width: int,
    right_cheek_width: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    poll_timeout: float | None = None,
) -> tuple[dict[str, str], dict[str, str], dict[str, str], dict]:
    """End-to-end v3 manifest with **per-frame** bboxes + latents + parsing.

    Returns ``(bboxes_urls, latents_urls, parsing_urls, meta)``:

    - ``bboxes_urls``: ``{clip_name: presigned_url to bboxes.json}``
    - ``latents_urls``: ``{clip_name: presigned_url to latents.pt}``
    - ``parsing_urls``: ``{clip_name: presigned_url to parsing.npz}``
    - ``meta`` includes ``cache_hit`` and the full set hash

    Cache miss runs **preprocess_full** which does its own per-frame face
    detection — no separate preprocess_clips call. Cache hit just re-presigns
    existing R2 keys.
    """
    full_set_hash = compute_full_set_hash(
        unique_clips,
        bbox_shift,
        extra_margin,
        parsing_mode,
        left_cheek_width,
        right_cheek_width,
    )

    expected_names = {c.name for c in unique_clips}
    cached = cache_load(full_set_hash, cache_dir)
    full_payload: dict[str, dict] | None = None
    cache_hit = False
    full_elapsed: float | None = None

    if cached and set((cached.get("full_payload") or {}).keys()) == expected_names:
        full_payload = cached["full_payload"]
        keys: list[str] = []
        for payload in full_payload.values():
            for k in ("bboxes_key", "latents_key", "parsing_key"):
                if k in payload:
                    keys.append(payload[k])
        # require all three keys per clip to consider the cache valid
        per_clip_keys_ok = all(
            all(k in payload for k in ("bboxes_key", "latents_key", "parsing_key"))
            for payload in full_payload.values()
        )
        if per_clip_keys_ok and all(r2_exists(k) for k in keys):
            cache_hit = True
        else:
            print("[manifest] local cache present but R2 blobs missing or schema "
                  "outdated — recomputing")
            full_payload = None

    if full_payload is None:
        t_full = time.perf_counter()
        full_payload = request_full_preprocess_runpod(
            unique_clips,
            full_set_hash,
            bbox_shift,
            extra_margin,
            parsing_mode,
            left_cheek_width,
            right_cheek_width,
            timeout_seconds=poll_timeout,
        )
        full_elapsed = time.perf_counter() - t_full

        cache_save(
            full_set_hash,
            {
                "full_set_hash": full_set_hash,
                "bbox_shift": bbox_shift,
                "extra_margin": extra_margin,
                "parsing_mode": parsing_mode,
                "left_cheek_width": left_cheek_width,
                "right_cheek_width": right_cheek_width,
                "full_payload": full_payload,
                "preprocess_full_elapsed_s": full_elapsed,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            cache_dir,
        )

    bboxes_urls = {
        name: r2_presigned_url(payload["bboxes_key"], expires_in=86400)
        for name, payload in full_payload.items()
    }
    latents_urls = {
        name: r2_presigned_url(payload["latents_key"], expires_in=86400)
        for name, payload in full_payload.items()
    }
    parsing_urls = {
        name: r2_presigned_url(payload["parsing_key"], expires_in=86400)
        for name, payload in full_payload.items()
    }
    meta = {
        "full_set_hash": full_set_hash,
        "cache_hit": cache_hit,
        "preprocess_full_elapsed_s": full_elapsed,
        "unique_clip_count": len(unique_clips),
        "clip_n_frames": {
            name: int(payload.get("n_frames", 0))
            for name, payload in full_payload.items()
        },
    }
    return bboxes_urls, latents_urls, parsing_urls, meta
