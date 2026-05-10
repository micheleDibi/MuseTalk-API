"""Video assembly via ffmpeg: probe duration, random-sample clips with replacement,
concatenate with re-encoding, trim to exact duration."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

DEFAULT_VIDEO_EXTS: tuple[str, ...] = (".mp4", ".mov", ".mkv", ".webm", ".avi")


def _run(cmd: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=True,
        capture_output=capture,
        text=True,
    )


def probe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    p = _run(cmd)
    out = (p.stdout or "").strip()
    if not out:
        raise RuntimeError(f"ffprobe returned empty duration for {video_path}")
    return float(out)


def probe_resolution(video_path: Path) -> tuple[int, int]:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path),
    ]
    p = _run(cmd)
    data = json.loads(p.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"ffprobe found no video stream in {video_path}")
    s = streams[0]
    return int(s["width"]), int(s["height"])


def list_clips(clips_dir: Path, exts: tuple[str, ...] = DEFAULT_VIDEO_EXTS) -> list[Path]:
    if not clips_dir.is_dir():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")
    clips = sorted(p for p in clips_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)
    if not clips:
        raise FileNotFoundError(f"No video clips with extensions {exts} in {clips_dir}")
    return clips


def sample_clips_until_duration(
    clips: list[Path],
    durations: dict[Path, float],
    target_seconds: float,
    rng: random.Random,
) -> list[Path]:
    if target_seconds <= 0:
        raise ValueError("target_seconds must be > 0")
    if not clips:
        raise ValueError("clips list is empty")

    selected: list[Path] = []
    total = 0.0
    while total < target_seconds:
        c = rng.choice(clips)
        selected.append(c)
        total += durations[c]
    return selected


def concat_clips_reencode(
    clip_paths: list[Path],
    output_path: Path,
    target_fps: int = 25,
    target_resolution: tuple[int, int] | None = None,
    use_nvenc: bool = False,
) -> None:
    if not clip_paths:
        raise ValueError("clip_paths is empty")

    if target_resolution is None:
        target_resolution = probe_resolution(clip_paths[0])
    w, h = target_resolution

    inputs: list[str] = []
    for c in clip_paths:
        inputs.extend(["-i", str(c)])

    n = len(clip_paths)
    norm_chains = [
        f"[{i}:v]scale={w}:{h}:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={target_fps},format=yuv420p[v{i}]"
        for i in range(n)
    ]
    concat_inputs = "".join(f"[v{i}]" for i in range(n))
    filter_complex = ";".join(norm_chains) + f";{concat_inputs}concat=n={n}:v=1:a=0[v]"

    if use_nvenc:
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "warning",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[v]",
        *codec_args,
        "-an",
        str(output_path),
    ]
    _run(cmd, capture=False)


def trim_to_exact_duration(
    input_path: Path,
    output_path: Path,
    duration_seconds: float,
    use_nvenc: bool = False,
) -> None:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be > 0")

    if use_nvenc:
        codec_args = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"]
    else:
        codec_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "warning",
        "-i", str(input_path),
        "-t", f"{duration_seconds:.3f}",
        *codec_args,
        "-an",
        str(output_path),
    ]
    _run(cmd, capture=False)


def build_random_video(
    clips_dir: Path,
    output_path: Path,
    target_minutes: float,
    seed: int | None = None,
    target_fps: int = 25,
    use_nvenc: bool = False,
) -> dict:
    if target_minutes <= 0:
        raise ValueError("target_minutes must be > 0")

    target_seconds = target_minutes * 60.0
    rng = random.Random(seed) if seed is not None else random.Random()

    t0 = time.perf_counter()

    clips = list_clips(clips_dir)
    durations = {c: probe_duration(c) for c in clips}

    sampled = sample_clips_until_duration(clips, durations, target_seconds, rng)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="musetalk_concat_"))
    try:
        raw_concat = tmpdir / "concat_raw.mp4"
        concat_clips_reencode(
            clip_paths=sampled,
            output_path=raw_concat,
            target_fps=target_fps,
            use_nvenc=use_nvenc,
        )
        trim_to_exact_duration(
            input_path=raw_concat,
            output_path=output_path,
            duration_seconds=target_seconds,
            use_nvenc=use_nvenc,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    actual = probe_duration(output_path)
    build_time = time.perf_counter() - t0

    return {
        "total_clips_input": len(clips),
        "input_clip_durations_s": {str(p): durations[p] for p in clips},
        "sampled_clips": [str(p) for p in sampled],
        "sampled_count": len(sampled),
        "target_duration_s": target_seconds,
        "actual_duration_s": actual,
        "target_fps": target_fps,
        "use_nvenc": use_nvenc,
        "seed": seed,
        "build_time_s": build_time,
        "output_path": str(output_path.resolve()),
    }
