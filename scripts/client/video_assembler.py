"""Video assembly via ffmpeg: probe duration, random-sample clips with replacement,
concatenate with re-encoding, trim to exact duration."""

from __future__ import annotations

import json
import math
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
        stdin=subprocess.DEVNULL,
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


def probe_fps(video_path: Path) -> float:
    """Return the source video's native frame rate (e.g. 24.0 for 24fps clips).

    Needed because if a clip is 24fps but the assembled video targets 25fps,
    ffmpeg duplicates frames to fill the gap; the per-clip cache (latents,
    parsing, bboxes) stores only the 24fps native frames, so the assembled
    timeline must be mapped back via the source/target ratio.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    p = _run(cmd)
    out = (p.stdout or "").strip()
    if not out:
        raise RuntimeError(f"ffprobe returned empty fps for {video_path}")
    if "/" in out:
        num, den = out.split("/")
        den_f = float(den)
        if den_f == 0:
            raise RuntimeError(f"ffprobe returned zero denominator for {video_path}: {out}")
        return float(num) / den_f
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
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
        f"fps={target_fps}:round=up,format=yuv420p[v{i}]"
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
    source_fps_map = {c: probe_fps(c) for c in clips}

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
    total_frames = max(1, int(round(actual * target_fps)))
    frame_to_clip, frame_to_clip_idx = _build_frame_to_clip_map(
        sampled, durations, source_fps_map, target_fps, total_frames
    )
    build_time = time.perf_counter() - t0

    unique_clips = list(dict.fromkeys(str(p) for p in sampled))

    return {
        "total_clips_input": len(clips),
        "input_clip_durations_s": {str(p): durations[p] for p in clips},
        "input_clip_source_fps": {str(p): source_fps_map[p] for p in clips},
        "sampled_clips": [str(p) for p in sampled],
        "sampled_count": len(sampled),
        "unique_clips_used": unique_clips,
        "frame_to_clip": [str(p) for p in frame_to_clip],
        "frame_to_clip_idx": frame_to_clip_idx,
        "total_frames": total_frames,
        "target_duration_s": target_seconds,
        "actual_duration_s": actual,
        "target_fps": target_fps,
        "use_nvenc": use_nvenc,
        "seed": seed,
        "build_time_s": build_time,
        "output_path": str(output_path.resolve()),
    }


def _build_frame_to_clip_map(
    sampled: list[Path],
    durations: dict[Path, float],
    source_fps_map: dict[Path, float],
    target_fps: int,
    total_frames: int,
) -> tuple[list[Path], list[int]]:
    """Return two parallel lists of length ``total_frames``:

    - ``frame_to_clip``: per-frame source clip path
    - ``frame_to_clip_idx``: per-frame index *within that clip's native
      timeline* (0-based, in the clip's source fps).

    Cumulative tracking is in **frame counts** (integer), not seconds. The
    ffmpeg filter ``fps={target_fps}:round=up`` applied in
    :func:`concat_clips_reencode` produces exactly ``ceil(duration *
    target_fps)`` frames per clip in the assembled stream. Tracking the
    same integer here keeps client and ffmpeg perfectly aligned — fixes
    the ~13-frame (520 ms) cumulative drift at end-of-video observed in
    v7's `cumulative_s` arithmetic over 160 clips.

    For a clip at 24 fps assembled into a 25 fps stream, the segment
    occupies ``ceil(5.917 × 25) = 148`` frames; per-frame source idx is
    ``int(local_target_frame * source_fps / target_fps)`` clamped to
    ``n_native - 1``, mirroring ffmpeg's per-frame nearest-neighbor
    duplication.
    """
    if not sampled or total_frames <= 0:
        return [], []

    clip_per_frame: list[Path | None] = [None] * total_frames
    idx_per_frame: list[int] = [0] * total_frames

    start_f = 0
    last_clip_state: tuple[Path, int, int] | None = None  # (clip, start_f, n_native)
    for clip in sampled:
        if start_f >= total_frames:
            break
        # Must match `fps={target_fps}:round=up` in concat_clips_reencode.
        clip_frames = math.ceil(durations[clip] * target_fps)
        end_f = min(start_f + clip_frames, total_frames)
        source_fps = source_fps_map[clip]
        n_native = max(1, int(round(durations[clip] * source_fps)))
        scale = source_fps / target_fps
        for f in range(start_f, end_f):
            clip_per_frame[f] = clip
            local = f - start_f
            idx_per_frame[f] = min(int(local * scale), n_native - 1)
        last_clip_state = (clip, start_f, n_native)
        start_f = end_f

    # Trailing rounding remainder (rare; only if total_frames exceeds the
    # sum of per-clip frame counts). Fall back on the last clip's last frame.
    if last_clip_state is not None:
        last_clip, last_start, last_n = last_clip_state
        last_scale = source_fps_map[last_clip] / target_fps
        for i in range(total_frames):
            if clip_per_frame[i] is None:
                clip_per_frame[i] = last_clip
                local = max(0, i - last_start)
                idx_per_frame[i] = min(int(local * last_scale), last_n - 1)
    return clip_per_frame, idx_per_frame  # type: ignore[return-value]
