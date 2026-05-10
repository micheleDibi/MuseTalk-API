"""Random-sample N short clips into a single video of target duration X minutes,
then call the local MuseTalk-API for lipsync against a given audio track.
Measures wall-clock time from the MuseTalk request to the final saved output."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from scripts.client.musetalk_client import (
    MuseTalkAPIError,
    copy_local_output,
    download_result,
    generate_lipsync_json,
    generate_lipsync_upload,
    health_check,
)
from scripts.client.video_assembler import build_random_video, probe_duration


def _positive_float(value: str) -> float:
    f = float(value)
    if f <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {value}")
    return f


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="synth_random_lipsync",
        description="Build a random-sampled video from N clips and lipsync it via MuseTalk-API.",
    )
    p.add_argument("--clips-dir", type=Path, required=True,
                   help="Directory with N short clips (same subject)")
    p.add_argument("--audio", type=Path, required=True,
                   help="Driving audio track (wav/mp3/...)")
    p.add_argument("--minutes", type=_positive_float, default=None,
                   help="Target duration in minutes. If omitted, uses the duration of --audio.")
    p.add_argument("--output", type=Path, required=True,
                   help="Final lipsynced .mp4 output path")

    p.add_argument("--api-url", default="http://localhost:8000")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for clip sampling (for reproducibility)")

    p.add_argument("--enhance", action="store_true",
                   help="Apply GFPGAN face enhancement (~3x slower)")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Inference batch size (1-32). Lower to 4 if OOM or with --enhance")
    p.add_argument("--bbox-shift", type=int, default=0)
    p.add_argument("--extra-margin", type=int, default=10)
    p.add_argument("--parsing-mode", default="jaw", choices=["jaw", "raw"])
    p.add_argument("--left-cheek-width", type=int, default=90)
    p.add_argument("--right-cheek-width", type=int, default=90)
    p.add_argument("--gfpgan-weight", type=float, default=0.5)
    p.add_argument("--fps", type=int, default=25,
                   help="Frames per second (used for both concat and API)")

    p.add_argument("--use-nvenc", action="store_true",
                   help="Use h264_nvenc for concat (much faster on RTX GPUs)")
    p.add_argument("--use-upload", action="store_true",
                   help="Force /generate (multipart upload) instead of /generate/json")

    p.add_argument("--intermediate-dir", type=Path, default=Path("data/generated"),
                   help="Where to put video_completo.mp4")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="Do not delete video_completo.mp4 after lipsync")

    p.add_argument("--api-timeout-seconds", type=float, default=None,
                   help="HTTP timeout in seconds for MuseTalk API (default: no timeout)")

    return p.parse_args(argv)


def _format_elapsed(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.clips_dir.is_dir():
        print(f"ERROR: --clips-dir does not exist or is not a directory: {args.clips_dir}",
              file=sys.stderr)
        return 2
    if not args.audio.is_file():
        print(f"ERROR: --audio file not found: {args.audio}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)

    print(f"[health]  Pinging {args.api_url}/health ...")
    try:
        health = health_check(args.api_url)
    except MuseTalkAPIError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    print(f"[health]  ok — gpu={health.get('gpu_name')}  "
          f"vram={health.get('gpu_memory_gb')} GB  "
          f"models_loaded={health.get('models_loaded')}")

    if args.minutes is None:
        audio_duration_s = probe_duration(args.audio)
        target_minutes = audio_duration_s / 60.0
        print(f"[audio ]  duration={audio_duration_s:.2f}s ({target_minutes:.3f} min) "
              f"— using as target")
    else:
        target_minutes = args.minutes
        print(f"[audio ]  target overridden by --minutes={target_minutes}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    intermediate = args.intermediate_dir / f"video_completo_{ts}.mp4"

    print(f"[build ]  random-sampling clips from {args.clips_dir} "
          f"-> {intermediate} ({target_minutes:.3f} min target)")
    build_meta = build_random_video(
        clips_dir=args.clips_dir,
        output_path=intermediate,
        target_minutes=target_minutes,
        seed=args.seed,
        target_fps=args.fps,
        use_nvenc=args.use_nvenc,
    )
    print(f"[build ]  done in {build_meta['build_time_s']:.2f}s — "
          f"{build_meta['sampled_count']} clips concatenated, "
          f"actual={build_meta['actual_duration_s']:.2f}s "
          f"target={build_meta['target_duration_s']:.2f}s")

    output_name = f"{args.output.stem}_{ts}"

    common_params = dict(
        enhance=args.enhance,
        bbox_shift=args.bbox_shift,
        extra_margin=args.extra_margin,
        parsing_mode=args.parsing_mode,
        left_cheek_width=args.left_cheek_width,
        right_cheek_width=args.right_cheek_width,
        fps=args.fps,
        batch_size=args.batch_size,
        gfpgan_weight=args.gfpgan_weight,
        timeout=args.api_timeout_seconds,
    )

    start_dt = datetime.now()
    start_pc = time.perf_counter()
    endpoint_label = "/generate (multipart)" if args.use_upload else "/generate/json"
    print(f"[start ]  {start_dt.isoformat()}  -> POST {args.api_url}{endpoint_label}")

    try:
        if args.use_upload:
            api_meta = generate_lipsync_upload(
                api_url=args.api_url,
                video_path=intermediate.resolve(),
                audio_path=args.audio.resolve(),
                output_name=output_name,
                **common_params,
            )
            download_result(args.api_url, api_meta["filename"], args.output)
        else:
            api_meta = generate_lipsync_json(
                api_url=args.api_url,
                video_path=intermediate.resolve(),
                audio_path=args.audio.resolve(),
                output_name=output_name,
                **common_params,
            )
            copy_local_output(Path(api_meta["output_video_path"]), args.output)
    except MuseTalkAPIError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.batch_size > 4:
            print(f"HINT: if VRAM is the issue, retry with --batch-size 4", file=sys.stderr)
        return 4

    end_pc = time.perf_counter()
    end_dt = datetime.now()
    elapsed = end_pc - start_pc

    print(f"[end   ]  {end_dt.isoformat()}")
    print(f"[elapsed] {elapsed:.2f} s   ({elapsed/60:.2f} min)   "
          f"HH:MM:SS={_format_elapsed(elapsed)}")
    print(f"[api   ]  processing_time_seconds={api_meta.get('processing_time_seconds')}")
    print(f"[output]  {args.output.resolve()}")

    metadata_path = args.output.with_suffix(".json")
    metadata = {
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "health": health,
        "build_phase": build_meta,
        "lipsync_phase": {
            "endpoint": endpoint_label,
            "start_iso": start_dt.isoformat(),
            "end_iso": end_dt.isoformat(),
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
            "elapsed_hms": _format_elapsed(elapsed),
            "api_processing_time_seconds": api_meta.get("processing_time_seconds"),
            "api_response": api_meta,
        },
        "output_path": str(args.output.resolve()),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    print(f"[meta  ]  {metadata_path}")

    if not args.keep_intermediate:
        try:
            intermediate.unlink()
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    sys.exit(main())
