"""Random-sample N short clips into a single video matching the audio duration
(or an explicit ``--minutes`` target), then call the MuseTalk lipsync service
on RunPod Serverless against the given audio track.

Measures wall-clock time from the moment the RunPod job is submitted to the
moment the final MP4 is fully saved locally.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path

from scripts.client.clip_manifest import (
    get_or_compute_full_manifest,
    get_or_compute_manifest,
)
from scripts.client.runpod_client import (
    RunPodClientError,
    ensure_env,
    poll_job,
    r2_delete,
    r2_download,
    r2_upload,
    submit_job,
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
        description="Build a random-sampled video from N clips and lipsync it via MuseTalk on RunPod.",
    )
    p.add_argument("--clips-dir", type=Path, required=True,
                   help="Directory with N short clips (same subject)")
    p.add_argument("--audio", type=Path, required=True,
                   help="Driving audio track (wav/mp3/...)")
    p.add_argument("--minutes", type=_positive_float, default=None,
                   help="Target duration in minutes. If omitted, uses the duration of --audio.")
    p.add_argument("--output", type=Path, required=True,
                   help="Final lipsynced .mp4 output path")

    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for clip sampling (for reproducibility)")

    p.add_argument("--enhance", action="store_true",
                   help="Apply GFPGAN face enhancement (adds ~5-7 min on 15 min audio). "
                        "Sharpens the generated mouth region to match the rest of the face.")
    p.add_argument("--quality", action="store_true",
                   help="Preset: enables --enhance with gfpgan-weight 0.6. Use this "
                        "when polish matters more than speed. Total runtime expected "
                        "~1.4x compared to default (e.g. 20 min for 15 min audio).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="UNet inference batch size (1-32). Default 16 fits comfortably on "
                        "H100 80GB. Drop to 4-8 with --enhance.")
    p.add_argument("--bbox-shift", type=int, default=0)
    p.add_argument("--extra-margin", type=int, default=10)
    p.add_argument("--parsing-mode", default="jaw", choices=["jaw", "raw"])
    p.add_argument("--left-cheek-width", type=int, default=90)
    p.add_argument("--right-cheek-width", type=int, default=90)
    p.add_argument("--gfpgan-weight", type=float, default=0.6,
                   help="GFPGAN blend strength: 0=original face, 1=fully enhanced. "
                        "Only effective with --enhance or --quality.")
    p.add_argument("--fps", type=int, default=25,
                   help="Frames per second (used for both concat and API). "
                        "Recommended: 25 (MuseTalk model rate). Other values, "
                        "especially non-multiples of 50, can crash inference "
                        "on long audio due to a known upstream boundary bug "
                        "in musetalk/utils/audio_processor.py.")

    p.add_argument("--use-nvenc", action="store_true",
                   help="Use h264_nvenc for concat (much faster on RTX GPUs)")

    p.add_argument("--intermediate-dir", type=Path, default=Path("data/generated"),
                   help="Where to put video_completo.mp4")
    p.add_argument("--keep-intermediate", action="store_true",
                   help="Do not delete video_completo.mp4 after lipsync")

    p.add_argument("--api-timeout-seconds", type=float, default=None,
                   help="Polling timeout for the RunPod job (default: no timeout)")
    p.add_argument("--keep-r2-objects", action="store_true",
                   help="Do not delete the uploaded inputs and the output from R2 after the run")

    p.add_argument("--skip-manifest", action="store_true",
                   help="Skip the per-clip bbox manifest (RunPod will run full s3fd+DWPose "
                        "face detection on every frame — 30-90 min on 15+ min videos).")
    p.add_argument("--skip-full-preprocess", action="store_true",
                   help="Skip the latents+masks preprocessing step. The handler will then "
                        "run VAE encode and FaceParsing during the lipsync job (slower).")
    p.add_argument("--manifest-cache-dir", type=Path, default=Path("data/manifests"),
                   help="Local cache directory for clip bbox manifests")
    p.add_argument("--preprocess-timeout-seconds", type=float, default=1800.0,
                   help="Polling timeout for preprocess jobs (default: 30 min). The full "
                        "preprocess_full pass can run ~60-90s per unique clip.")

    return p.parse_args(argv)


def _format_elapsed(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.quality:
        args.enhance = True
        # Honor an explicit user-set weight; only override the bare default.
        if args.gfpgan_weight == 0.6:
            args.gfpgan_weight = 0.6
        print(f"[quality] preset enabled: --enhance + gfpgan-weight={args.gfpgan_weight}")

    if not args.clips_dir.is_dir():
        print(f"ERROR: --clips-dir does not exist or is not a directory: {args.clips_dir}",
              file=sys.stderr)
        return 2
    if not args.audio.is_file():
        print(f"ERROR: --audio file not found: {args.audio}", file=sys.stderr)
        return 2
    if args.fps != 25:
        print(f"WARNING: --fps={args.fps} is non-standard. MuseTalk is trained at 25 fps; "
              f"values that are not multiples of 50 (e.g. 60) can hit an upstream "
              f"out-of-bounds bug at the last frame on long audio. "
              f"Consider --fps 25 unless you know what you are doing.",
              file=sys.stderr)

    try:
        ensure_env()
    except RunPodClientError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)

    if args.minutes is None:
        audio_duration_s = probe_duration(args.audio)
        target_minutes = audio_duration_s / 60.0
        print(f"[audio ]  duration={audio_duration_s:.2f}s ({target_minutes:.3f} min) "
              f"— using as target")
    else:
        target_minutes = args.minutes
        print(f"[audio ]  target overridden by --minutes={target_minutes}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{uuid.uuid4().hex[:8]}"
    intermediate = args.intermediate_dir / f"video_completo_{run_id}.mp4"

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

    video_key = f"inputs/{run_id}/video.mp4"
    audio_key = f"inputs/{run_id}/{args.audio.name}"
    output_key = f"outputs/{run_id}/{args.output.name}"
    manifest_key = f"inputs/{run_id}/manifest.json"

    start_dt = datetime.now()
    start_pc = time.perf_counter()
    print(f"[start ]  {start_dt.isoformat()}  -> preprocess + upload + RunPod job")

    job_id = None
    job_result = {}
    manifest_meta: dict = {}
    manifest_url: str | None = None
    bboxes_urls: dict[str, str] | None = None
    latents_urls: dict[str, str] | None = None
    parsing_urls: dict[str, str] | None = None
    try:
        if not args.skip_manifest:
            unique_clip_paths = [Path(p) for p in build_meta["unique_clips_used"]]
            frame_to_clip = build_meta["frame_to_clip"]
            manifest_t0 = time.perf_counter()
            if args.skip_full_preprocess:
                print(f"[probe ]  computing/loading bbox manifest only for "
                      f"{len(unique_clip_paths)} unique clip(s) ...")
                per_frame_bboxes, manifest_meta = get_or_compute_manifest(
                    unique_clips=unique_clip_paths,
                    frame_to_clip=frame_to_clip,
                    bbox_shift=args.bbox_shift,
                    r2_prefix=f"probes/{run_id}",
                    cache_dir=args.manifest_cache_dir,
                    poll_timeout=args.preprocess_timeout_seconds,
                )
                local_manifest_path = (
                    args.intermediate_dir / f"manifest_{run_id}.json"
                )
                local_manifest_path.write_text(
                    json.dumps(per_frame_bboxes), encoding="utf-8"
                )
                manifest_url = r2_upload(local_manifest_path, manifest_key)
                manifest_elapsed = time.perf_counter() - manifest_t0
                origin = "cache" if manifest_meta.get("cache_hit") else "fresh"
                print(f"[probe ]  manifest ready ({origin}) in {manifest_elapsed:.2f}s — "
                      f"{len(per_frame_bboxes)} per-frame bboxes (legacy bbox-only path); "
                      f"set_hash={manifest_meta.get('set_hash')}")
                print(f"[upload]  manifest uploaded to R2 ({manifest_key})")
            else:
                print(f"[probe ]  computing/loading v3 FULL manifest "
                      f"(per-frame bbox + latents + parsing) for "
                      f"{len(unique_clip_paths)} unique clip(s) ...")
                bboxes_urls, latents_urls, parsing_urls, manifest_meta = (
                    get_or_compute_full_manifest(
                        unique_clips=unique_clip_paths,
                        frame_to_clip=frame_to_clip,
                        frame_to_clip_idx=build_meta["frame_to_clip_idx"],
                        bbox_shift=args.bbox_shift,
                        extra_margin=args.extra_margin,
                        parsing_mode=args.parsing_mode,
                        left_cheek_width=args.left_cheek_width,
                        right_cheek_width=args.right_cheek_width,
                        cache_dir=args.manifest_cache_dir,
                        poll_timeout=args.preprocess_timeout_seconds,
                    )
                )
                manifest_elapsed = time.perf_counter() - manifest_t0
                origin = "cache" if manifest_meta.get("cache_hit") else "fresh"
                print(f"[probe ]  v3 manifest ready ({origin}) in {manifest_elapsed:.2f}s — "
                      f"{len(latents_urls)} latents + {len(parsing_urls)} parsing + "
                      f"{len(bboxes_urls)} bbox blobs; "
                      f"full_set_hash={manifest_meta.get('full_set_hash')}")

        upload_t0 = time.perf_counter()
        video_url = r2_upload(intermediate, video_key)
        audio_url = r2_upload(args.audio, audio_key)
        upload_elapsed = time.perf_counter() - upload_t0
        print(f"[upload]  R2 inputs ready in {upload_elapsed:.2f}s "
              f"(video={video_key}, audio={audio_key})")

        input_payload = {
            "video_url": video_url,
            "audio_url": audio_url,
            "output_object_key": output_key,
            "enhance": args.enhance,
            "bbox_shift": args.bbox_shift,
            "extra_margin": args.extra_margin,
            "parsing_mode": args.parsing_mode,
            "left_cheek_width": args.left_cheek_width,
            "right_cheek_width": args.right_cheek_width,
            "fps": args.fps,
            "batch_size": args.batch_size,
            "gfpgan_weight": args.gfpgan_weight,
        }
        if manifest_url:
            input_payload["manifest_url"] = manifest_url
        if bboxes_urls and latents_urls and parsing_urls:
            input_payload["bboxes_urls"] = bboxes_urls
            input_payload["latents_urls"] = latents_urls
            input_payload["parsing_urls"] = parsing_urls
            input_payload["frame_to_clip"] = [
                Path(p).name for p in build_meta["frame_to_clip"]
            ]
            input_payload["frame_to_clip_idx"] = build_meta["frame_to_clip_idx"]
        job_id = submit_job(input_payload)
        print(f"[submit]  RunPod job_id={job_id}")

        job_result = poll_job(job_id, timeout_seconds=args.api_timeout_seconds)

        if job_result.get("status") != "COMPLETED":
            detail = (job_result.get("output") or {}).get("detail")
            trace = (job_result.get("output") or {}).get("trace")
            print(f"ERROR: RunPod job ended with status={job_result.get('status')}",
                  file=sys.stderr)
            if detail:
                print(f"  detail: {detail}", file=sys.stderr)
            if trace:
                print(trace, file=sys.stderr)
            return 4

        handler_output = job_result.get("output") or {}
        if handler_output.get("status") != "success":
            print(f"ERROR: handler returned non-success payload: {handler_output}",
                  file=sys.stderr)
            return 4

        result_output_url = handler_output["output_url"]
        download_t0 = time.perf_counter()
        r2_download(result_output_url, args.output)
        download_elapsed = time.perf_counter() - download_t0
        print(f"[dload ]  output downloaded in {download_elapsed:.2f}s")

    except RunPodClientError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if args.batch_size > 4:
            print("HINT: if VRAM is the issue, retry with --batch-size 4", file=sys.stderr)
        return 4

    end_pc = time.perf_counter()
    end_dt = datetime.now()
    elapsed = end_pc - start_pc

    api_processing_time = (
        job_result.get("executionTime", 0) / 1000.0 if job_result.get("executionTime") else None
    )
    delay_time = (
        job_result.get("delayTime", 0) / 1000.0 if job_result.get("delayTime") else None
    )

    print(f"[end   ]  {end_dt.isoformat()}")
    print(f"[elapsed] {elapsed:.2f} s   ({elapsed/60:.2f} min)   "
          f"HH:MM:SS={_format_elapsed(elapsed)}")
    if api_processing_time is not None:
        print(f"[api   ]  execution_time_seconds={api_processing_time:.2f}  "
              f"queue_delay_seconds={delay_time}")
    print(f"[output]  {args.output.resolve()}")

    metadata_path = args.output.with_suffix(".json")
    metadata = {
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "run_id": run_id,
        "build_phase": build_meta,
        "lipsync_phase": {
            "backend": "runpod_serverless",
            "runpod_job_id": job_id,
            "runpod_status": job_result.get("status"),
            "runpod_execution_time_seconds": api_processing_time,
            "runpod_queue_delay_seconds": delay_time,
            "start_iso": start_dt.isoformat(),
            "end_iso": end_dt.isoformat(),
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
            "elapsed_hms": _format_elapsed(elapsed),
            "r2_keys": {
                "video": video_key,
                "audio": audio_key,
                "output": output_key,
                "manifest": manifest_key if manifest_url else None,
            },
            "manifest": manifest_meta,
            "runpod_response": job_result,
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

    if not args.keep_r2_objects:
        keys_to_delete = [video_key, audio_key, output_key]
        if manifest_url:
            keys_to_delete.append(manifest_key)
        try:
            r2_delete(keys_to_delete)
            print(f"[clean ]  removed {len(keys_to_delete)} R2 objects under {run_id}/")
        except Exception as e:
            print(f"WARNING: R2 cleanup failed: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
