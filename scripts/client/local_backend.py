"""In-process backend for ``synth_random_lipsync --backend local``.

Replicates the RunPod handler's preprocess_full + lipsync logic but:
- Reads clip mp4s from the local filesystem (no R2 download).
- Persists the per-clip cache blobs on local disk (no R2 upload).
- Calls :class:`MuseTalkInference` directly in-process instead of submitting
  a RunPod job and polling for completion.

Designed for HPC deployments (MareNostrum 5 ACC, Leonardo, etc.) where:
- No outbound internet on compute nodes (everything must be pre-staged).
- No Docker, only Singularity/Apptainer.
- The cache lives under a project-shared directory so multiple SLURM jobs
  can reuse it (sub-1:1 on second run with the same clip set).

The RunPod cloud path in :func:`musetalk_runpod.handler` is untouched —
this module is invoked only when the user passes ``--backend local``.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from api.inference_service import MuseTalkInference

from scripts.client.clip_manifest import (
    DEFAULT_CACHE_DIR,
    cache_load,
    cache_save,
    compute_full_set_hash,
)


# ---------------------------------------------------------------------------
# Engine lifecycle (lazy load — heavy imports only on first --backend local
# invocation, not at module import time)
# ---------------------------------------------------------------------------


_ENGINE_SINGLETON: "MuseTalkInference | None" = None


def get_or_load_engine(use_float16: bool = True, gpu_id: int = 0) -> "MuseTalkInference":
    """Return a process-wide singleton of MuseTalkInference, loading weights
    on first call. Subsequent calls reuse the same instance — important on
    HPC where a single SLURM job may chain preprocess + lipsync invocations.
    """
    global _ENGINE_SINGLETON
    if _ENGINE_SINGLETON is not None:
        return _ENGINE_SINGLETON

    from api.inference_service import MuseTalkInference  # heavy: lazy
    import musetalk.utils.preprocessing as _prep

    eng = MuseTalkInference(use_float16=use_float16, gpu_id=gpu_id)
    eng.load_models()
    # Align the global ``device`` used by the s3fd + DWPose helpers with the
    # engine's device. The RunPod handler does this too at module load.
    _prep.device = eng.device

    _ENGINE_SINGLETON = eng
    return eng


# ---------------------------------------------------------------------------
# In-process per-clip preprocessing
# ---------------------------------------------------------------------------


def _decode_all_frames(video_path: Path) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")
    frames: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames


def preprocess_full_in_process(
    unique_clips: List[Path],
    cache_dir: Path,
    *,
    bbox_shift: int = 0,
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    left_cheek_width: int = 90,
    right_cheek_width: int = 90,
    vae_batch_size: int = 16,
    parsing_batch_size: int = 16,
    engine: "MuseTalkInference | None" = None,
) -> Dict[str, Dict[str, Any]]:
    """Mirror of ``musetalk_runpod.handler._handle_preprocess_full`` for the
    local backend.

    For every frame of every unique clip we:
    1. Detect its own face bbox via s3fd + DWPose (so head motion within a
       clip is tracked frame-by-frame).
    2. Crop + batched VAE-encode that frame's face → ``[N, 8, 32, 32]``.
    3. Crop the ``face_large`` region and batched BiSeNet-parse it. The raw
       512x512 parsing is stored — the blend mask is rebuilt at lipsync
       time using the per-frame bbox.

    Writes ``<clip>_bboxes.json``, ``<clip>_latents.pt``, ``<clip>_parsing.npz``
    under ``cache_dir`` (no R2). Returns the same per-clip metadata schema
    used by the RunPod handler, with ``*_path`` fields holding absolute
    local paths (so ``get_or_compute_full_manifest_local`` can persist them
    in the local cache JSON).
    """
    if engine is None:
        engine = get_or_load_engine()

    # Local import to avoid touching the GPU stack when the runpod path is used
    from musetalk.utils.blending import get_crop_box
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[preprocess_full_in_process] {len(unique_clips)} clips, "
        f"parsing_mode={parsing_mode}, extra_margin={extra_margin}, "
        f"vae_bs={vae_batch_size}, parsing_bs={parsing_batch_size}, "
        f"cache_dir={cache_dir}",
        flush=True,
    )

    fp = FaceParsing(
        left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width
    )
    result_clips: Dict[str, Dict[str, Any]] = {}

    for clip_path in unique_clips:
        clip_name = clip_path.name

        t0 = time.perf_counter()
        frames = _decode_all_frames(clip_path)
        H, W = frames[0].shape[:2]
        print(
            f"  {clip_name}: decoded {len(frames)} frames ({W}x{H}) in "
            f"{time.perf_counter()-t0:.2f}s",
            flush=True,
        )

        # --- Per-frame face detection ----------------------------------------
        t_fd = time.perf_counter()
        coords, _ = get_landmark_and_bbox(
            img_list=None,
            upperbondrange=bbox_shift,
            frames=frames,
        )
        per_frame_bboxes: List[Tuple[int, int, int, int]] = []
        last_valid: Tuple[int, int, int, int] | None = None
        for bbox in coords:
            if bbox == coord_placeholder:
                if last_valid is None:
                    raise RuntimeError(
                        f"face detection failed on the very first frame of {clip_name}"
                    )
                per_frame_bboxes.append(last_valid)
            else:
                b = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
                per_frame_bboxes.append(b)
                last_valid = b
        print(
            f"  {clip_name}: per-frame face detection {len(per_frame_bboxes)} "
            f"bboxes in {time.perf_counter()-t_fd:.2f}s",
            flush=True,
        )

        # --- Per-frame VAE encode (batched) ----------------------------------
        t_vae = time.perf_counter()
        crops: List[np.ndarray] = []
        for frame, bbox in zip(frames, per_frame_bboxes):
            x1, y1, x2, y2 = bbox
            y2c = min(y2 + extra_margin, frame.shape[0])
            crop = frame[y1:y2c, x1:x2]
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            crops.append(crop)

        all_latents: List[torch.Tensor] = []
        for i in range(0, len(crops), vae_batch_size):
            chunk = crops[i:i + vae_batch_size]
            batched = engine.vae.get_latents_for_unet_batch(chunk)
            all_latents.append(batched.detach().cpu())
        latents = torch.cat(all_latents, dim=0).contiguous()
        print(
            f"  {clip_name}: VAE encode {tuple(latents.shape)} in "
            f"{time.perf_counter()-t_vae:.2f}s",
            flush=True,
        )

        # --- Per-frame face parsing (batched, save 512x512 raw) --------------
        t_fp = time.perf_counter()
        face_large_list: List[Image.Image] = []
        for frame, bbox in zip(frames, per_frame_bboxes):
            x1, y1, x2, y2 = bbox
            y2c = min(y2 + extra_margin, frame.shape[0])
            face_box = (x1, y1, x2, y2c)
            crop_box, _ = get_crop_box(face_box, 1.5)
            body_pil = Image.fromarray(frame[:, :, ::-1])
            face_large_pil = body_pil.crop(crop_box)
            face_large_list.append(face_large_pil)

        parsing_arrays: List[np.ndarray] = []
        for i in range(0, len(face_large_list), parsing_batch_size):
            chunk = face_large_list[i:i + parsing_batch_size]
            parsing_results = fp.batch_call(chunk, mode=parsing_mode)
            for parsing_pil in parsing_results:
                parsing_arrays.append(np.array(parsing_pil, dtype=np.uint8))
        parsing_stack = np.stack(parsing_arrays, axis=0)  # [N, 512, 512]
        print(
            f"  {clip_name}: parsing {parsing_stack.shape} in "
            f"{time.perf_counter()-t_fp:.2f}s",
            flush=True,
        )

        # --- Persist on local disk (no R2) -----------------------------------
        t_save = time.perf_counter()
        bboxes_path = cache_dir / f"{clip_name}_bboxes.json"
        latents_path = cache_dir / f"{clip_name}_latents.pt"
        parsing_path = cache_dir / f"{clip_name}_parsing.npz"
        bboxes_path.write_text(
            json.dumps({
                "frame_shape": [int(H), int(W)],
                "bboxes": [list(b) for b in per_frame_bboxes],
            }),
            encoding="utf-8",
        )
        torch.save(latents, str(latents_path))
        np.savez_compressed(str(parsing_path), parsing=parsing_stack)
        print(
            f"  {clip_name}: wrote {latents_path.stat().st_size/1e6:.1f} MB latents "
            f"+ {parsing_path.stat().st_size/1e6:.1f} MB parsing "
            f"+ {bboxes_path.stat().st_size/1e3:.1f} KB bboxes in "
            f"{time.perf_counter()-t_save:.2f}s",
            flush=True,
        )

        result_clips[clip_name] = {
            "bboxes_path": str(bboxes_path.resolve()),
            "latents_path": str(latents_path.resolve()),
            "parsing_path": str(parsing_path.resolve()),
            "n_frames": int(latents.shape[0]),
            "frame_shape": [int(H), int(W)],
        }

        del frames, crops, face_large_list, parsing_arrays, parsing_stack
        del latents, all_latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return result_clips


# ---------------------------------------------------------------------------
# Local cache lookup / build
# ---------------------------------------------------------------------------


def get_or_compute_full_manifest_local(
    unique_clips: List[Path],
    frame_to_clip: List[str],
    frame_to_clip_idx: List[int],
    *,
    bbox_shift: int,
    extra_margin: int,
    parsing_mode: str,
    left_cheek_width: int,
    right_cheek_width: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    blobs_root: Path | None = None,
    engine: "MuseTalkInference | None" = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """End-to-end v3 manifest with **per-frame** bboxes + latents + parsing
    on local disk.

    Returns ``(per_clip_paths, meta)``:
    - ``per_clip_paths``: ``{clip_name: {bboxes_path, latents_path,
      parsing_path, n_frames, frame_shape}}``
    - ``meta`` includes ``cache_hit`` and the full set hash.

    Cache layout mirrors the cloud one (R2 prefix → local subdir):
    ``<cache_dir>/blobs/<full_set_hash>/<clip>_{bboxes.json,latents.pt,parsing.npz}``.

    Cache miss runs :func:`preprocess_full_in_process`, persisting both the
    blobs and a sibling ``<full_set_hash>.json`` manifest under
    ``cache_dir``.
    """
    cache_dir = Path(cache_dir)
    if blobs_root is None:
        blobs_root = cache_dir / "blobs"
    blobs_root = Path(blobs_root)

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
    per_clip_paths: Dict[str, Dict[str, Any]] | None = None
    cache_hit = False
    preprocess_elapsed: float | None = None

    if cached:
        full_payload = cached.get("full_payload") or {}
        if set(full_payload.keys()) == expected_names:
            # Validate that every blob actually exists on disk; otherwise
            # treat as cache miss (matches the R2-exists check the cloud
            # backend does via :func:`r2_exists`).
            ok = True
            for name, payload in full_payload.items():
                for k in ("bboxes_path", "latents_path", "parsing_path"):
                    p = payload.get(k)
                    if not p or not Path(p).is_file():
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                per_clip_paths = full_payload
                cache_hit = True
            else:
                print(
                    "[manifest_local] local cache JSON present but some "
                    "blobs missing on disk — recomputing",
                    flush=True,
                )

    if per_clip_paths is None:
        clip_blob_dir = blobs_root / full_set_hash
        clip_blob_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        per_clip_paths = preprocess_full_in_process(
            unique_clips,
            clip_blob_dir,
            bbox_shift=bbox_shift,
            extra_margin=extra_margin,
            parsing_mode=parsing_mode,
            left_cheek_width=left_cheek_width,
            right_cheek_width=right_cheek_width,
            engine=engine,
        )
        preprocess_elapsed = time.perf_counter() - t0

        cache_save(
            full_set_hash,
            {
                "backend": "local",
                "full_set_hash": full_set_hash,
                "bbox_shift": bbox_shift,
                "extra_margin": extra_margin,
                "parsing_mode": parsing_mode,
                "left_cheek_width": left_cheek_width,
                "right_cheek_width": right_cheek_width,
                "full_payload": per_clip_paths,
                "preprocess_full_elapsed_s": preprocess_elapsed,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            cache_dir,
        )

    meta = {
        "full_set_hash": full_set_hash,
        "cache_hit": cache_hit,
        "preprocess_full_elapsed_s": preprocess_elapsed,
        "unique_clip_count": len(unique_clips),
        "clip_n_frames": {
            name: int(payload.get("n_frames", 0))
            for name, payload in per_clip_paths.items()
        },
        "backend": "local",
    }
    return per_clip_paths, meta


# ---------------------------------------------------------------------------
# Cache expansion (cache → per-frame lists for engine.generate)
# ---------------------------------------------------------------------------


def expand_cached_per_clip_local(
    per_clip_paths: Dict[str, Dict[str, Any]],
    frame_to_clip: List[str],
    frame_to_clip_idx: List[int],
    *,
    extra_margin: int,
    engine: "MuseTalkInference | None" = None,
) -> Tuple[List[List[int]], List[torch.Tensor], List[np.ndarray]]:
    """Mirror of ``musetalk_runpod.handler._expand_cached_per_clip`` but
    loading blobs from local disk paths instead of R2.

    Returns ``(precomputed_bboxes, precomputed_latents, precomputed_masks)``:
    - per-assembled-frame [x1,y1,x2,y2]
    - per-assembled-frame ``[1, 8, 32, 32]`` latent tensors (on engine.device)
    - per-assembled-frame blend masks rebuilt from the cached 512x512
      parsing using the per-frame bbox (so the mask tracks head motion)
    """
    if engine is None:
        engine = get_or_load_engine()

    from musetalk.utils.blending import _build_blend_mask_from_parsing, get_crop_box

    if len(frame_to_clip) != len(frame_to_clip_idx):
        raise RuntimeError(
            f"frame_to_clip ({len(frame_to_clip)}) and frame_to_clip_idx "
            f"({len(frame_to_clip_idx)}) length mismatch"
        )

    clip_bboxes: Dict[str, List[List[int]]] = {}
    clip_latents: Dict[str, torch.Tensor] = {}
    clip_parsing: Dict[str, np.ndarray] = {}
    clip_frame_shape: Dict[str, Tuple[int, int]] = {}

    t0 = time.perf_counter()
    for clip_name, payload in per_clip_paths.items():
        # --- bboxes ---
        bboxes_path = Path(payload["bboxes_path"])
        data = json.loads(bboxes_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            clip_bboxes[clip_name] = data["bboxes"]
            fs = data.get("frame_shape")
            if isinstance(fs, list) and len(fs) == 2:
                clip_frame_shape[clip_name] = (int(fs[0]), int(fs[1]))
        else:
            clip_bboxes[clip_name] = data

        # --- latents → engine device ---
        latents_path = Path(payload["latents_path"])
        clip_latents[clip_name] = torch.load(
            str(latents_path), map_location=engine.device
        )

        # --- parsing ---
        parsing_path = Path(payload["parsing_path"])
        clip_parsing[clip_name] = np.load(str(parsing_path))["parsing"]

    print(
        f"[expand_local] loaded {len(per_clip_paths)} clip blobs in "
        f"{time.perf_counter()-t0:.2f}s",
        flush=True,
    )

    precomputed_bboxes: List[List[int]] = []
    precomputed_latents: List[torch.Tensor] = []
    precomputed_masks: List[np.ndarray] = []

    t_mask = time.perf_counter()
    for clip_name, idx in zip(frame_to_clip, frame_to_clip_idx):
        if clip_name not in clip_latents or clip_name not in clip_parsing \
                or clip_name not in clip_bboxes:
            raise RuntimeError(f"cached blob missing for clip {clip_name!r}")
        n_lat = clip_latents[clip_name].shape[0]
        n_par = clip_parsing[clip_name].shape[0]
        n_bb = len(clip_bboxes[clip_name])
        lat_idx = min(int(idx), n_lat - 1)
        par_idx = min(int(idx), n_par - 1)
        bb_idx = min(int(idx), n_bb - 1)

        bbox = clip_bboxes[clip_name][bb_idx]
        x1, y1, x2, y2 = bbox
        H, _W = clip_frame_shape.get(clip_name, (10_000, 10_000))
        y2c = min(y2 + extra_margin, H)
        face_box = (x1, y1, x2, y2c)
        crop_box, _ = get_crop_box(face_box, 1.5)
        ori_shape = (crop_box[2] - crop_box[0], crop_box[3] - crop_box[1])

        parsing_pil = Image.fromarray(clip_parsing[clip_name][par_idx])
        parsing_resized = parsing_pil.resize(ori_shape)
        blend_mask = _build_blend_mask_from_parsing(
            parsing_resized, face_box, crop_box, ori_shape
        )

        precomputed_bboxes.append([x1, y1, x2, y2])
        precomputed_latents.append(clip_latents[clip_name][lat_idx:lat_idx + 1])
        precomputed_masks.append(blend_mask)

    print(
        f"[expand_local] built {len(precomputed_masks)} per-frame blend masks "
        f"in {time.perf_counter()-t_mask:.2f}s",
        flush=True,
    )
    return precomputed_bboxes, precomputed_latents, precomputed_masks


# ---------------------------------------------------------------------------
# Top-level orchestrator (called from synth_random_lipsync main on --backend local)
# ---------------------------------------------------------------------------


def run_lipsync_local(
    *,
    args,                       # parsed argparse namespace from synth_random_lipsync
    build_meta: Dict[str, Any],  # the dict returned by build_random_video
    intermediate_path: Path,    # the assembled video_completo.mp4
    run_id: str,
) -> Dict[str, Any]:
    """End-to-end local backend driver: load engine, build/lookup local
    manifest, expand cache, call engine.generate, copy output. Returns a
    dict shape-compatible with the RunPod ``poll_job`` payload so the
    metadata block at the bottom of ``synth_random_lipsync.main`` can be
    reused unchanged.
    """
    t_total = time.perf_counter()

    engine = get_or_load_engine()

    manifest_meta: Dict[str, Any] = {}
    per_clip_paths: Dict[str, Dict[str, Any]] | None = None

    if not getattr(args, "skip_manifest", False):
        unique_clip_paths = [Path(p) for p in build_meta["unique_clips_used"]]
        t_manifest = time.perf_counter()
        per_clip_paths, manifest_meta = get_or_compute_full_manifest_local(
            unique_clips=unique_clip_paths,
            frame_to_clip=build_meta["frame_to_clip"],
            frame_to_clip_idx=build_meta["frame_to_clip_idx"],
            bbox_shift=args.bbox_shift,
            extra_margin=args.extra_margin,
            parsing_mode=args.parsing_mode,
            left_cheek_width=args.left_cheek_width,
            right_cheek_width=args.right_cheek_width,
            cache_dir=Path(args.manifest_cache_dir),
            engine=engine,
        )
        origin = "cache" if manifest_meta.get("cache_hit") else "fresh"
        print(
            f"[local-backend] v3 manifest ready ({origin}) in "
            f"{time.perf_counter()-t_manifest:.2f}s — "
            f"{len(per_clip_paths)} clip blob bundles; "
            f"full_set_hash={manifest_meta.get('full_set_hash')}",
            flush=True,
        )

    # ------- expand cache to per-frame lists -------
    precomputed_bboxes = None
    precomputed_latents = None
    precomputed_masks = None
    if per_clip_paths:
        precomputed_bboxes, precomputed_latents, precomputed_masks = (
            expand_cached_per_clip_local(
                per_clip_paths,
                [Path(p).name for p in build_meta["frame_to_clip"]],
                build_meta["frame_to_clip_idx"],
                extra_margin=args.extra_margin,
                engine=engine,
            )
        )

    # ------- call engine.generate directly -------
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_dir = args.intermediate_dir.resolve()
    output_name = f"output_{run_id}"

    t_engine = time.perf_counter()
    output_local = engine.generate(
        audio_path=str(Path(args.audio).resolve()),
        video_path=str(Path(intermediate_path).resolve()),
        enhance=bool(args.enhance),
        bbox_shift=int(args.bbox_shift),
        extra_margin=int(args.extra_margin),
        parsing_mode=str(args.parsing_mode),
        left_cheek_width=int(args.left_cheek_width),
        right_cheek_width=int(args.right_cheek_width),
        fps=int(args.fps),
        batch_size=int(args.batch_size),
        output_name=output_name,
        result_dir=str(result_dir),
        gfpgan_weight=float(args.gfpgan_weight),
        precomputed_bboxes=precomputed_bboxes,
        precomputed_latents=precomputed_latents,
        precomputed_masks=precomputed_masks,
        use_nvenc=bool(args.use_nvenc),
    )
    engine_elapsed = time.perf_counter() - t_engine
    print(
        f"[local-backend] engine.generate completed in {engine_elapsed:.1f}s "
        f"-> {output_local}",
        flush=True,
    )

    # ------- copy to user-requested output path -------
    final_path = Path(args.output).resolve()
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if Path(output_local).resolve() != final_path:
        shutil.copy2(output_local, final_path)

    total_elapsed = time.perf_counter() - t_total

    # Shape-compatible with RunPod poll_job response so the metadata block
    # in synth_random_lipsync.main can be reused unchanged.
    return {
        "status": "COMPLETED",
        "executionTime": int(engine_elapsed * 1000),     # ms, like RunPod
        "delayTime": 0,
        "output": {
            "status": "success",
            "output_path": str(final_path),
            "manifest_meta": manifest_meta,
            "total_elapsed_s": total_elapsed,
        },
    }
