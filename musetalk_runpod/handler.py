"""RunPod Serverless handler for MuseTalk lipsync.

Module-scope initialization is intentional: each worker loads the model
weights once and serves N jobs over its lifetime.

The handler dispatches on ``job["input"]["action"]``:
- ``lipsync`` (default): full inference pipeline. Accepts optional
  ``manifest_url`` (per-frame bboxes), ``latents_urls`` (per-clip VAE
  latents) and ``masks_urls`` (per-clip blend masks) plus ``frame_to_clip``
  and ``frame_to_clip_idx`` so cached preprocessing can be reused.
- ``preprocess_clips``: lightweight, single probe PNG per clip — returns
  the bbox.
- ``preprocess_full``: decodes the FULL mp4 per clip, runs batched VAE
  encode + batched FaceParsing, uploads ``<clip>_latents.pt`` and
  ``<clip>_masks.npz`` to R2, returns presigned URLs and frame counts.
"""

from __future__ import annotations

import json
import os
import sys
import time as _time
import traceback
import uuid
from pathlib import Path

REPO_ROOT = Path(os.environ.get("MUSETALK_REPO_ROOT", "/app"))
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from api.inference_service import MuseTalkInference  # noqa: E402
from musetalk.utils.blending import (  # noqa: E402
    _build_blend_mask_from_parsing,
    get_crop_box,
)
from musetalk.utils.face_parsing import FaceParsing  # noqa: E402
from musetalk.utils.preprocessing import (  # noqa: E402
    coord_placeholder,
    get_landmark_and_bbox,
)
from musetalk_runpod.storage import download_to_local, upload_from_local  # noqa: E402

ENGINE = MuseTalkInference(use_float16=True, gpu_id=0)
ENGINE.load_models()

import musetalk.utils.preprocessing as _prep  # noqa: E402

_prep.device = ENGINE.device

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _log(msg: str) -> None:
    print(f"[handler] {msg}", flush=True)


def _required(inp: dict, key: str):
    if key not in inp:
        raise KeyError(f"missing required input field '{key}'")
    return inp[key]


def _handle_preprocess(inp: dict, work_dir: Path) -> dict:
    probe_frames = _required(inp, "probe_frames")
    bbox_shift = int(inp.get("bbox_shift", 0))
    _log(f"preprocess_clips: {len(probe_frames)} clips, bbox_shift={bbox_shift}")

    bboxes: dict[str, list[int]] = {}
    failures: list[str] = []
    for clip_name, url in probe_frames.items():
        png_path = work_dir / f"probe_{clip_name}.png"
        download_to_local(url, png_path)
        coords, _frames = get_landmark_and_bbox([str(png_path)], bbox_shift)
        if not coords or coords[0] == coord_placeholder:
            failures.append(clip_name)
            continue
        x1, y1, x2, y2 = coords[0]
        bboxes[clip_name] = [int(x1), int(y1), int(x2), int(y2)]
        _log(f"  {clip_name}: bbox={bboxes[clip_name]}")

    if failures:
        raise RuntimeError(
            f"face detection failed on probe frames: {failures}. "
            f"Verify the clips contain a visible frontal face on frame 0."
        )
    return {"status": "success", "bboxes": bboxes}


def _decode_all_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {video_path}")
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames


def _handle_preprocess_full(inp: dict, work_dir: Path) -> dict:
    """Decode every frame of each unique clip and build the *per-frame* cache.

    For every frame we:
    1. Detect its own face bbox (so head motion within the clip is tracked).
    2. Crop + VAE-encode that frame's face.
    3. Crop the surrounding ``face_large`` region and run BiSeNet face parsing.
       The 512x512 parsing mask is stored raw — the blend mask is built at
       lipsync time using the per-frame bbox so it stays aligned with the
       moving head.

    Outputs per clip on R2: ``<clip>_bboxes.json``, ``<clip>_latents.pt``,
    ``<clip>_parsing.npz``. Old per-clip blend masks (``<clip>_masks.npz``)
    are no longer produced — the v3 cache key invalidates the old format.
    """
    clips = _required(inp, "clips")
    if not isinstance(clips, list) or not clips:
        raise ValueError("'clips' must be a non-empty list of {name, url}")

    bbox_shift = int(inp.get("bbox_shift", 0))
    extra_margin = int(inp.get("extra_margin", 10))
    parsing_mode = str(inp.get("parsing_mode", "jaw"))
    left_cw = int(inp.get("left_cheek_width", 90))
    right_cw = int(inp.get("right_cheek_width", 90))
    vae_batch_size = int(inp.get("vae_batch_size", 16))
    parsing_batch_size = int(inp.get("parsing_batch_size", 16))
    r2_prefix = str(inp.get("r2_prefix") or f"cache/{uuid.uuid4().hex}").rstrip("/")
    presign_expires = int(inp.get("presign_expires", 86400))

    _log(
        f"preprocess_full(v3 per-frame): {len(clips)} clips, "
        f"parsing_mode={parsing_mode}, extra_margin={extra_margin}, "
        f"vae_bs={vae_batch_size}, parsing_bs={parsing_batch_size}, "
        f"r2_prefix={r2_prefix}"
    )

    fp = FaceParsing(left_cheek_width=left_cw, right_cheek_width=right_cw)
    result_clips: dict[str, dict] = {}

    for clip in clips:
        clip_name = clip["name"]
        clip_url = clip["url"]

        t0 = _time.perf_counter()
        local_mp4 = work_dir / clip_name
        download_to_local(clip_url, local_mp4)
        frames = _decode_all_frames(local_mp4)
        H, W = frames[0].shape[:2]
        _log(
            f"  {clip_name}: decoded {len(frames)} frames ({W}x{H}) in "
            f"{_time.perf_counter()-t0:.2f}s"
        )

        # --- Per-frame face detection ---------------------------------------
        t_fd = _time.perf_counter()
        coords, _ = get_landmark_and_bbox(
            img_list=None,
            upperbondrange=bbox_shift,
            frames=frames,
        )
        # If a frame has no detection, reuse the most recent valid bbox so
        # downstream arrays stay aligned. The first frame must have a face.
        per_frame_bboxes: list[tuple[int, int, int, int]] = []
        last_valid: tuple[int, int, int, int] | None = None
        for i, bbox in enumerate(coords):
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
        _log(
            f"  {clip_name}: per-frame face detection {len(per_frame_bboxes)} "
            f"bboxes in {_time.perf_counter()-t_fd:.2f}s"
        )

        # --- Per-frame VAE encode (batched) ---------------------------------
        t_vae = _time.perf_counter()
        crops: list[np.ndarray] = []
        for frame, bbox in zip(frames, per_frame_bboxes):
            x1, y1, x2, y2 = bbox
            y2c = min(y2 + extra_margin, frame.shape[0])
            crop = frame[y1:y2c, x1:x2]
            crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            crops.append(crop)

        all_latents: list[torch.Tensor] = []
        for i in range(0, len(crops), vae_batch_size):
            chunk = crops[i:i + vae_batch_size]
            batched = ENGINE.vae.get_latents_for_unet_batch(chunk)
            all_latents.append(batched.detach().cpu())
        latents = torch.cat(all_latents, dim=0).contiguous()
        _log(
            f"  {clip_name}: VAE encode {latents.shape} in "
            f"{_time.perf_counter()-t_vae:.2f}s"
        )

        # --- Per-frame face parsing (batched, save 512x512 raw) -------------
        t_fp = _time.perf_counter()
        face_large_list: list[Image.Image] = []
        for frame, bbox in zip(frames, per_frame_bboxes):
            x1, y1, x2, y2 = bbox
            y2c = min(y2 + extra_margin, frame.shape[0])
            face_box = (x1, y1, x2, y2c)
            crop_box, _ = get_crop_box(face_box, 1.5)
            body_pil = Image.fromarray(frame[:, :, ::-1])
            face_large_pil = body_pil.crop(crop_box)
            face_large_list.append(face_large_pil)

        parsing_arrays: list[np.ndarray] = []
        for i in range(0, len(face_large_list), parsing_batch_size):
            chunk = face_large_list[i:i + parsing_batch_size]
            parsing_results = fp.batch_call(chunk, mode=parsing_mode)
            for parsing_pil in parsing_results:
                parsing_arrays.append(np.array(parsing_pil, dtype=np.uint8))
        parsing_stack = np.stack(parsing_arrays, axis=0)  # [N, 512, 512]
        _log(
            f"  {clip_name}: parsing {parsing_stack.shape} in "
            f"{_time.perf_counter()-t_fp:.2f}s"
        )

        # --- Persist + upload (bboxes.json, latents.pt, parsing.npz) --------
        t_up = _time.perf_counter()
        bboxes_local = work_dir / f"{clip_name}_bboxes.json"
        latents_local = work_dir / f"{clip_name}_latents.pt"
        parsing_local = work_dir / f"{clip_name}_parsing.npz"
        bboxes_local.write_text(
            json.dumps({
                "frame_shape": [int(H), int(W)],
                "bboxes": [list(b) for b in per_frame_bboxes],
            }),
            encoding="utf-8",
        )
        torch.save(latents, str(latents_local))
        np.savez_compressed(str(parsing_local), parsing=parsing_stack)

        bboxes_key = f"{r2_prefix}/{clip_name}_bboxes.json"
        latents_key = f"{r2_prefix}/{clip_name}_latents.pt"
        parsing_key = f"{r2_prefix}/{clip_name}_parsing.npz"
        bboxes_url = upload_from_local(bboxes_local, bboxes_key, presign_expires=presign_expires)
        latents_url = upload_from_local(latents_local, latents_key, presign_expires=presign_expires)
        parsing_url = upload_from_local(parsing_local, parsing_key, presign_expires=presign_expires)
        _log(
            f"  {clip_name}: upload {latents_local.stat().st_size/1e6:.1f} MB latents "
            f"+ {parsing_local.stat().st_size/1e6:.1f} MB parsing "
            f"+ {bboxes_local.stat().st_size/1e3:.1f} KB bboxes in "
            f"{_time.perf_counter()-t_up:.2f}s"
        )

        result_clips[clip_name] = {
            "bboxes_url": bboxes_url,
            "bboxes_key": bboxes_key,
            "latents_url": latents_url,
            "latents_key": latents_key,
            "parsing_url": parsing_url,
            "parsing_key": parsing_key,
            "n_frames": int(latents.shape[0]),
            "frame_shape": [int(H), int(W)],
        }

        del frames, crops, face_large_list, parsing_arrays, parsing_stack, latents, all_latents
        torch.cuda.empty_cache()

    return {"status": "success", "clips": result_clips, "r2_prefix": r2_prefix}


def _expand_cached_per_clip(
    work_dir: Path,
    bboxes_urls: dict,
    latents_urls: dict,
    parsing_urls: dict,
    frame_to_clip: list,
    frame_to_clip_idx: list,
    extra_margin: int,
) -> tuple[list, list, list]:
    """Download per-clip per-frame cached blobs and expand them into parallel
    per-assembled-frame lists.

    Returns ``(precomputed_bboxes, precomputed_latents, precomputed_masks)``:
    - ``precomputed_bboxes``: list of [x1,y1,x2,y2] per assembled frame
    - ``precomputed_latents``: list of [1,8,32,32] tensors per assembled frame
    - ``precomputed_masks``: list of blend-mask ndarrays per assembled frame,
      built on the fly from the 512x512 parsing using the per-frame bbox so
      the mask tracks head motion correctly.
    """
    if len(frame_to_clip) != len(frame_to_clip_idx):
        raise RuntimeError(
            f"frame_to_clip ({len(frame_to_clip)}) and frame_to_clip_idx "
            f"({len(frame_to_clip_idx)}) length mismatch"
        )

    clip_bboxes: dict[str, list[list[int]]] = {}
    clip_latents: dict[str, torch.Tensor] = {}
    clip_parsing: dict[str, np.ndarray] = {}
    clip_frame_shape: dict[str, tuple[int, int]] = {}

    t0 = _time.perf_counter()
    for clip_name, url in bboxes_urls.items():
        local = work_dir / f"{clip_name}_bboxes.json"
        download_to_local(url, local)
        data = json.loads(local.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # v3 format: {"frame_shape": [H, W], "bboxes": [...]}
            clip_bboxes[clip_name] = data["bboxes"]
            fs = data.get("frame_shape")
            if isinstance(fs, list) and len(fs) == 2:
                clip_frame_shape[clip_name] = (int(fs[0]), int(fs[1]))
        else:
            # legacy flat list of bboxes — frame_shape unknown
            clip_bboxes[clip_name] = data
    for clip_name, url in latents_urls.items():
        local = work_dir / f"{clip_name}_latents.pt"
        download_to_local(url, local)
        clip_latents[clip_name] = torch.load(str(local), map_location=ENGINE.device)
    for clip_name, url in parsing_urls.items():
        local = work_dir / f"{clip_name}_parsing.npz"
        download_to_local(url, local)
        clip_parsing[clip_name] = np.load(str(local))["parsing"]
    _log(
        f"downloaded per-clip blobs: bboxes={len(bboxes_urls)}, "
        f"latents={len(latents_urls)}, parsing={len(parsing_urls)} in "
        f"{_time.perf_counter()-t0:.2f}s"
    )

    precomputed_bboxes: list[list[int]] = []
    precomputed_latents: list[torch.Tensor] = []
    precomputed_masks: list[np.ndarray] = []

    t_mask = _time.perf_counter()
    for clip_name, idx in zip(frame_to_clip, frame_to_clip_idx):
        if clip_name not in clip_latents or clip_name not in clip_parsing \
                or clip_name not in clip_bboxes:
            raise RuntimeError(f"cached blob missing for clip {clip_name!r}")
        n_lat = clip_latents[clip_name].shape[0]
        n_par = clip_parsing[clip_name].shape[0]
        n_bb = len(clip_bboxes[clip_name])
        # Defensive clamp: client-side _build_frame_to_clip_map now produces
        # idx already in [0, n_native-1] thanks to source/target fps scaling.
        # We clamp instead of wrapping (%) so older clients that send an
        # out-of-range idx degrade gracefully — duplicating the last cached
        # frame instead of jumping back to frame 0 mid-clip.
        lat_idx = min(int(idx), n_lat - 1)
        par_idx = min(int(idx), n_par - 1)
        bb_idx = min(int(idx), n_bb - 1)

        bbox = clip_bboxes[clip_name][bb_idx]
        x1, y1, x2, y2 = bbox

        # Use real per-clip frame shape from bboxes.json so the y2c clamp
        # matches what the engine does on the same frame; otherwise crop_box
        # and ori_shape can drift by extra_margin pixels at the bottom edge.
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

    _log(
        f"built {len(precomputed_masks)} per-frame blend masks (per-frame bbox) "
        f"in {_time.perf_counter()-t_mask:.2f}s"
    )
    return precomputed_bboxes, precomputed_latents, precomputed_masks


def _handle_lipsync(inp: dict, run_id: str, work_dir: Path) -> dict:
    video_url = _required(inp, "video_url")
    audio_url = _required(inp, "audio_url")
    output_object_key = _required(inp, "output_object_key")
    manifest_url = inp.get("manifest_url")
    bboxes_urls = inp.get("bboxes_urls")
    latents_urls = inp.get("latents_urls")
    parsing_urls = inp.get("parsing_urls")
    frame_to_clip = inp.get("frame_to_clip")
    frame_to_clip_idx = inp.get("frame_to_clip_idx")
    have_cache = bool(
        bboxes_urls and latents_urls and parsing_urls
        and frame_to_clip and frame_to_clip_idx
    )
    _log(
        f"input parsed: output_object_key={output_object_key} "
        f"manifest={'yes' if manifest_url else 'no'} "
        f"cache={'yes' if have_cache else 'no'}"
    )

    t0 = _time.perf_counter()
    _log("downloading video from R2 ...")
    video_local = download_to_local(video_url, work_dir / "video.mp4")
    v_size = video_local.stat().st_size
    _log(f"video downloaded: {v_size/1e6:.1f} MB in {_time.perf_counter()-t0:.1f}s")

    t1 = _time.perf_counter()
    _log("downloading audio from R2 ...")
    audio_local = download_to_local(audio_url, work_dir / "audio.wav")
    a_size = audio_local.stat().st_size
    _log(f"audio downloaded: {a_size/1e6:.1f} MB in {_time.perf_counter()-t1:.1f}s")

    precomputed_bboxes = None
    precomputed_latents = None
    precomputed_masks = None

    if have_cache:
        precomputed_bboxes, precomputed_latents, precomputed_masks = (
            _expand_cached_per_clip(
                work_dir,
                bboxes_urls,
                latents_urls,
                parsing_urls,
                frame_to_clip,
                frame_to_clip_idx,
                extra_margin=int(inp.get("extra_margin", 10)),
            )
        )
        _log(
            f"expanded cache: {len(precomputed_bboxes)} bboxes + "
            f"{len(precomputed_latents)} latents + {len(precomputed_masks)} masks"
        )
    elif manifest_url:
        # Legacy path: only per-frame bboxes available, no latents/masks cache.
        tm = _time.perf_counter()
        _log("downloading bbox manifest from R2 (legacy path, no D/F cache) ...")
        manifest_local = download_to_local(manifest_url, work_dir / "manifest.json")
        try:
            manifest = json.loads(manifest_local.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"failed to parse manifest JSON: {e}")
        precomputed_bboxes = manifest if isinstance(manifest, list) else manifest.get("bboxes")
        if not isinstance(precomputed_bboxes, list):
            raise RuntimeError(
                "manifest must be a JSON list of [x1,y1,x2,y2] or {bboxes: [...]}"
            )
        _log(f"manifest loaded: {len(precomputed_bboxes)} frame bboxes "
             f"in {_time.perf_counter()-tm:.1f}s")

    t2 = _time.perf_counter()
    _log(f"calling ENGINE.generate ... (batch_size={inp.get('batch_size', 16)}, "
         f"enhance={inp.get('enhance', False)}, fps={inp.get('fps', 25)}, "
         f"precomputed_bboxes={'yes' if precomputed_bboxes else 'no'}, "
         f"precomputed_latents={'yes' if precomputed_latents else 'no'}, "
         f"precomputed_masks={'yes' if precomputed_masks else 'no'})")
    output_local = ENGINE.generate(
        audio_path=str(audio_local),
        video_path=str(video_local),
        enhance=bool(inp.get("enhance", False)),
        bbox_shift=int(inp.get("bbox_shift", 0)),
        extra_margin=int(inp.get("extra_margin", 10)),
        parsing_mode=str(inp.get("parsing_mode", "jaw")),
        left_cheek_width=int(inp.get("left_cheek_width", 90)),
        right_cheek_width=int(inp.get("right_cheek_width", 90)),
        fps=int(inp.get("fps", 25)),
        batch_size=int(inp.get("batch_size", 16)),
        output_name=f"output_{run_id}",
        result_dir=str(RESULTS_DIR),
        gfpgan_weight=float(inp.get("gfpgan_weight", 0.5)),
        precomputed_bboxes=precomputed_bboxes,
        precomputed_latents=precomputed_latents,
        precomputed_masks=precomputed_masks,
        use_nvenc=bool(inp.get("use_nvenc", True)),
        vae_batch_size=int(inp.get("vae_batch_size", 16)),
        parsing_batch_size=int(inp.get("parsing_batch_size", 16)),
    )
    _log(f"engine.generate finished in {_time.perf_counter()-t2:.1f}s -> {output_local}")

    t3 = _time.perf_counter()
    _log(f"uploading output to R2 key={output_object_key} ...")
    output_url = upload_from_local(Path(output_local), output_object_key)
    _log(f"upload done in {_time.perf_counter()-t3:.1f}s")

    try:
        Path(output_local).unlink(missing_ok=True)
    except OSError:
        pass

    return {
        "status": "success",
        "output_url": output_url,
        "output_object_key": output_object_key,
    }


def handler(job: dict) -> dict:
    work_dir: Path | None = None
    try:
        inp = job.get("input") or {}
        run_id = job.get("id") or str(uuid.uuid4())
        work_dir = Path(f"/tmp/musetalk_{run_id}")
        work_dir.mkdir(parents=True, exist_ok=True)

        action = inp.get("action", "lipsync")
        _log(f"handler entered: action={action} job_id={run_id} work_dir={work_dir}")

        if action == "preprocess_clips":
            return _handle_preprocess(inp, work_dir)
        if action == "preprocess_full":
            return _handle_preprocess_full(inp, work_dir)
        if action == "lipsync":
            return _handle_lipsync(inp, run_id, work_dir)
        raise ValueError(f"unknown action: {action!r}")

    except Exception as e:
        _log(f"FATAL: {type(e).__name__}: {e}")
        _log(traceback.format_exc())
        return {
            "status": "error",
            "detail": str(e),
            "trace": traceback.format_exc(),
        }
    finally:
        if work_dir is not None and work_dir.is_dir():
            try:
                for p in work_dir.iterdir():
                    p.unlink(missing_ok=True)
                work_dir.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
