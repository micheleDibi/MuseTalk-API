import os
import shutil
import subprocess
import time
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import WhisperModel

from musetalk.utils.blending import get_image, _build_blend_mask_from_parsing, get_crop_box
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_file_type, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder


def _write_video_pipe(
    frames: List[np.ndarray],
    output_path: str,
    width: int,
    height: int,
    fps: int,
    codec: str = "h264_nvenc",
) -> None:
    """Pipe uint8 BGR frames through ffmpeg's stdin into ``output_path``.

    Avoids writing PNG intermediates to disk. Raises ``CalledProcessError`` on
    non-zero ffmpeg exit so the caller can fall back to libx264.
    """
    if codec == "h264_nvenc":
        encoder_args = [
            "-c:v", "h264_nvenc",
            "-preset", "p5",
            "-rc", "vbr",
            "-cq", "18",
            "-b:v", "3M",
            "-maxrate", "5M",
            "-bufsize", "10M",
            "-pix_fmt", "yuv420p",
        ]
    else:
        encoder_args = [
            "-c:v", codec,
            "-preset", "slow",
            "-crf", "16",
            "-pix_fmt", "yuv420p",
        ]

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        *encoder_args,
        output_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except BrokenPipeError:
        pass
    ret = proc.wait()
    if ret != 0:
        raise subprocess.CalledProcessError(ret, cmd)


class MuseTalkInference:
    def __init__(self, use_float16: bool = True, gpu_id: int = 0):
        self.use_float16 = use_float16
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        self.models_loaded = False
        self.gfpgan_restorer = None

        self.vae = None
        self.unet = None
        self.pe = None
        self.timesteps = None
        self.weight_dtype = torch.float32
        self.audio_processor = None
        self.whisper = None

    def load_models(self) -> None:
        if self.models_loaded:
            return

        print(f"Loading models on device: {self.device}")

        self.vae, self.unet, self.pe = load_all_model(
            unet_model_path="./models/musetalkV15/unet.pth",
            vae_type="sd-vae",
            unet_config="./models/musetalkV15/musetalk.json",
            device=self.device,
        )

        self.timesteps = torch.tensor([0], device=self.device)

        if self.use_float16:
            self.pe = self.pe.half()
            self.vae.vae = self.vae.vae.half()
            self.unet.model = self.unet.model.half()
            self.weight_dtype = torch.float16
        else:
            self.weight_dtype = torch.float32

        self.pe = self.pe.to(self.device)
        self.vae.vae = self.vae.vae.to(self.device)
        self.unet.model = self.unet.model.to(self.device)

        self.audio_processor = AudioProcessor(feature_extractor_path="./models/whisper")
        self.whisper = WhisperModel.from_pretrained("./models/whisper")
        self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
        self.whisper.requires_grad_(False)

        self.models_loaded = True
        print("Models loaded successfully!")

    def _load_gfpgan(self) -> None:
        if self.gfpgan_restorer is not None:
            return

        from gfpgan import GFPGANer

        print("Loading GFPGAN model...")
        self.gfpgan_restorer = GFPGANer(
            model_path="https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
            upscale=1,
            arch="clean",
            channel_multiplier=2,
        )
        print("GFPGAN loaded!")

    def _enhance_face_aligned(self, face_crop: np.ndarray, weight: float = 0.5) -> np.ndarray:
        """
        Enhance a pre-cropped face using GFPGAN with has_aligned=True.

        This skips face detection entirely since MuseTalk already extracted the face.
        GFPGAN expects 512x512 input, so we resize, enhance, then resize back.

        Args:
            face_crop: Face crop from MuseTalk (typically 256x256)
            weight: Blending weight (0=original, 1=fully enhanced)

        Returns:
            Enhanced face crop at original resolution
        """
        if self.gfpgan_restorer is None:
            return face_crop

        original_size = (face_crop.shape[1], face_crop.shape[0])  # (w, h)

        # GFPGAN expects 512x512 for optimal quality
        face_512 = cv2.resize(face_crop, (512, 512), interpolation=cv2.INTER_LANCZOS4)

        try:
            # has_aligned=True skips face detection - HUGE speedup!
            # paste_back=False since we're handling the blending ourselves
            _, restored_faces, _ = self.gfpgan_restorer.enhance(
                face_512,
                has_aligned=True,
                only_center_face=False,
                paste_back=False,
                weight=weight,
            )

            if restored_faces and len(restored_faces) > 0:
                enhanced_512 = restored_faces[0]
                # Resize back to original face crop size
                enhanced_crop = cv2.resize(
                    enhanced_512, original_size, interpolation=cv2.INTER_LANCZOS4
                )
                return enhanced_crop
        except Exception as e:
            print(f"GFPGAN enhancement failed: {e}")

        return face_crop

    @torch.no_grad()
    def generate(
        self,
        audio_path: str,
        video_path: str,
        enhance: bool = False,
        bbox_shift: int = 0,
        extra_margin: int = 10,
        parsing_mode: str = "jaw",
        left_cheek_width: int = 90,
        right_cheek_width: int = 90,
        fps: int = 25,
        batch_size: int = 8,
        output_name: Optional[str] = None,
        result_dir: str = "./results",
        gfpgan_weight: float = 0.5,
        precomputed_bboxes: Optional[list] = None,
        precomputed_latents: Optional[List[torch.Tensor]] = None,
        precomputed_masks: Optional[List[np.ndarray]] = None,
        use_nvenc: bool = True,
        vae_batch_size: int = 16,
        parsing_batch_size: int = 16,
    ) -> str:
        if not self.models_loaded:
            self.load_models()

        os.makedirs(result_dir, exist_ok=True)

        input_basename = os.path.basename(video_path).split(".")[0]
        audio_basename = os.path.basename(audio_path).split(".")[0]

        if output_name:
            output_name = (
                os.path.splitext(output_name)[0] if output_name.endswith(".mp4") else output_name
            )
            output_vid_name = os.path.join(result_dir, f"{output_name}.mp4")
        else:
            output_vid_name = os.path.join(result_dir, f"{input_basename}_{audio_basename}.mp4")

        temp_dir = os.path.join(result_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # ---- Phase A: frame extraction (in-memory, no PNG to disk) -----------
        t0 = time.perf_counter()
        file_type = get_file_type(video_path)
        if file_type == "video":
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"cv2 could not open video: {video_path}")
            in_memory_frames: List[np.ndarray] = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                in_memory_frames.append(frame)
            detected_fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if detected_fps and detected_fps > 0:
                fps = int(round(detected_fps))
            if not in_memory_frames:
                raise RuntimeError(f"no frames decoded from {video_path}")
        elif file_type == "image":
            img = cv2.imread(video_path)
            if img is None:
                raise RuntimeError(f"cv2 could not read image: {video_path}")
            in_memory_frames = [img]
        else:
            raise ValueError(f"{video_path} should be a video file or an image file")
        print(f"[phase A] frame extract: {len(in_memory_frames)} frames in {time.perf_counter() - t0:.2f}s")

        # ---- Phase B: whisper features ---------------------------------------
        t0 = time.perf_counter()
        whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(audio_path)
        whisper_chunks = self.audio_processor.get_whisper_chunk(
            whisper_input_features,
            self.device,
            self.weight_dtype,
            self.whisper,
            librosa_length,
            fps=fps,
            audio_padding_length_left=2,
            audio_padding_length_right=2,
        )
        print(f"[phase B] whisper: {len(whisper_chunks)} chunks in {time.perf_counter() - t0:.2f}s")

        # ---- Phase C: face detection (or use precomputed bboxes) -------------
        t0 = time.perf_counter()
        coord_list, frame_list = get_landmark_and_bbox(
            img_list=None,
            upperbondrange=bbox_shift,
            precomputed_bboxes=precomputed_bboxes,
            frames=in_memory_frames,
        )
        print(f"[phase C] bbox: {time.perf_counter() - t0:.2f}s")

        # ---- Phase D: VAE encoding (batched OR skip if cached) ---------------
        t0 = time.perf_counter()
        if precomputed_latents is not None:
            input_latent_list = list(precomputed_latents)
            # Align with decoded frame_list — caller's length may differ by a
            # frame or two due to ffmpeg/cv2 rounding around the trim point.
            n_dec = len(frame_list)
            if len(input_latent_list) > n_dec:
                input_latent_list = input_latent_list[:n_dec]
            elif len(input_latent_list) < n_dec:
                raise RuntimeError(
                    f"precomputed_latents length {len(input_latent_list)} is shorter "
                    f"than decoded frames {n_dec}; cannot align safely"
                )
            print(f"[phase D] VAE encode SKIPPED (cache hit): {len(input_latent_list)} latents reused")
        else:
            crops_to_encode: List[np.ndarray] = []
            for bbox, frame in zip(coord_list, frame_list):
                if bbox == coord_placeholder:
                    continue
                x1, y1, x2, y2 = bbox
                y2c = min(y2 + extra_margin, frame.shape[0])
                crop_frame = frame[y1:y2c, x1:x2]
                crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
                crops_to_encode.append(crop_frame)

            input_latent_list = []
            for i in range(0, len(crops_to_encode), vae_batch_size):
                chunk = crops_to_encode[i:i + vae_batch_size]
                batched = self.vae.get_latents_for_unet_batch(chunk)  # [N, 8, 32, 32]
                for j in range(batched.shape[0]):
                    input_latent_list.append(batched[j:j+1])
            print(f"[phase D] VAE encode: {len(input_latent_list)} latents in {time.perf_counter() - t0:.2f}s")

        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        if precomputed_masks is not None:
            linear_masks = list(precomputed_masks)
            n_dec = len(frame_list)
            if len(linear_masks) > n_dec:
                linear_masks = linear_masks[:n_dec]
            elif len(linear_masks) < n_dec:
                raise RuntimeError(
                    f"precomputed_masks length {len(linear_masks)} is shorter "
                    f"than decoded frames {n_dec}; cannot align safely"
                )
            precomputed_masks_cycle: Optional[List[np.ndarray]] = linear_masks + linear_masks[::-1]
        else:
            precomputed_masks_cycle = None

        # ---- Phase E: UNet inference ----------------------------------------
        t0 = time.perf_counter()
        print("Starting UNet inference...")
        video_num = len(whisper_chunks)
        device_str = str(self.device)
        gen = datagen(
            whisper_chunks=whisper_chunks,
            vae_encode_latents=input_latent_list_cycle,
            batch_size=batch_size,
            delay_frame=0,
            device=device_str,
        )

        res_frame_list: List[np.ndarray] = []
        total = int(np.ceil(float(video_num) / batch_size))

        for i, (whisper_batch, latent_batch) in enumerate(tqdm(gen, total=total)):
            audio_feature_batch = self.pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=self.weight_dtype)

            pred_latents = self.unet.model(
                latent_batch, self.timesteps, encoder_hidden_states=audio_feature_batch
            ).sample
            recon = self.vae.decode_latents(pred_latents)
            for res_frame in recon:
                res_frame_list.append(res_frame)
        print(f"[phase E] UNet inference: {len(res_frame_list)} frames in {time.perf_counter() - t0:.2f}s")

        # ---- Phase F: Blending (batched face parsing + precomputed masks) ----
        t0 = time.perf_counter()
        print("Blending frames" + (" with GFPGAN enhancement" if enhance else ""))

        if enhance:
            self._load_gfpgan()

        n_blend = len(res_frame_list)
        face_boxes: List[Optional[tuple]] = [None] * n_blend
        crop_boxes: List[Optional[tuple]] = [None] * n_blend
        ori_shapes: List[Optional[tuple]] = [None] * n_blend
        face_large_pil_list: List[Optional[Image.Image]] = [None] * n_blend
        ori_frames_for_blend: List[Optional[np.ndarray]] = [None] * n_blend

        for i in range(n_blend):
            bbox = coord_list_cycle[i % len(coord_list_cycle)]
            if bbox == coord_placeholder:
                continue
            ori_frame = frame_list_cycle[i % len(frame_list_cycle)]
            x1, y1, x2, y2 = bbox
            y2c = min(y2 + extra_margin, ori_frame.shape[0])
            face_box = (x1, y1, x2, y2c)
            crop_box, _ = get_crop_box(face_box, 1.5)
            face_boxes[i] = face_box
            crop_boxes[i] = crop_box
            ori_frames_for_blend[i] = ori_frame
            # face_large is BGR->RGB PIL, cropped on the expanded box
            body_pil = Image.fromarray(ori_frame[:, :, ::-1])
            face_large_pil = body_pil.crop(crop_box)
            face_large_pil_list[i] = face_large_pil
            ori_shapes[i] = face_large_pil.size

        if precomputed_masks_cycle is None:
            fp = FaceParsing(
                left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width
            )
            blend_masks: List[Optional[np.ndarray]] = [None] * n_blend
            valid_idx = [i for i in range(n_blend) if face_large_pil_list[i] is not None]
            for start in range(0, len(valid_idx), parsing_batch_size):
                batch_idx = valid_idx[start:start + parsing_batch_size]
                batch_imgs = [face_large_pil_list[i] for i in batch_idx]
                parsing_results = fp.batch_call(batch_imgs, mode=parsing_mode)
                for i_local, i_global in enumerate(batch_idx):
                    parsing_pil = parsing_results[i_local].resize(ori_shapes[i_global])
                    blend_masks[i_global] = _build_blend_mask_from_parsing(
                        parsing_pil,
                        face_boxes[i_global],
                        crop_boxes[i_global],
                        ori_shapes[i_global],
                    )
        else:
            blend_masks = precomputed_masks_cycle
            fp = None

        combine_frames: List[np.ndarray] = []
        for i, res_frame in enumerate(tqdm(res_frame_list, desc="Blending")):
            bbox = coord_list_cycle[i % len(coord_list_cycle)]
            if bbox == coord_placeholder:
                combine_frames.append(frame_list_cycle[i % len(frame_list_cycle)])
                continue
            x1, y1, x2, y2c = face_boxes[i]
            ori_frame = ori_frames_for_blend[i].copy()

            face_crop = res_frame.astype(np.uint8)
            if enhance:
                face_crop = self._enhance_face_aligned(face_crop, gfpgan_weight)
            try:
                face_resized = cv2.resize(face_crop, (x2 - x1, y2c - y1))
            except Exception:
                combine_frames.append(ori_frame)
                continue

            combine_frame = get_image(
                ori_frame,
                face_resized,
                [x1, y1, x2, y2c],
                mode=parsing_mode,
                fp=fp,
                precomputed_mask=blend_masks[i],
            )
            combine_frames.append(combine_frame)
        print(f"[phase F] blending: {len(combine_frames)} frames in {time.perf_counter() - t0:.2f}s")

        # ---- Phase G: ffmpeg encode (pipe stdin, h264_nvenc with fallback) ---
        t0 = time.perf_counter()
        temp_vid_path = os.path.join(temp_dir, f"temp_{input_basename}_{audio_basename}.mp4")
        h, w = combine_frames[0].shape[:2]

        encoded_ok = False
        if use_nvenc:
            try:
                _write_video_pipe(combine_frames, temp_vid_path, w, h, fps, codec="h264_nvenc")
                encoded_ok = True
            except subprocess.CalledProcessError as e:
                print(f"[phase G] h264_nvenc failed (rc={e.returncode}); fallback to libx264")
            except FileNotFoundError:
                print("[phase G] ffmpeg not found in PATH; cannot encode")
                raise
        if not encoded_ok:
            _write_video_pipe(combine_frames, temp_vid_path, w, h, fps, codec="libx264")
        print(f"[phase G] video encode: {time.perf_counter() - t0:.2f}s")

        # mux audio (copy video stream, encode audio to AAC)
        t0 = time.perf_counter()
        mux_cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-i", audio_path, "-i", temp_vid_path,
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            output_vid_name,
        ]
        subprocess.run(mux_cmd, check=True, stdin=subprocess.DEVNULL)
        print(f"[phase G] audio mux: {time.perf_counter() - t0:.2f}s")

        shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"Results saved to {output_vid_name}")
        return output_vid_name

    def get_gpu_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "gpu_available": torch.cuda.is_available(),
            "gpu_name": None,
            "memory_allocated": None,
            "memory_reserved": None,
            "memory_total": None,
        }
        if info["gpu_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["memory_allocated"] = torch.cuda.memory_allocated(0)
            info["memory_reserved"] = torch.cuda.memory_reserved(0)
            props = torch.cuda.get_device_properties(0)
            info["memory_total"] = props.total_memory
        return info
