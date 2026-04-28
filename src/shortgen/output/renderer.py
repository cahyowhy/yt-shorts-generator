"""Final video rendering using OpenCV for smooth cropping and FFmpeg for multiplexing."""

import asyncio
import shutil
import tempfile
import re
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

from pathlib import Path
from typing import Optional
from tqdm import tqdm
from loguru import logger

from shortgen.config import settings
from shortgen.core.exceptions import ProcessingError, RenderError
from shortgen.core.models import CropWindow, Platform
from shortgen.processing.captioner import Caption, Captioner

IMG_HOOK_CONFIG = {
    "width": 1080,
    "height": 1920,                          # 9:16 portrait (Reels / TikTok)
    "accent_color": (40, 160, 220),          # BGR – warm amber-gold
    "text_primary": "The secret nobody\ntells you about",
    "text_secondary": "KEEP WATCHING →",
    "output_path": "hook_frame.png",
    "input_path": None,                      # Set via CLI or CONFIG directly
    "dark_overlay_alpha": 0.52,              # 0=no darken, 1=full black
}

class VideoRenderer:
    """Render final short videos with smooth OpenCV cropping and FFmpeg multiplexing."""

    def __init__(self):
        self.captioner = Captioner()

    async def render(
        self,
        input_path: Path,
        output_path: Path,
        crop_windows: list[CropWindow],
        captions: list[Caption],
        hook: Optional[str],
        hook_speaker_path: Optional[Path],
        platform: Platform = Platform.YOUTUBE_SHORTS,
    ) -> Path:
        logger.info(f"Preparing render for: {str(output_path)}")

        caption_file: Optional[str] = None
        temp_video_file: Optional[str] = None
        temp_merged_video_file: Optional[str] = None

        if captions:
            caption_file = await self._write_caption_file(captions)

        try:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                temp_video_file = f.name

            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._process_video_opencv,
                input_path,
                Path(temp_video_file),
                crop_windows
            )

            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                temp_merged_video_file = f.name

            await self._merge_with_ffmpeg(
                temp_video_path=Path(temp_video_file),
                original_input_path=input_path,
                output_path=Path(temp_merged_video_file),
                caption_file=caption_file,
                platform=platform,
                total_duration=crop_windows[-1].timestamp if crop_windows else 0.0
            )

            await self._appendHook(
                temp_video_path=Path(temp_merged_video_file),
                output_path=output_path,
                hook=hook,
                hook_speaker_path=hook_speaker_path
            )

            return output_path

        finally:
            for p in [caption_file, temp_video_file, temp_merged_video_file]:
                if p and Path(p).exists():
                    try:
                        Path(p).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {p}: {e}")

    def _process_video_opencv(self, input_path: Path, output_path: Path, crop_windows: list[CropWindow]) -> None:
        """Reads video, applies crop frame-by-frame, and writes to a temporary file."""
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out_w = settings.output_resolution_width
        out_h = settings.output_resolution_height

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

        frame_idx = 0

        with tqdm(total=total_frames, desc="Cropping Frames (OpenCV)", unit="fr") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                window_idx = min(frame_idx, len(crop_windows) - 1)
                window = crop_windows[window_idx]

                x = int(window.x_offset)
                y = int(window.y_offset)
                w = int(window.width)
                h = int(window.height)

                img_h, img_w = frame.shape[:2]
                x = max(0, min(x, img_w - w))
                y = max(0, min(y, img_h - h))

                cropped_frame = frame[y:y+h, x:x+w]
                resized_frame = cv2.resize(cropped_frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                out.write(resized_frame)
                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()

    async def _run_ffmpeg(self, cmd: list[str], error_prefix: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise ProcessingError(f"{error_prefix}: {error_msg}")
        except FileNotFoundError:
            raise ProcessingError(
                "FFmpeg not found. Please install FFmpeg: https://ffmpeg.org/download.html"
            )

    def load_and_fit_jpeg(self, path: str, target_w: int, target_h: int) -> np.ndarray:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Input image not found: {path!r}")

        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"OpenCV could not read the file: {path!r}")

        src_h, src_w = img.shape[:2]
        scale = max(target_w / src_w, target_h / src_h)

        new_w = int(src_w * scale)
        new_h = int(src_h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        x0 = (new_w - target_w) // 2
        y0 = (new_h - target_h) // 2
        img = img[y0:y0 + target_h, x0:x0 + target_w]
        return img

    def add_dark_overlay(self, img: np.ndarray, alpha: float = 0.50) -> np.ndarray:
        black = np.zeros_like(img)
        return cv2.addWeighted(img, 1.0 - alpha, black, alpha, 0)

    def add_gradient(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        gradient = np.zeros((h, w), dtype=np.float32)

        for y in range(h):
            t = y / h
            gradient[y, :] = 0.12 * np.sin(np.pi * t)

        gradient_bgr = cv2.merge([gradient, gradient, gradient])
        result = np.clip(img.astype(np.float32) / 255.0 + gradient_bgr, 0, 1)
        return (result * 255).astype(np.uint8)

    def add_film_grain(self, img: np.ndarray, intensity: float = 18.0) -> np.ndarray:
        h, w = img.shape[:2]
        noise = np.random.normal(0, intensity, (h, w, 3)).astype(np.float32)
        result = np.clip(img.astype(np.float32) + noise, 0, 255)
        return result.astype(np.uint8)

    def add_vignette(self, img: np.ndarray, strength: float = 0.80) -> np.ndarray:
        h, w = img.shape[:2]
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2, h / 2

        dist = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
        dist = np.clip(dist, 0, 1)

        mask = 1.0 - strength * (dist ** 1.6)
        mask = np.clip(mask, 0, 1)
        mask_3ch = np.dstack([mask] * 3)

        result = (img.astype(np.float32) * mask_3ch).astype(np.uint8)
        return result

    def color_grade(self, img: np.ndarray) -> np.ndarray:
        lut = np.arange(256, dtype=np.float32)
        r_lut = np.clip(lut * 1.08 + 8, 0, 255).astype(np.uint8)
        g_lut = np.clip(lut * 0.96 + 2, 0, 255).astype(np.uint8)
        b_lut = np.clip(lut * 0.92 - 4, 0, 255).astype(np.uint8)

        b, g, r = cv2.split(img)
        r = cv2.LUT(r, r_lut)
        g = cv2.LUT(g, g_lut)
        b = cv2.LUT(b, b_lut)

        graded = cv2.merge([b, g, r])
        warm_fill = np.full_like(graded, (20, 30, 50), dtype=np.uint8)
        graded = cv2.addWeighted(graded, 0.92, warm_fill, 0.08, 0)
        return graded

    def add_decorative_lines(self, img: np.ndarray, cfg: dict) -> np.ndarray:
        h, w = img.shape[:2]
        accent = cfg["accent_color"]
        canvas = img.copy()

        for frac in (0.35, 0.65):
            y = int(h * frac)
            cv2.line(canvas, (80, y), (w - 80, y), accent, 1, cv2.LINE_AA)
            cv2.line(canvas, (80, y), (200, y), accent, 3, cv2.LINE_AA)

        return canvas

    def _find_serif_font(self) -> str | None:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
            "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
            "/usr/share/fonts/truetype/fonts-gujr-extra/Rekha.ttf",
            "/Library/Fonts/Georgia Bold.ttf",
            "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
            "C:/Windows/Fonts/georgiab.ttf",
            "C:/Windows/Fonts/timesbd.ttf",
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def add_text_pil(self, img: np.ndarray, cfg: dict) -> np.ndarray:
        h, w = img.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        font_path = self._find_serif_font()

        # ── PRIMARY headline ──────────────────────────────────────────────
        primary_size = int(w * 0.088)   # ~95px on 1080-wide canvas
        if font_path:
            font_primary = ImageFont.truetype(font_path, primary_size)
        else:
            font_primary = ImageFont.load_default()

        # Solusi Text Terpotong: Fungsi pembungkus (word wrap) teks dinamis
        def wrap_text(text: str, font, max_w: int) -> list[str]:
            paragraphs = text.split("\n")
            wrapped = []
            for p in paragraphs:
                words = p.split()
                curr_line = []
                for word in words:
                    test_line = " ".join(curr_line + [word])
                    bbox = draw.textbbox((0, 0), test_line, font=font)
                    if (bbox[2] - bbox[0]) <= max_w:
                        curr_line.append(word)
                    else:
                        if not curr_line:
                            wrapped.append(word)
                        else:
                            wrapped.append(" ".join(curr_line))
                            curr_line = [word]
                if curr_line:
                    wrapped.append(" ".join(curr_line))
            return wrapped

        # Beri margin yang aman: ~15% kosong di layar (7.5% di kiri dan kanan)
        max_text_width = int(w * 0.85)
        lines = wrap_text(cfg["text_primary"], font_primary, max_text_width)
        
        line_h = primary_size + 18
        total_h = len(lines) * line_h

        start_y = int(h * 0.50) - total_h // 2

        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=font_primary)
            text_w = bbox[2] - bbox[0]
            x = (w - text_w) // 2
            y = start_y + i * line_h

            shadow_offset = 4
            draw.text(
                (x + shadow_offset, y + shadow_offset),
                line, font=font_primary, fill=(0, 0, 0, 180),
            )
            draw.text((x, y), line, font=font_primary, fill=(245, 240, 230))

        # ── SECONDARY CTA ─────────────────────────────────────────────────
        secondary_size = int(w * 0.032)
        if font_path:
            font_secondary = ImageFont.truetype(font_path, secondary_size)
        else:
            font_secondary = font_primary

        cta = cfg["text_secondary"]
        bbox2 = draw.textbbox((0, 0), cta, font=font_secondary)
        cta_w = bbox2[2] - bbox2[0]
        cta_x = (w - cta_w) // 2
        cta_y = int(h * 0.66)

        accent_rgb = cfg["accent_color"][::-1]
        draw.text((cta_x, cta_y), cta, font=font_secondary, fill=accent_rgb)

        result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return result

    def add_lens_flare(self, img: np.ndarray, cfg: dict) -> np.ndarray:
        h, w = img.shape[:2]
        overlay = img.copy()
        cx, cy = int(w * 0.82), int(h * 0.14)
        accent = cfg["accent_color"]

        for radius, alpha in [(120, 0.04), (60, 0.08), (20, 0.20), (6, 0.60)]:
            circle_layer = overlay.copy()
            cv2.circle(circle_layer, (cx, cy), radius, accent, -1, cv2.LINE_AA)
            overlay = cv2.addWeighted(overlay, 1 - alpha, circle_layer, alpha, 0)

        return overlay

    def generate_hook_image(self, cfg: dict = IMG_HOOK_CONFIG) -> np.ndarray:
        input_path = cfg.get("input_path")
        if not input_path:
            raise ValueError("No input image provided.")
            
        img = self.load_and_fit_jpeg(input_path, cfg["width"], cfg["height"])
        img = self.add_dark_overlay(img, alpha=cfg.get("dark_overlay_alpha", 0.52))
        img = self.add_gradient(img)
        img = self.color_grade(img)
        img = self.add_lens_flare(img, cfg)
        img = self.add_decorative_lines(img, cfg)
        img = self.add_text_pil(img, cfg)
        img = self.add_film_grain(img, intensity=14)
        img = self.add_vignette(img, strength=0.75)

        out = cfg["output_path"]
        cv2.imwrite(out, img)
        return img

    def _generate_styled_hook_frame(
        self,
        raw_frame_path: Path,
        hook_text: str,
        cta_text: str = "WATCH TO THE END →",
    ) -> Path:
        styled_path = raw_frame_path.with_name(f"{raw_frame_path.stem}_styled.png")

        hook_cfg = {
            **IMG_HOOK_CONFIG,
            "input_path":   str(raw_frame_path),
            "output_path":  str(styled_path),
            "text_primary":   hook_text,
            "text_secondary": cta_text,
        }

        self.generate_hook_image(cfg=hook_cfg)

        if not styled_path.exists():
            raise ProcessingError(f"video_hook_generator did not produce the expected output: {styled_path}")
        return styled_path

    async def _appendHook(
        self,
        temp_video_path: Path,
        output_path: Path,
        hook: Optional[str],
        hook_speaker_path: Optional[Path],
    ) -> None:
        target_path = output_path

        if hook is None:
            await asyncio.to_thread(shutil.move, str(temp_video_path), str(target_path))
            return

        raw_frame_path = temp_video_path.with_name(f"{temp_video_path.stem}_hook_raw.jpg")
        temp_hook_video = temp_video_path.with_name(f"{temp_video_path.stem}_hook_vid.mp4")
        styled_frame_path: Optional[Path] = None

        try:
            cmd_extract = [
                "ffmpeg", "-y",
                "-ss", "00:00:00",
                "-i", str(temp_video_path),
                "-frames:v", "1",
                str(raw_frame_path),
            ]
            await self._run_ffmpeg(cmd_extract, "FFmpeg failed extracting hook frame")

            if not raw_frame_path.exists():
                raise ProcessingError(f"Hook frame extraction produced no output: {raw_frame_path}")

            styled_frame_path = await asyncio.to_thread(
                self._generate_styled_hook_frame,
                raw_frame_path,
                hook,
                "WATCH TO THE END →",
            )

            # Solusi FFmpeg Hang: Render gambar menjadi video fix-length terlebih dahulu
            fps_str = str(settings.output_fps)
            if hook_speaker_path is None:
                cmd_hook_vid = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-t", "1.2", "-i", str(styled_frame_path),
                    "-f", "lavfi", "-t", "1.2", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20", 
                    "-c:a", "aac", "-b:a", "192k",
                    "-pix_fmt", "yuv420p", "-r", fps_str,
                    "-shortest", str(temp_hook_video)
                ]
            else:
                cmd_hook_vid = [
                    "ffmpeg", "-y",
                    "-loop", "1", "-i", str(styled_frame_path),
                    "-i", str(hook_speaker_path),
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20", 
                    "-c:a", "aac", "-b:a", "192k",
                    "-pix_fmt", "yuv420p", "-r", fps_str,
                    "-shortest", str(temp_hook_video)
                ]
                
            await self._run_ffmpeg(cmd_hook_vid, "FFmpeg failed generating temporary hook video")

            # Solusi Lanjutan FFmpeg Hang: Baru gabungkan kedua video yang memiliki batas durasi jelas
            # Perhatikan resolusi pad dan scale sudah diperbaiki ke format potrait 1080:1920
            cmd_concat = [
                "ffmpeg", "-y",
                "-i", str(temp_hook_video),
                "-i", str(temp_video_path),
                "-filter_complex", (
                    "[0:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
                    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v0]; "
                    "[1:v]scale=1080:1920:force_original_aspect_ratio=decrease,"
                    "pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1,format=yuv420p[v1]; "
                    "[v0][0:a][v1][1:a]concat=n=2:v=1:a=1[v][a]"
                ),
                "-map", "[v]",
                "-map", "[a]",
                str(target_path),
            ]
            await self._run_ffmpeg(cmd_concat, "FFmpeg failed concatenating hook and main video")

            if not target_path.exists():
                raise ProcessingError(f"Final output video not created: {target_path}")

        finally:
            for tmp in [raw_frame_path, styled_frame_path, temp_hook_video]:
                if tmp is not None and tmp.exists():
                    try:
                        tmp.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {tmp}: {e}")

    async def _merge_with_ffmpeg(
        self,
        temp_video_path: Path,
        original_input_path: Path,
        output_path: Path,
        caption_file: Optional[str],
        platform: Platform,
        total_duration: float,
    ) -> None:
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_video_path),
            "-i", str(original_input_path),
        ]

        filters = []
        if caption_file:
            escaped_path = caption_file.replace("\\", "/").replace(":", "\\:")
            filters.append(f"ass='{escaped_path}'")

        if filters:
            cmd.extend(["-vf", ",".join(filters)])

        cmd.extend([
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-r", str(settings.output_fps),
            "-movflags", "+faststart",
        ])

        bitrates = {
            Platform.YOUTUBE_SHORTS:  ["-b:v", "8M",  "-maxrate", "12M", "-bufsize", "16M"],
            Platform.INSTAGRAM_REELS: ["-b:v", "6M",  "-maxrate", "8M",  "-bufsize", "12M"],
            Platform.TIKTOK:          ["-b:v", "6M",  "-maxrate", "8M",  "-bufsize", "12M"],
        }
        cmd.extend(bitrates.get(platform, []))
        cmd.append(str(output_path))

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        time_pattern = re.compile(r"time=(\d{2}):(\d{2}):(\d{2})\.\d+")
        stderr_lines: list[str] = []

        with tqdm(total=100, desc=f"Encoding {output_path.name}", unit="%") as pbar:
            last_pct = 0

            async def drain_stderr() -> None:
                nonlocal last_pct
                assert process.stderr is not None
                async for raw in process.stderr:
                    line_str = raw.decode(errors="replace").strip()
                    stderr_lines.append(line_str)

                    match = time_pattern.search(line_str)
                    if match and total_duration > 0:
                        h, m, s = map(int, match.groups())
                        current_time = h * 3600 + m * 60 + s
                        pct = min(99, int((current_time / total_duration) * 100))
                        if pct > last_pct:
                            pbar.update(pct - last_pct)
                            last_pct = pct

            await asyncio.gather(drain_stderr(), process.wait())
            pbar.update(100 - last_pct)

        if process.returncode != 0:
            tail = "\n".join(stderr_lines[-20:])
            raise RenderError(f"FFmpeg Merge failed with code {process.returncode}:\n{tail}")

    async def _write_caption_file(self, captions: list[Caption]) -> str:
        ass_content = self.captioner.to_ass(captions)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ass", delete=False, encoding="utf-8"
        ) as f:
            f.write(ass_content)
            return f.name