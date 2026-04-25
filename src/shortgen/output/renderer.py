"""Final video rendering using OpenCV for smooth cropping and FFmpeg for multiplexing."""

import asyncio
import tempfile
import re
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from loguru import logger

from shortgen.config import settings
from shortgen.core.exceptions import RenderError
from shortgen.core.models import CropWindow, Platform
from shortgen.processing.captioner import Caption, Captioner


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
        platform: Platform = Platform.YOUTUBE_SHORTS,
    ) -> Path:
        logger.info(f"Preparing render for: {output_path.name}")

        caption_file: Optional[str] = None
        temp_video_file: Optional[str] = None

        if captions:
            caption_file = await self._write_caption_file(captions)

        try:
            # Create a temporary file for the OpenCV cropped video (silent)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
                temp_video_file = f.name

            # Step 1: Crop frame-by-frame using OpenCV
            # We run this in an executor so it doesn't block the async event loop
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._process_video_opencv,
                input_path,
                Path(temp_video_file),
                crop_windows
            )

            # Step 2: Merge audio, burn subtitles, and encode using FFmpeg
            await self._merge_with_ffmpeg(
                temp_video_path=Path(temp_video_file),
                original_input_path=input_path,
                output_path=output_path,
                caption_file=caption_file,
                platform=platform,
                total_duration=crop_windows[-1].timestamp if crop_windows else 0.0
            )

            return output_path

        finally:
            # Cleanup temporary files
            for p in [caption_file, temp_video_file]:
                if p and Path(p).exists():
                    try:
                        Path(p).unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete temp file {p}: {e}")

    def _process_video_opencv(self, input_path: Path, output_path: Path, crop_windows: list[CropWindow]) -> None:
        """Reads video, applies crop frame-by-frame, and writes to a temporary file."""
        import cv2

        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Target output resolution
        out_w = settings.output_resolution_width
        out_h = settings.output_resolution_height

        # Initialize Video Writer (using mp4v codec for temporary file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (out_w, out_h))

        frame_idx = 0
        
        with tqdm(total=total_frames, desc="Cropping Frames (OpenCV)", unit="fr") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Get the crop window for the current frame
                # If we have more frames than windows, use the last known window
                window_idx = min(frame_idx, len(crop_windows) - 1)
                window = crop_windows[window_idx]

                x = int(window.x_offset)
                y = int(window.y_offset)
                w = int(window.width)
                h = int(window.height)

                # Safety boundary check to prevent OpenCV slicing errors
                img_h, img_w = frame.shape[:2]
                x = max(0, min(x, img_w - w))
                y = max(0, min(y, img_h - h))

                # Crop the frame using NumPy slicing
                cropped_frame = frame[y:y+h, x:x+w]

                # Resize to perfectly match the target resolution (1080x1920)
                resized_frame = cv2.resize(cropped_frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                # Write to temp file
                out.write(resized_frame)
                
                frame_idx += 1
                pbar.update(1)

        cap.release()
        out.release()

    async def _merge_with_ffmpeg(
        self,
        temp_video_path: Path,
        original_input_path: Path,
        output_path: Path,
        caption_file: Optional[str],
        platform: Platform,
        total_duration: float,
    ) -> None:
        """Multiplexes the OpenCV video with the original audio and subtitles."""
        
        # -i 0 is the silent cropped video
        # -i 1 is the original horizontal video (we only want its audio)
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_video_path),
            "-i", str(original_input_path),
        ]

        # Apply subtitles via video filter if they exist
        filters = []
        if caption_file:
            escaped_path = caption_file.replace("\\", "/").replace(":", "\\:")
            filters.append(f"ass='{escaped_path}'")

        if filters:
            cmd.extend(["-vf", ",".join(filters)])

        # Map streams: Video from input 0, Audio from input 1 (using ? in case audio is missing)
        cmd.extend([
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-r", str(settings.output_fps),
            "-movflags", "+faststart",
        ])

        # Platform specific bitrates
        bitrates = {
            Platform.YOUTUBE_SHORTS:  ["-b:v", "8M",  "-maxrate", "12M", "-bufsize", "16M"],
            Platform.INSTAGRAM_REELS: ["-b:v", "6M",  "-maxrate", "8M",  "-bufsize", "12M"],
            Platform.TIKTOK:          ["-b:v", "6M",  "-maxrate", "8M",  "-bufsize", "12M"],
        }
        cmd.extend(bitrates.get(platform, []))
        cmd.append(str(output_path))

        logger.debug("FFmpeg Merge Command: " + " ".join(cmd))

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
        """Write captions to a temporary ASS file."""
        ass_content = self.captioner.to_ass(captions)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ass", delete=False, encoding="utf-8"
        ) as f:
            f.write(ass_content)
            return f.name