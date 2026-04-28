"""LLM-based highlight detection using the Gemini REST API."""

import asyncio
import json
import os
import base64
import uuid
from pathlib import Path
from typing import Optional

import httpx
from loguru import logger

from shortgen.config import settings
from shortgen.core.exceptions import AnalysisError


class GeminiHighlightFinder:
    """Find highlight-worthy segments using Gemini LLM over HTTP."""

    def __init__(
        self,
        timeout: float = 120.0,
    ):
        self.api_key = settings.gemini_api_key
        self.model_name = settings.gemini_model
        self.timeout = timeout
        
        if not self.api_key:
            logger.warning("No Gemini API key found. Will use fallback scoring.")

    async def find_highlights(
        self,
        subtitle_path: str,
        video_duration: float,
        subtitle_lang: str = 'en',
        num_highlights: int = 2,
    ) -> list[dict]:
        if not subtitle_path.strip():
            logger.warning("Empty transcript, skipping highlight detection")
            return []
        
        with open(subtitle_path, 'r', encoding='utf-8') as file:
            content = file.read()

        logger.info(f"Finding highlights with Gemini (HTTP): {self.model_name}, duration: {video_duration}")

        try:
            if not self.api_key:
                logger.warning("Gemini API key not configured, using fallback scoring")
                return self._fallback_highlights(content, video_duration)

            prompt = self._build_prompt(video_duration, num_highlights, subtitle_lang)
            logger.info("======================================================================")
            logger.info(prompt)
            logger.info("======================================================================")
            
            # Use a shared client for connection pooling across the upload, query, and delete steps
            async with httpx.AsyncClient() as client:
                # 1. Upload the transcript to the File API
                logger.debug("Uploading transcript to Gemini File API...")
                file_info = await self._upload_transcript(client, content)
                file_uri = file_info.get("uri")
                file_name = file_info.get("name")
                mime_type = file_info.get("mimeType", "text/plain")

                if not file_uri:
                    raise AnalysisError("Failed to get file URI after upload.")

                try:
                    # 2. Query the LLM with the attached file
                    response_text = await self._query_llm(client, prompt, file_uri, mime_type)
                finally:
                    # 3. Clean up the uploaded file to free up storage quota
                    if file_name:
                        await self._delete_file(client, file_name)

                highlights = self._parse_response(response_text, video_duration)
                
                # 4. Generate TTS for each highlight hook
                temp_dir = Path("data/temp")
                temp_dir.mkdir(parents=True, exist_ok=True)

                for i, highlight in enumerate(highlights):
                    hook_text = highlight.get("hook")
                    if hook_text:
                        wav_filename = f"hook_{uuid.uuid4().hex[:8]}.wav"
                        wav_path = temp_dir / wav_filename
                        
                        logger.info(f"Generating TTS for highlight {i+1}/{len(highlights)}: '{hook_text}'")
                        try:
                            await self._generate_tts(client, hook_text, wav_path)
                            highlight["hook_audio_path"] = str(wav_path)
                        except Exception as e:
                            logger.error(f"Failed to generate TTS for hook '{hook_text}': {e}")
                            highlight["hook_audio_path"] = None

            logger.info(f"Found {len(highlights)} highlights")
            logger.info(f"Highlight: {json.dumps(highlights, indent=2)}")
            return highlights

        except Exception as e:
            logger.error(f"Highlight detection failed: {e}")
            return self._fallback_highlights(subtitle_path, video_duration)

    async def _generate_tts(self, client: httpx.AsyncClient, text: str, output_wav_path: Path) -> None:
        """Generate audio using Gemini TTS API and convert it to WAV via FFmpeg."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.1-flash-tts-preview:generateContent?key={self.api_key}"
        
        payload = {
            "contents": [{
                "parts": [{"text": text}]
            }],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": "Charon"
                        }
                    }
                }
            }
        }
        
        response = await client.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=self.timeout
        )

        if response.status_code != 200:
            raise AnalysisError(f"TTS API failed ({response.status_code}): {response.text}")

        result = response.json()
        
        try:
            base64_audio = result["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        except (KeyError, IndexError) as e:
            raise AnalysisError(f"Failed to parse TTS response structure: {e}\nRaw: {result}")

        # Decode base64 to PCM
        pcm_data = base64.b64decode(base64_audio)
        temp_pcm_path = output_wav_path.with_suffix(".pcm")
        
        with open(temp_pcm_path, "wb") as f:
            f.write(pcm_data)
            
        # Convert PCM to WAV using FFmpeg
        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le",
            "-ar", "24000",
            "-ac", "1",
            "-i", str(temp_pcm_path),
            str(output_wav_path)
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await process.communicate()
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise AnalysisError(f"FFmpeg failed converting TTS PCM to WAV: {error_msg}")
        finally:
            if temp_pcm_path.exists():
                try:
                    temp_pcm_path.unlink()
                except Exception as e:
                    logger.warning(f"Failed to clean up temp PCM file {temp_pcm_path}: {e}")

    async def _upload_transcript(self, client: httpx.AsyncClient, transcript: str) -> dict:
        """Uploads the transcript to the Gemini File API."""
        url = f"https://generativelanguage.googleapis.com/upload/v1beta/files?key={self.api_key}"
        headers = {
            "Content-Type": "text/plain",
        }
        
        response = await client.post(
            url,
            content=transcript.encode("utf-8"),
            headers=headers,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise AnalysisError(f"Failed to upload transcript file ({response.status_code}): {response.text}")
            
        result = response.json()
        return result.get("file", {})

    async def _delete_file(self, client: httpx.AsyncClient, file_name: str) -> None:
        """Deletes the uploaded file from Gemini storage."""
        url = f"https://generativelanguage.googleapis.com/v1beta/{file_name}?key={self.api_key}"
        try:
            response = await client.delete(url, timeout=10.0)
            if response.status_code == 200:
                logger.debug(f"Successfully cleaned up temporary file: {file_name}")
            else:
                logger.warning(f"Failed to clean up file {file_name}: {response.text}")
        except Exception as e:
            logger.warning(f"Exception cleaning up file {file_name}: {e}")

    def _build_prompt(
        self,
        video_duration: float,
        num_highlights: int,
        subtitle_lang: str
    ) -> str:
        """Build prompt for highlight detection."""
        actual_lang = ""
        if subtitle_lang.lower().startswith("id"):
            actual_lang = "Indonesian"
        elif subtitle_lang.lower().startswith("ko"):
            actual_lang = "Korean"
        else: 
            actual_lang = "English"

        return f"""Analyze the attached video transcript file (.srt) and identify the {num_highlights} most engaging moments that would make great short-form video clips (45-120 seconds each).
            VIDEO DURATION: {video_duration:.0f} seconds

            CRITICAL LANGUAGE REQUIREMENT:
            The generated "hook" text ABSOLUTELY MUST be written in {actual_lang.upper()}. Do NOT translate the hook into English unless {actual_lang} is English.

            CRITICAL INSTRUCTION TO PREVENT TIMESTAMPS HALLUCINATION:
            Do not guess, estimate, or calculate the timestamps. You must GROUND your extraction.
            1. Find the exact block of text you want to use.
            2. Extract the exact first 5-8 words ("exact_start_quote").
            3. Look directly above those words in the SRT file and copy the EXACT timestamp string ("start_time").
            4. Do the same for the end of the clip ("exact_end_quote" and "end_time").

            Look for:
            - Surprising or counterintuitive statements
            - Emotional moments (humor, inspiration, shock)
            - Key insights or "aha" moments
            - Quotable statements
            - Dramatic tension or conflict

            Respond ONLY with valid JSON using this exact schema:
            {{
                "highlights": [
                    {{
                        "exact_start_quote": "kecelakaan sejarah karena waktu itu",
                        "start_time": "00:04:04,040",
                        "exact_end_quote": "yang menyenangi kecelakaan sejarah yang",
                        "end_time": "00:04:39,020",
                        "score": 0.9,
                        "reason": "Humorous and relatable opening about philosophy students becoming journalists.",
                        "hook": "[MUST BE IN {actual_lang.upper()}] Catchy hook words for short video 6-10 words"
                    }}
                ]
            }}"""
    
    async def _query_llm(
        self, 
        client: httpx.AsyncClient, 
        prompt: str, 
        file_uri: str, 
        mime_type: str
    ) -> str:
        """Query the Gemini REST API using httpx with the attached file."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        
        # Attach the uploaded file as `fileData` next to the text prompt
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {"fileData": {"fileUri": file_uri, "mimeType": mime_type}}
                ]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "responseMimeType": "application/json",
                # ENFORCE THE EXACT JSON STRUCTURE HERE
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "highlights": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "exact_start_quote": {"type": "STRING"},
                                    "start_time": {"type": "STRING"},
                                    "exact_end_quote": {"type": "STRING"},
                                    "end_time": {"type": "STRING"},
                                    "score": {"type": "NUMBER"},
                                    "reason": {"type": "STRING"},
                                    "hook": {"type": "STRING"}
                                },
                                "required": ["exact_start_quote", "start_time", "exact_end_quote", "end_time", "score", "reason", "hook"]
                            }
                        }
                    },
                    "required": ["highlights"]
                }
            }
        }

        response = await client.post(
            url,
            json=payload,
            timeout=self.timeout,
        )

        if response.status_code != 200:
            raise AnalysisError(f"Gemini API HTTP error ({response.status_code}): {response.text}")

        result = response.json()
        
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise AnalysisError(f"Failed to parse Gemini API JSON response structure: {e}\nRaw response: {result}")

    def _time_to_seconds(self, time_str) -> float:
        """Convert HH:MM:SS,000 or MM:SS string from SRT to seconds."""
        try:
            # If the LLM disobeys and returns a raw float/int anyway, just return it
            if isinstance(time_str, (int, float)):
                return float(time_str)
                
            time_str = str(time_str).strip()
            
            # If the LLM accidentally copies the whole SRT line (e.g., "00:00:00,000 --> 00:00:02,399")
            if "-->" in time_str:
                time_str = time_str.split("-->")[0].strip()

            # Clean up the string (handle commas used for milliseconds in SRT)
            time_str = time_str.replace(',', '.')
            
            if '.' in time_str:
                time_parts = time_str.split('.')
                main_time = time_parts[0]
                ms_val = float(f"0.{time_parts[1]}")
            else:
                main_time = time_str
                ms_val = 0.0

            parts = main_time.split(':')

            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + int(s) + ms_val
            elif len(parts) == 2:
                m, s = parts
                return int(m) * 60 + int(s) + ms_val
                
            return float(time_str)
        except Exception as e:
            logger.warning(f"Could not parse time string '{time_str}': {e}")
            return 0.0

    def _parse_response(
        self,
        response_text: str,
        video_duration: float,
    ) -> list[dict]:
        """Parse LLM response into highlight list."""
        try:
            data = json.loads(response_text)

            # Handle both dictionary {"highlights": [...]} and direct list [...] formats
            if isinstance(data, dict):
                raw_highlights = data.get("highlights", [])
            elif isinstance(data, list):
                raw_highlights = data
            else:
                logger.warning(f"Unexpected JSON structure from LLM. Expected dict or list, got {type(data)}")
                return []

            highlights = []
            for h in raw_highlights:
                # Extract the strings (or fallback to older 'start'/'end' keys just in case)
                start_raw = h.get("start_time", h.get("start", 0))
                end_raw = h.get("end_time", h.get("end", 0))

                # Convert strings to seconds in Python
                start_sec = self._time_to_seconds(start_raw)
                end_sec = self._time_to_seconds(end_raw)

                # Validate and clamp values
                start = max(0.0, start_sec)
                end = min(video_duration, max(start + 15.0, end_sec)) 
                score = max(0.0, min(1.0, float(h.get("score", 0.5))))

                if end > start:
                    highlights.append({
                        "start": start,
                        "end": end,
                        "score": score,
                        "reason": h.get("reason", ""),
                        "hook": h.get("hook", ""),  # Memasukkan data 'hook' agar bisa dipakai TTS
                    })

            return highlights

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Raw response was: {response_text}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing highlights: {e}")
            logger.debug(f"Raw response was: {response_text}")
            return []

    def _fallback_highlights(
        self,
        transcript: str,
        video_duration: float,
    ) -> list[dict]:
        """Simple keyword-based fallback when LLM unavailable."""
        engagement_keywords = [
            "amazing", "incredible", "surprising", "secret", "actually",
            "important", "key", "crucial", "interesting", "funny",
            "crazy", "unbelievable", "truth", "mistake", "problem",
            "solution", "tip", "trick", "hack", "best", "worst",
        ]

        words = transcript.lower().split()
        words_per_second = len(words) / video_duration if video_duration > 0 else 3

        highlights = []
        segment_duration = getattr(settings, "max_segment_duration", 60)

        for i in range(0, len(words), int(words_per_second * segment_duration / 2)):
            segment_words = words[i:i + int(words_per_second * segment_duration)]
            segment_text = " ".join(segment_words)

            keyword_count = sum(
                1 for kw in engagement_keywords
                if kw in segment_text
            )

            if keyword_count > 0:
                start_time = i / words_per_second
                end_time = min(start_time + segment_duration, video_duration)

                score = min(1.0, keyword_count * 0.15)

                highlights.append({
                    "start": start_time,
                    "end": end_time,
                    "score": score,
                    "reason": "Contains engaging keywords",
                    "hook": "" # Fallback kosong
                })

        highlights.sort(key=lambda x: x["score"], reverse=True)
        return highlights[:10]