"""LLM-based highlight detection using Ollama."""

import asyncio
import json
from typing import Optional

import httpx
from loguru import logger

from shortgen.config import settings
from shortgen.core.exceptions import AnalysisError


class HighlightFinder:
    """Find highlight-worthy segments using local LLM."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: float = 120.0,
    ):
        self.base_url = base_url or settings.ollama_base_url
        self.model = model or settings.ollama_model
        self.timeout = timeout

    async def find_highlights(
        self,
        transcript: str,
        video_duration: float,
        num_highlights: int = 10,
    ) -> list[dict]:
        """
        Find highlight-worthy segments in transcript.

        Args:
            transcript: Full transcript text
            video_duration: Video duration in seconds
            num_highlights: Number of highlights to find

        Returns:
            List of highlight dicts with 'start', 'end', 'score', 'reason'
        """
        if not transcript.strip():
            logger.warning("Empty transcript, skipping highlight detection")
            return []

        logger.info(f"Finding highlights with LLM :{ self.model}")

        try:
            # Check if Ollama is available
            if not await self._check_ollama():
                logger.warning("Ollama not available, using fallback scoring")
                return self._fallback_highlights(transcript, video_duration)

            prompt = self._build_prompt(transcript, video_duration, num_highlights)
            response = await self._query_llm(prompt)
            highlights = self._parse_response(response, video_duration)

            logger.info(f"Found {len(highlights)} highlights")
            logger.info(f"Highlight: {json.dumps(highlights)}")
            return highlights

        except Exception as e:
            logger.error(f"Highlight detection failed: {e}")
            return self._fallback_highlights(transcript, video_duration)

    async def _check_ollama(self) -> bool:
        """Check if Ollama is running."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/api/tags",
                    timeout=5.0,
                )
                return response.status_code == 200
        except Exception:
            return False

    def _build_prompt(
        self,
        transcript: str,
        video_duration: float,
        num_highlights: int,
    ) -> str:
        """Build prompt for highlight detection."""

        return f"""Analyze this video transcript and identify the {num_highlights} most engaging moments that would make great short-form video clips (15-120 seconds each).

VIDEO DURATION: {video_duration:.0f} seconds

TRANSCRIPT:
```
{transcript}
```

For each highlight, identify:
1. Exact start time (in seconds, capturing the start of the thought)
2. Exact end time (in seconds, after the thought is fully completed)
3. Engagement score (0.0 to 1.0)
4. Brief reason why this would make a good short

Look for:
- Surprising or counterintuitive statements
- Emotional moments (humor, inspiration, shock)
- Key insights or "aha" moments
- Quotable statements
- Dramatic tension or conflict
- Clear, self-contained ideas that make sense without surrounding context

Respond ONLY with valid JSON in this exact format:
{{
    "highlights": [
        {{"start": 0, "end": 34, "score": 0.9, "reason": "Opening hook with surprising statistic. Ends on a complete thought."}},
        {{"start": 120, "end": 172, "score": 0.85, "reason": "Funny anecdote about the topic. Ends exactly after the punchline."}}
    ]
}}"""

    async def _query_llm(self, prompt: str) -> str:
        """Query Ollama API."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 2000,
                    },
                },
                timeout=self.timeout,
            )

            if response.status_code != 200:
                raise AnalysisError(f"Ollama API error: {response.status_code}")

            result = response.json()
            return result.get("response", "")

    def _parse_response(
        self,
        response: str,
        video_duration: float,
    ) -> list[dict]:
        """Parse LLM response into highlight list."""
        try:
            # Try to extract JSON from response
            # Handle cases where LLM adds extra text
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                logger.warning("No JSON found in LLM response")
                return []

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            highlights = []
            for h in data.get("highlights", []):
                # Validate and clamp values
                start = max(0, float(h.get("start", 0)))
                end = min(video_duration, float(h.get("end", start + 30)))
                score = max(0, min(1, float(h.get("score", 0.5))))

                if end > start:
                    highlights.append({
                        "start": start,
                        "end": end,
                        "score": score,
                        "reason": h.get("reason", ""),
                    })

            return highlights

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response: {e}")
            return []

    def _fallback_highlights(
        self,
        transcript: str,
        video_duration: float,
    ) -> list[dict]:
        """Simple keyword-based fallback when LLM unavailable."""
        # Keywords that often indicate engaging content
        engagement_keywords = [
            "amazing", "incredible", "surprising", "secret", "actually",
            "important", "key", "crucial", "interesting", "funny",
            "crazy", "unbelievable", "truth", "mistake", "problem",
            "solution", "tip", "trick", "hack", "best", "worst",
        ]

        # Split transcript into rough segments
        words = transcript.lower().split()
        words_per_second = len(words) / video_duration if video_duration > 0 else 3

        highlights = []
        segment_duration = settings.max_segment_duration

        for i in range(0, len(words), int(words_per_second * segment_duration / 2)):
            segment_words = words[i:i + int(words_per_second * segment_duration)]
            segment_text = " ".join(segment_words)

            # Count engagement keywords
            keyword_count = sum(
                1 for kw in engagement_keywords
                if kw in segment_text
            )

            if keyword_count > 0:
                start_time = i / words_per_second
                end_time = min(start_time + segment_duration, video_duration)

                score = min(1.0, keyword_count * 0.15)  # Cap at 1.0

                highlights.append({
                    "start": start_time,
                    "end": end_time,
                    "score": score,
                    "reason": "Contains engaging keywords",
                })

        # Sort by score and return top results
        highlights.sort(key=lambda x: x["score"], reverse=True)
        return highlights[:10]
