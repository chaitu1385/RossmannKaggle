"""Shared base class for AI features using Anthropic Claude.

Extracts the common client-initialization, API-call, and response-parsing
patterns from ``LLMAnalyzer`` into a reusable base that all AI feature
classes inherit.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Load .env from project root (if present) so ANTHROPIC_API_KEY is available
try:
    from dotenv import load_dotenv

    _project_root = Path(__file__).resolve().parents[3]
    for _candidate in [_project_root, _project_root / "forecasting-product"]:
        _env_path = _candidate / ".env"
        if _env_path.exists():
            load_dotenv(_env_path)
            break
except ImportError:
    pass

logger = logging.getLogger(__name__)


class AIFeatureBase:
    """Base class for Claude-powered AI features.

    Parameters
    ----------
    api_key : str, optional
        Anthropic API key.  Falls back to ``ANTHROPIC_API_KEY`` env var.
    model : str
        Claude model to use.
    max_tokens : int
        Default max tokens for API responses.
    timeout : float
        HTTP timeout in seconds for Claude API calls.
    max_retries : int
        Number of retries on transient failures (rate limits, server errors).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 2000,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self._client = None
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        try:
            import anthropic

            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if key:
                self._client = anthropic.Anthropic(
                    api_key=key,
                    timeout=timeout,
                )
            else:
                logger.info("%s: no API key provided, AI features disabled", self.__class__.__name__)
        except ImportError:
            logger.info("%s: anthropic package not installed, AI features disabled", self.__class__.__name__)

    @property
    def available(self) -> bool:
        """Whether the Claude client is configured and ready."""
        return self._client is not None

    def _call_claude(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Send a single prompt to Claude and return the response text.

        Retries on transient failures (rate limits, server errors,
        connection issues) with exponential backoff.  Non-retryable
        errors (auth failures, invalid requests) are raised immediately.

        Parameters
        ----------
        system_prompt : str
            System message setting Claude's persona and constraints.
        user_prompt : str
            User message with the actual query / data.
        max_tokens : int, optional
            Override instance default.

        Returns
        -------
        str
            Raw response text from Claude.

        Raises
        ------
        RuntimeError
            If the client is not available.
        Exception
            Propagates non-retryable API errors or exhausted retries.
        """
        if not self.available:
            raise RuntimeError("Claude client not available")

        import anthropic

        last_exc: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=max_tokens or self._max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                return response.content[0].text
            except anthropic.RateLimitError as e:
                last_exc = e
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "Claude rate limited (attempt %d/%d), retrying in %.1fs",
                    attempt + 1, self._max_retries + 1, wait,
                )
            except anthropic.InternalServerError as e:
                last_exc = e
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "Claude server error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self._max_retries + 1, wait, e,
                )
            except anthropic.APIConnectionError as e:
                last_exc = e
                wait = min(2 ** attempt, 30)
                logger.warning(
                    "Claude connection error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, self._max_retries + 1, wait, e,
                )
            except (anthropic.AuthenticationError, anthropic.BadRequestError):
                raise  # non-retryable
            except anthropic.APIStatusError as e:
                if e.status_code >= 500:
                    last_exc = e
                    wait = min(2 ** attempt, 30)
                    logger.warning(
                        "Claude API error %d (attempt %d/%d), retrying in %.1fs",
                        e.status_code, attempt + 1, self._max_retries + 1, wait,
                    )
                else:
                    raise  # 4xx other than rate limit — non-retryable

            if attempt < self._max_retries:
                time.sleep(wait)

        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _parse_sections(text: str, section_names: List[str]) -> Dict[str, str]:
        """Parse Claude's section-headed response into a dict.

        Looks for markdown headers (### SECTION_NAME) and collects all text
        between them.

        Parameters
        ----------
        text : str
            Raw Claude response text.
        section_names : list of str
            Expected section header names (case-insensitive matching).

        Returns
        -------
        dict mapping section name → collected text (stripped).
        """
        sections: Dict[str, str] = {name: "" for name in section_names}
        current_section = None

        for line in text.split("\n"):
            stripped = line.strip()
            # Check for section headers
            matched = False
            for key in section_names:
                if key.upper() in stripped.upper() and stripped.startswith("#"):
                    current_section = key
                    matched = True
                    break
            if not matched and current_section is not None:
                sections[current_section] += line + "\n"

        return {k: v.strip() for k, v in sections.items()}

    @staticmethod
    def _parse_bullets(text: str) -> List[str]:
        """Extract bullet points from markdown text."""
        bullets = []
        for line in text.strip().split("\n"):
            line = line.strip()
            if line.startswith(("-", "*", "•")):
                content = line.lstrip("-*• ").strip()
                if content:
                    bullets.append(content)
        return bullets

    @staticmethod
    def _parse_json_block(text: str) -> Any:
        """Extract and parse a JSON block from text.

        Handles both fenced code blocks (```json ... ```) and raw JSON.

        Returns
        -------
        Parsed JSON object, or None if parsing fails.
        """
        # Try to find fenced JSON block first
        import re

        fence_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find raw JSON array or object
        for start_char, end_char in [("[", "]"), ("{", "}")]:
            start = text.find(start_char)
            end = text.rfind(end_char)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue

        return None
