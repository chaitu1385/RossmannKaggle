"""Tests for AIFeatureBase — shared Claude client wrapper."""

import unittest
from unittest.mock import MagicMock, patch

from src.ai.base import AIFeatureBase

import pytest

pytestmark = pytest.mark.unit


def _make_base(client=None, max_retries=3):
    """Create an AIFeatureBase instance with a mock or no client."""
    base = AIFeatureBase.__new__(AIFeatureBase)
    base._client = client
    base._model = "claude-sonnet-4-20250514"
    base._max_tokens = 2000
    base._timeout = 60.0
    base._max_retries = max_retries
    return base


class TestAIFeatureBaseInit(unittest.TestCase):
    def test_available_with_client(self):
        base = _make_base(client=MagicMock())
        self.assertTrue(base.available)

    def test_unavailable_without_client(self):
        base = _make_base()
        self.assertFalse(base.available)

    def test_custom_model(self):
        base = _make_base()
        base._model = "claude-opus-4-20250514"
        self.assertEqual(base._model, "claude-opus-4-20250514")


class TestCallClaude(unittest.TestCase):
    def setUp(self):
        self.base = _make_base()

    def test_call_claude_success(self):
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="Hello response")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        self.base._client = mock_client

        result = self.base._call_claude("system", "user")
        self.assertEqual(result, "Hello response")
        mock_client.messages.create.assert_called_once()

    def test_call_claude_raises_when_unavailable(self):
        self.base._client = None
        with self.assertRaises(RuntimeError):
            self.base._call_claude("system", "user")

    def test_call_claude_custom_max_tokens(self):
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="response")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message
        self.base._client = mock_client

        self.base._call_claude("system", "user", max_tokens=500)
        call_kwargs = mock_client.messages.create.call_args.kwargs
        self.assertEqual(call_kwargs["max_tokens"], 500)

    def test_call_claude_propagates_non_retryable_exception(self):
        """Auth errors and bad requests are raised immediately."""
        import anthropic

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.AuthenticationError(
            message="Invalid key",
            response=MagicMock(status_code=401),
            body=None,
        )
        self.base._client = mock_client

        with self.assertRaises(anthropic.AuthenticationError):
            self.base._call_claude("system", "user")
        # Should not retry — only 1 call
        self.assertEqual(mock_client.messages.create.call_count, 1)


class TestCallClaudeRetry(unittest.TestCase):
    """Tests for retry and timeout behavior."""

    @patch("src.ai.base.time.sleep")
    def test_retries_on_rate_limit(self, mock_sleep):
        """Rate limit errors trigger retries with backoff."""
        import anthropic

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="success")]
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            anthropic.RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            anthropic.RateLimitError(
                message="Rate limited",
                response=MagicMock(status_code=429),
                body=None,
            ),
            mock_message,  # succeeds on 3rd attempt
        ]

        base = _make_base(client=mock_client, max_retries=3)
        result = base._call_claude("system", "user")

        self.assertEqual(result, "success")
        self.assertEqual(mock_client.messages.create.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("src.ai.base.time.sleep")
    def test_retries_on_server_error(self, mock_sleep):
        """5xx server errors trigger retries."""
        import anthropic

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="success")]
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            anthropic.InternalServerError(
                message="Server error",
                response=MagicMock(status_code=500),
                body=None,
            ),
            mock_message,
        ]

        base = _make_base(client=mock_client, max_retries=2)
        result = base._call_claude("system", "user")

        self.assertEqual(result, "success")
        self.assertEqual(mock_client.messages.create.call_count, 2)

    @patch("src.ai.base.time.sleep")
    def test_retries_on_connection_error(self, mock_sleep):
        """Connection errors trigger retries."""
        import anthropic

        mock_message = MagicMock()
        mock_message.content = [MagicMock(text="success")]
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = [
            anthropic.APIConnectionError(request=MagicMock()),
            mock_message,
        ]

        base = _make_base(client=mock_client, max_retries=2)
        result = base._call_claude("system", "user")

        self.assertEqual(result, "success")
        self.assertEqual(mock_client.messages.create.call_count, 2)

    @patch("src.ai.base.time.sleep")
    def test_raises_after_exhausting_retries(self, mock_sleep):
        """Raises the last exception after all retries are exhausted."""
        import anthropic

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )

        base = _make_base(client=mock_client, max_retries=2)
        with self.assertRaises(anthropic.RateLimitError):
            base._call_claude("system", "user")
        # 1 initial + 2 retries = 3 calls
        self.assertEqual(mock_client.messages.create.call_count, 3)

    def test_no_retry_on_zero_max_retries(self):
        """With max_retries=0, failures are raised immediately."""
        import anthropic

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = anthropic.RateLimitError(
            message="Rate limited",
            response=MagicMock(status_code=429),
            body=None,
        )

        base = _make_base(client=mock_client, max_retries=0)
        with self.assertRaises(anthropic.RateLimitError):
            base._call_claude("system", "user")
        self.assertEqual(mock_client.messages.create.call_count, 1)


class TestParseSections(unittest.TestCase):
    def test_parses_multiple_sections(self):
        text = """### ANSWER
This is the answer.

### CONFIDENCE
high

### DATA
Some data here."""

        result = AIFeatureBase._parse_sections(text, ["ANSWER", "CONFIDENCE", "DATA"])
        self.assertEqual(result["ANSWER"], "This is the answer.")
        self.assertEqual(result["CONFIDENCE"], "high")
        self.assertEqual(result["DATA"], "Some data here.")

    def test_empty_text(self):
        result = AIFeatureBase._parse_sections("", ["ANSWER"])
        self.assertEqual(result["ANSWER"], "")

    def test_missing_sections(self):
        text = "### ANSWER\nSome answer."
        result = AIFeatureBase._parse_sections(text, ["ANSWER", "MISSING"])
        self.assertIn("Some answer", result["ANSWER"])
        self.assertEqual(result["MISSING"], "")

    def test_case_insensitive_matching(self):
        text = "### Answer\nContent here."
        result = AIFeatureBase._parse_sections(text, ["ANSWER"])
        self.assertIn("Content here", result["ANSWER"])


class TestParseBullets(unittest.TestCase):
    def test_dash_bullets(self):
        text = "- First\n- Second\n- Third"
        self.assertEqual(AIFeatureBase._parse_bullets(text), ["First", "Second", "Third"])

    def test_star_bullets(self):
        text = "* Alpha\n* Beta"
        self.assertEqual(AIFeatureBase._parse_bullets(text), ["Alpha", "Beta"])

    def test_mixed_bullets(self):
        text = "- One\n* Two\n• Three"
        self.assertEqual(len(AIFeatureBase._parse_bullets(text)), 3)

    def test_empty_string(self):
        self.assertEqual(AIFeatureBase._parse_bullets(""), [])

    def test_no_bullets(self):
        self.assertEqual(AIFeatureBase._parse_bullets("Just plain text\nAnother line"), [])


class TestParseJsonBlock(unittest.TestCase):
    def test_fenced_json(self):
        text = '```json\n{"key": "value"}\n```'
        result = AIFeatureBase._parse_json_block(text)
        self.assertEqual(result, {"key": "value"})

    def test_raw_json_object(self):
        text = 'Some text {"key": "value"} more text'
        result = AIFeatureBase._parse_json_block(text)
        self.assertEqual(result, {"key": "value"})

    def test_raw_json_array(self):
        text = 'Some text [1, 2, 3] more text'
        result = AIFeatureBase._parse_json_block(text)
        self.assertEqual(result, [1, 2, 3])

    def test_invalid_json(self):
        text = "No JSON here at all"
        result = AIFeatureBase._parse_json_block(text)
        self.assertIsNone(result)

    def test_fenced_json_array(self):
        text = '```json\n[{"a": 1}, {"a": 2}]\n```'
        result = AIFeatureBase._parse_json_block(text)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
