import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from akoma_markup.converter import (
    DEFAULT_RATE_CONFIG,
    RETRYABLE_KEYWORDS,
    _load_checkpoint,
    _save_checkpoint,
    build_chain,
    process_all_sections,
)


class TestBuildChain:
    def test_returns_chainlike(self):
        llm = MagicMock()
        # Pipe operators on real runnables work; here we just ensure call succeeds.
        # Using a LangChain-compatible MagicMock that supports __or__.
        llm.__or__ = lambda self, other: MagicMock()
        chain = build_chain(llm, document_name="Test Act")
        assert chain is not None

    def test_includes_document_name_in_prompt(self):
        llm = MagicMock()
        llm.__or__ = lambda self, other: MagicMock()
        with patch("akoma_markup.converter.ChatPromptTemplate.from_messages") as mock_from_messages:
            mock_from_messages.return_value = MagicMock()
            mock_from_messages.return_value.__or__ = lambda self, other: MagicMock()
            build_chain(llm, document_name="My Special Document")
            messages = mock_from_messages.call_args[0][0]
            system_message = messages[0][1]
            assert "My Special Document" in system_message


class TestCheckpointIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        path = tmp_path / "nested" / "cp.json"
        results = [{"num": "1", "markup": "SEC 1."}]
        _save_checkpoint(path, last_index=0, results=results, total=5)
        loaded = _load_checkpoint(path)
        assert loaded["last_completed_index"] == 0
        assert loaded["completed_sections"] == results
        assert loaded["total_sections"] == 5
        assert "timestamp" in loaded

    def test_load_returns_none_when_missing(self, tmp_path):
        assert _load_checkpoint(tmp_path / "missing.json") is None

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "cp.json"
        _save_checkpoint(path, 0, [], 1)
        assert path.exists()


class TestProcessAllSections:
    @pytest.fixture
    def sections(self):
        return [
            {"num": "1", "heading": "Title", "content": "Body 1"},
            {"num": "2", "heading": "Defs", "content": "Body 2"},
        ]

    @pytest.fixture
    def fast_rate_config(self):
        return {
            "delay_between_requests": 0,
            "batch_size": 100,
            "batch_delay": 0,
            "max_retries": 3,
            "initial_backoff": 0,
        }

    def test_successful_conversion(self, sections, fast_rate_config):
        chain = MagicMock()
        chain.invoke.side_effect = ["markup1", "markup2"]
        with patch("akoma_markup.converter.time.sleep"):
            results, errors = process_all_sections(chain, sections, rate_config=fast_rate_config)
        assert len(results) == 2
        assert errors == []
        assert results[0] == {"num": "1", "markup": "markup1"}
        assert results[1] == {"num": "2", "markup": "markup2"}

    def test_non_retryable_error_recorded(self, sections, fast_rate_config):
        chain = MagicMock()
        chain.invoke.side_effect = [ValueError("permanent failure"), "markup2"]
        with patch("akoma_markup.converter.time.sleep"):
            results, errors = process_all_sections(chain, sections, rate_config=fast_rate_config)
        assert len(errors) == 1
        assert errors[0]["num"] == "1"
        assert "permanent failure" in errors[0]["error"]
        # Second section still processes
        assert len(results) == 1
        assert results[0]["num"] == "2"

    def test_retryable_error_retries(self, sections, fast_rate_config):
        chain = MagicMock()
        # First call fails with rate limit, retry succeeds
        chain.invoke.side_effect = [
            Exception("429 rate limit exceeded"),
            "markup1",
            "markup2",
        ]
        with patch("akoma_markup.converter.time.sleep"):
            results, errors = process_all_sections(
                chain, sections[:1] + sections[1:], rate_config=fast_rate_config
            )
        # Should have retried and succeeded on section 1
        assert any(r["num"] == "1" for r in results)

    def test_retryable_error_gives_up_after_max_retries(self, fast_rate_config):
        sections = [{"num": "1", "heading": "T", "content": "B"}]
        chain = MagicMock()
        chain.invoke.side_effect = Exception("429 rate limit")
        with patch("akoma_markup.converter.time.sleep"):
            results, errors = process_all_sections(chain, sections, rate_config=fast_rate_config)
        assert results == []
        # Each retry counts as an invocation attempt
        assert chain.invoke.call_count == fast_rate_config["max_retries"]

    def test_checkpoint_resume(self, tmp_path, sections, fast_rate_config):
        cp_path = tmp_path / "cp" / "checkpoint.json"
        cp_path.parent.mkdir()
        cp_path.write_text(json.dumps({
            "last_completed_index": 0,
            "completed_sections": [{"num": "1", "markup": "cached"}],
            "timestamp": "2024-01-01",
            "total_sections": 2,
        }))

        chain = MagicMock()
        chain.invoke.return_value = "markup2"
        with patch("akoma_markup.converter.time.sleep"):
            results, errors = process_all_sections(
                chain, sections, checkpoint_path=cp_path, rate_config=fast_rate_config
            )
        # Only section 2 should be invoked
        assert chain.invoke.call_count == 1
        assert any(r["markup"] == "cached" for r in results)
        assert any(r["markup"] == "markup2" for r in results)

    def test_checkpoint_saved_at_end(self, tmp_path, sections, fast_rate_config):
        cp_path = tmp_path / "cp" / "checkpoint.json"
        chain = MagicMock()
        chain.invoke.side_effect = ["m1", "m2"]
        with patch("akoma_markup.converter.time.sleep"):
            process_all_sections(chain, sections, checkpoint_path=cp_path, rate_config=fast_rate_config)
        assert cp_path.exists()

    def test_retryable_keywords_constant(self):
        assert "429" in RETRYABLE_KEYWORDS
        assert "rate limit" in RETRYABLE_KEYWORDS

    def test_default_rate_config_has_required_keys(self):
        for key in ("delay_between_requests", "batch_size", "batch_delay", "max_retries", "initial_backoff"):
            assert key in DEFAULT_RATE_CONFIG
