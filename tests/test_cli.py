import json
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from akoma_markup.cli import _config_from_env, main


class TestConfigFromEnv:
    def test_azure_from_provider(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "PROVIDER=azure\n"
            "AZURE_INFERENCE_ENDPOINT=https://x.azure.com\n"
            "AZURE_INFERENCE_CREDENTIAL=key123\n"
            "AZURE_MODEL_ID=Llama-3.3-70B-Instruct\n"
        )
        config = _config_from_env(str(env))
        assert config == {
            "provider": "azure",
            "endpoint": "https://x.azure.com",
            "credential": "key123",
            "model": "Llama-3.3-70B-Instruct",
        }

    def test_azure_inferred_from_endpoint(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "AZURE_INFERENCE_ENDPOINT=https://x.azure.com\n"
            "AZURE_INFERENCE_CREDENTIAL=key123\n"
        )
        config = _config_from_env(str(env))
        assert config["provider"] == "azure"

    def test_anthropic_from_provider(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "PROVIDER=anthropic\n"
            "ANTHROPIC_API_KEY=sk-test\n"
            "ANTHROPIC_MODEL_ID=claude-3\n"
        )
        config = _config_from_env(str(env))
        assert config == {
            "provider": "anthropic",
            "api_key": "sk-test",
            "model": "claude-3",
        }

    def test_anthropic_inferred_from_api_key(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("ANTHROPIC_API_KEY=sk-xyz\n")
        config = _config_from_env(str(env))
        assert config["provider"] == "anthropic"

    def test_strips_quotes_and_whitespace(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            'PROVIDER="anthropic"\n'
            "ANTHROPIC_API_KEY='sk-quoted'\n"
        )
        config = _config_from_env(str(env))
        assert config["api_key"] == "sk-quoted"

    def test_ignores_comments_and_blank_lines(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text(
            "# this is a comment\n"
            "\n"
            "PROVIDER=anthropic\n"
            "ANTHROPIC_API_KEY=sk\n"
        )
        config = _config_from_env(str(env))
        assert config["provider"] == "anthropic"

    def test_raises_when_no_provider_determinable(self, tmp_path):
        env = tmp_path / ".env"
        env.write_text("SOMETHING_ELSE=value\n")
        with pytest.raises(Exception) as exc_info:
            _config_from_env(str(env))
        assert "provider" in str(exc_info.value).lower()


class TestMainCli:
    def test_requires_a_config_source(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        runner = CliRunner()
        result = runner.invoke(main, [str(pdf)])
        assert result.exit_code != 0
        assert "Provide one of" in result.output

    def test_rejects_multiple_config_sources(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        json_path = tmp_path / "llm.json"
        json_path.write_text(json.dumps({"provider": "anthropic"}))
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(pdf),
                "--llm-inline",
                '{"provider":"anthropic"}',
                "--llm-json",
                str(json_path),
            ],
        )
        assert result.exit_code != 0
        assert "only one" in result.output.lower()

    def test_rejects_missing_pdf(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                str(tmp_path / "nonexistent.pdf"),
                "--llm-inline",
                '{"provider":"anthropic"}',
            ],
        )
        assert result.exit_code != 0

    def test_invalid_inline_json(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        runner = CliRunner()
        result = runner.invoke(main, [str(pdf), "--llm-inline", "{not-json"])
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output

    def test_calls_convert_with_inline_config(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        runner = CliRunner()
        with patch("akoma_markup.convert") as mock_convert:
            mock_convert.return_value = "/tmp/out_markup.txt"
            result = runner.invoke(
                main,
                [
                    str(pdf),
                    "--llm-inline",
                    '{"provider":"anthropic","api_key":"k"}',
                ],
            )
        assert result.exit_code == 0, result.output
        mock_convert.assert_called_once()
        kwargs = mock_convert.call_args.kwargs
        assert kwargs["llm_config"] == {"provider": "anthropic", "api_key": "k"}

    def test_calls_convert_with_json_file(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({"provider": "azure", "model": "Llama"}))
        runner = CliRunner()
        with patch("akoma_markup.convert") as mock_convert:
            mock_convert.return_value = "/tmp/out.txt"
            result = runner.invoke(main, [str(pdf), "--llm-json", str(cfg)])
        assert result.exit_code == 0, result.output
        assert mock_convert.call_args.kwargs["llm_config"]["provider"] == "azure"

    def test_calls_convert_with_env_file(self, tmp_path):
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4\n")
        env = tmp_path / ".env"
        env.write_text("PROVIDER=anthropic\nANTHROPIC_API_KEY=k\n")
        runner = CliRunner()
        with patch("akoma_markup.convert") as mock_convert:
            mock_convert.return_value = "/tmp/out.txt"
            result = runner.invoke(main, [str(pdf), "--llm-env", str(env)])
        assert result.exit_code == 0, result.output
        assert mock_convert.call_args.kwargs["llm_config"]["provider"] == "anthropic"
