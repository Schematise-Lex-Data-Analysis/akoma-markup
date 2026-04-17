import sys
import pytest
from unittest.mock import patch, MagicMock
from akoma_markup.llm import build_llm

def test_build_llm_azure():
    """Test build_llm with Azure configuration."""
    test_config = {
        "provider": "azure",
        "model": "Llama-3.3-70B-Instruct",
        "endpoint": "https://test.azure.com",
        "credential": "test-key"
    }
    
    # Patch the import at its source
    with patch('langchain_azure_ai.chat_models.AzureAIChatCompletionsModel', MagicMock()) as MockAzureAI:
        mock_instance = MagicMock()
        MockAzureAI.return_value = mock_instance
        result = build_llm(test_config)
        MockAzureAI.assert_called_once_with(
            endpoint="https://test.azure.com",
            credential="test-key",
            model="Llama-3.3-70B-Instruct",
            temperature=0,
            max_tokens=4096
        )
        assert result == mock_instance

def test_build_llm_azure_with_env_vars():
    """Test build_llm with Azure using environment variables."""
    test_config = {
        "provider": "azure",
        "model": "custom-model",
    }
    
    with patch('langchain_azure_ai.chat_models.AzureAIChatCompletionsModel', MagicMock()) as MockAzureAI, \
         patch.dict('os.environ', {
             'AZURE_INFERENCE_ENDPOINT': 'https://env.azure.com',
             'AZURE_INFERENCE_CREDENTIAL': 'env-key'
         }):
        mock_instance = MagicMock()
        MockAzureAI.return_value = mock_instance
        result = build_llm(test_config)
        MockAzureAI.assert_called_once_with(
            endpoint="https://env.azure.com",
            credential="env-key",
            model="custom-model",
            temperature=0,
            max_tokens=4096
        )
        assert result == mock_instance

def test_build_llm_anthropic():
    """Test build_llm with Anthropic configuration."""
    test_config = {
        "provider": "anthropic",
        "model": "claude-3-opus",
        "api_key": "test-key"
    }
    
    # Mock the entire langchain_anthropic module
    with patch.dict('sys.modules', {'langchain_anthropic': MagicMock()}):
        # Now patch the specific class
        with patch('langchain_anthropic.ChatAnthropic', MagicMock()) as MockChatAnthropic:
            mock_instance = MagicMock()
            MockChatAnthropic.return_value = mock_instance
            result = build_llm(test_config)
            MockChatAnthropic.assert_called_once_with(
                model="claude-3-opus",
                api_key="test-key",
                temperature=0,
                max_tokens=4096
            )
            assert result == mock_instance

def test_build_llm_missing_provider():
    """Test build_llm with missing provider raises ValueError."""
    test_config = {
        "model": "test-model"
    }
    
    with pytest.raises(ValueError, match="must include a 'provider' field"):
        build_llm(test_config)

def test_build_llm_invalid_provider():
    """Test build_llm with invalid provider raises ValueError."""
    test_config = {
        "provider": "invalid",
        "model": "test-model"
    }
    
    with pytest.raises(ValueError, match="Unknown provider 'invalid'"):
        build_llm(test_config)

def test_build_llm_azure_missing_credentials():
    """Test build_llm with Azure missing credentials raises ValueError."""
    test_config = {
        "provider": "azure",
        "model": "test-model"
    }
    
    with pytest.raises(ValueError, match="Provide endpoint and credential"):
        build_llm(test_config)

def test_build_llm_anthropic_missing_api_key():
    """Test build_llm with Anthropic missing API key raises ValueError."""
    test_config = {
        "provider": "anthropic",
        "model": "test-model"
    }
    
    with patch.dict('sys.modules', {'langchain_anthropic': MagicMock()}):
        with patch('langchain_anthropic.ChatAnthropic', MagicMock()):
            with pytest.raises(ValueError, match="Provide api_key"):
                build_llm(test_config)

def test_build_llm_with_temperature_and_max_tokens():
    """Test build_llm with custom temperature and max_tokens."""
    test_config = {
        "provider": "anthropic",
        "model": "claude-test",
        "api_key": "test-key",
        "temperature": 0.7,
        "max_tokens": 2048
    }
    
    with patch.dict('sys.modules', {'langchain_anthropic': MagicMock()}):
        with patch('langchain_anthropic.ChatAnthropic', MagicMock()) as MockChatAnthropic:
            mock_instance = MagicMock()
            MockChatAnthropic.return_value = mock_instance
            result = build_llm(test_config)
            MockChatAnthropic.assert_called_once_with(
                model="claude-test",
                api_key="test-key",
                temperature=0.7,
                max_tokens=2048
            )
            assert result == mock_instance

def test_build_llm_anthropic_import_error():
    """Test build_llm raises ImportError when package not installed."""
    test_config = {
        "provider": "anthropic",
        "model": "claude-test",
        "api_key": "test-key"
    }
    
    # Mock that langchain_anthropic module doesn't exist
    with patch.dict('sys.modules', {'langchain_anthropic': None}):
        with pytest.raises(ImportError, match="Install the anthropic extra"):
            build_llm(test_config)

def test_build_llm_azure_import_error():
    """Test build_llm raises ImportError when package not installed."""
    # Skip this test if langchain_azure_ai is already installed/imported
    # since we can't test the ImportError in that case
    try:
        import langchain_azure_ai
        pytest.skip("langchain_azure_ai is installed, cannot test ImportError")
    except ImportError:
        pass
    
    test_config = {
        "provider": "azure",
        "model": "test-model",
        "endpoint": "https://test.com",
        "credential": "test-key"
    }
    
    with pytest.raises(ImportError, match="Install the azure extra"):
        build_llm(test_config)
