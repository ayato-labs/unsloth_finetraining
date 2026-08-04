import pytest
from unittest.mock import MagicMock, patch

from src.training.model import load_base_model


def test_load_base_model_resizes_embeddings():
    """Verify that load_base_model calls resize_token_embeddings with tokenizer length padded to multiple of 8."""
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.__len__.return_value = 262145

    with patch("src.training.model.FastLanguageModel.from_pretrained", return_value=(mock_model, mock_tokenizer)):
        model, tokenizer = load_base_model(trace_id="test_trace")
        mock_model.resize_token_embeddings.assert_called_once_with(262145, pad_to_multiple_of=8)
        assert model == mock_model
        assert tokenizer == mock_tokenizer
