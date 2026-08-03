import pytest
from unittest.mock import MagicMock, patch

from src.training.model import setup_peft_model
from src.training.trainer import build_training_args


def test_unsloth_fast_gradient_checkpointing_and_dropout():
    """Unsloth Fast Gradient Checkpointing (use_gradient_checkpointing='unsloth') 及び lora_dropout=0 のテスト"""
    mock_model = MagicMock()
    with patch("src.training.model.FastLanguageModel.get_peft_model") as mock_get_peft:
        setup_peft_model(mock_model, trace_id="test_trace")
        mock_get_peft.assert_called_once()
        _, kwargs = mock_get_peft.call_args
        assert kwargs.get("use_gradient_checkpointing") == "unsloth", "use_gradient_checkpointing must be 'unsloth'"
        assert kwargs.get("lora_dropout") == 0, "lora_dropout must be 0 for Fast Patch"


def test_training_args_packing_and_optimizer():
    """packing=True および optim='adamw_8bit' の設定検証テスト"""
    args = build_training_args()
    assert getattr(args, "packing", False) is True, "packing must be True"
    assert getattr(args, "optim", None) == "adamw_8bit", "optim must be 'adamw_8bit'"
