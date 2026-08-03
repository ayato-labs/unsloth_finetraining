from unsloth import FastLanguageModel

from src.common.config import MODEL_ID, MAX_SEQ_LENGTH
from src.common.logger import logger, trace_context


def load_base_model(trace_id: str):
    with trace_context(trace_id, "load_base_model"):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        logger.info("model_loaded", model_id=MODEL_ID, max_seq_length=MAX_SEQ_LENGTH)
        return model, tokenizer


def setup_peft_model(model, trace_id: str):
    with trace_context(trace_id, "setup_peft_model"):
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        model.print_trainable_parameters()
        logger.info("peft_model_ready")
        return model
