import os
import sys
import torch
from peft import PeftModel
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

from src.common.config import DEFAULT_BASE_MODEL, DEFAULT_CHECKPOINT_DIR, DEFAULT_MERGED_DIR, MAX_SEQ_LENGTH
from src.common.logger import logger, trace_context
from src.common.model_utils import align_model_embeddings


def load_inference_model(checkpoint_path: str, is_merged: bool = False):
    if is_merged:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        base_model, _ = FastLanguageModel.from_pretrained(
            model_name=DEFAULT_BASE_MODEL,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        align_model_embeddings(base_model, tokenizer)
        model = PeftModel.from_pretrained(base_model, checkpoint_path)

    FastLanguageModel.for_inference(model)
    return model, tokenizer


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    repetition_penalty: float = 1.2,
    top_p: float = 0.9,
) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            do_sample=temperature > 0.0,
            use_cache=True,
        )

    prompt_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return response
