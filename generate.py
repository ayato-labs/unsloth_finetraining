import argparse
import json
import os
import sys
import uuid
from contextlib import contextmanager
from typing import Optional, Tuple

import torch
from loguru import logger
from transformers import AutoTokenizer
from peft import PeftModel
from unsloth import FastLanguageModel

VERSION = "0.3.0"
DEFAULT_CHECKPOINT_DIR = "gemma3-finetuned"
DEFAULT_MERGED_DIR = "gemma3-merged"
DEFAULT_BASE_MODEL = "unsloth/gemma-3-1b-it-bnb-4bit"
MAX_SEQ_LENGTH = 2048


def generate_trace_id() -> str:
    return str(uuid.uuid4())[:8]


@contextmanager
def trace_context(trace_id: str, step: str):
    logger.info("step_start", step=step, trace_id=trace_id)
    try:
        yield
        logger.info("step_complete", step=step, trace_id=trace_id)
    except Exception as exc:
        handle_failure(step, exc, trace_id)
        raise


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        level="INFO",
    )
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/generate_{time}.log",
        format="{time} | {level} | {message}",
        level="DEBUG",
        rotation="10 MB",
    )


def handle_failure(step: str, error: Exception, trace_id: str) -> None:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "unknown"
    line_number = exc_tb.tb_lineno if exc_tb else 0
    logger.error(
        "step_failed",
        step=step,
        cause=str(error),
        file=file_name,
        line=line_number,
        trace_id=trace_id,
    )


def get_latest_checkpoint(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base directory '{base_dir}' does not exist.")

    checkpoints = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))
    ]

    if not checkpoints:
        logger.warning(f"No checkpoint-* directories found in '{base_dir}'. Using base directory.")
        return base_dir

    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return latest_checkpoint


def load_model_from_merged(merged_path: str, trace_id: str) -> Tuple[object, object]:
    with trace_context(trace_id, "load_model_from_merged"):
        logger.info("loading_merged_model", path=merged_path)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=merged_path,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
        logger.info("merged_model_loaded_successfully", path=merged_path)
        return model, tokenizer


def load_model_from_checkpoint_adapter(checkpoint_path: str, trace_id: str) -> Tuple[object, object]:
    with trace_context(trace_id, "load_model_from_checkpoint_adapter"):
        logger.info("loading_adapter_checkpoint", path=checkpoint_path)

        # adapter_config.json からベースモデル名を特定
        config_path = os.path.join(checkpoint_path, "adapter_config.json")
        base_model_name = DEFAULT_BASE_MODEL
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    base_model_name = cfg.get("base_model_name_or_path", DEFAULT_BASE_MODEL)
            except Exception as e:
                logger.warning(f"failed_to_read_adapter_config: {e}")

        logger.info("loading_base_model", base_model=base_model_name)
        model, _ = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )

        logger.info("loading_tokenizer_from_checkpoint", checkpoint=checkpoint_path)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

        if len(tokenizer) != model.config.vocab_size:
            logger.info("resizing_token_embeddings", orig_size=model.config.vocab_size, new_size=len(tokenizer))
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

        logger.info("applying_peft_adapter", checkpoint=checkpoint_path)
        model = PeftModel.from_pretrained(model, checkpoint_path)

        FastLanguageModel.for_inference(model)
        logger.info("adapter_model_ready_for_inference")
        return model, tokenizer


def generate_text(
    model: object,
    tokenizer: object,
    prompt: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
    trace_id: str = "",
) -> str:
    with trace_context(trace_id, "generate_text"):
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )

        generated_tokens = outputs[0][inputs.input_ids.shape[1] :]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return response


def parse_args():
    parser = argparse.ArgumentParser(description="Text Generation Inference Script using Latest Checkpoint or Merged Model")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR, help="Base directory containing checkpoints")
    parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path")
    parser.add_argument("--merged-dir", type=str, default=None, help="Path to merged model directory (if exists)")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt text")
    parser.add_argument("--max-new-tokens", type=int, default=2048, help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--interactive", action="store_true", help="Run interactive loop prompt")
    return parser.parse_args()


def main() -> None:
    trace_id = generate_trace_id()
    setup_logging()
    logger.info("inference_start", version=VERSION, trace_id=trace_id)

    args = parse_args()

    try:
        # 1. merged ディレクトリが存在する場合は優先使用
        if args.merged_dir and os.path.exists(args.merged_dir):
            model, tokenizer = load_model_from_merged(args.merged_dir, trace_id)
        elif os.path.exists(DEFAULT_MERGED_DIR) and not args.checkpoint:
            logger.info("found_default_merged_dir", path=DEFAULT_MERGED_DIR)
            model, tokenizer = load_model_from_merged(DEFAULT_MERGED_DIR, trace_id)
        else:
            # 2. チェックポイントアダプタからのロード
            checkpoint_path = args.checkpoint if args.checkpoint else get_latest_checkpoint(args.checkpoint_dir)
            model, tokenizer = load_model_from_checkpoint_adapter(checkpoint_path, trace_id)

        if args.interactive or not args.prompt:
            print("\n=== Interactive Inference Mode (type 'exit' or 'quit' to stop) ===")
            while True:
                try:
                    user_prompt = input("\nPrompt > ")
                    if user_prompt.strip().lower() in ["exit", "quit"]:
                        break
                    if not user_prompt.strip():
                        continue

                    response = generate_text(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=user_prompt,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        trace_id=generate_trace_id(),
                    )
                    print(f"\n--- Output ---\n{response}\n--------------")
                except KeyboardInterrupt:
                    print("\nExiting interactive mode...")
                    break
        else:
            response = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                trace_id=trace_id,
            )
            print(f"\nPrompt: {args.prompt}")
            print(f"\n--- Output ---\n{response}\n--------------")

    except Exception as exc:
        handle_failure("main", exc, trace_id)
        sys.exit(1)

    logger.info("inference_complete", trace_id=trace_id)


if __name__ == "__main__":
    main()
