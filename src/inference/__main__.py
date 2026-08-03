import argparse
import os
import sys

from unsloth import FastLanguageModel  # noqa: I001 - must be imported before transformers/peft

from src.common.config import DEFAULT_CHECKPOINT_DIR, DEFAULT_MERGED_DIR, VERSION
from src.common.logger import generate_trace_id, handle_failure, logger, setup_logging, trace_context
from src.inference.generator import generate_text, load_inference_model


def main():
    parser = argparse.ArgumentParser(description=f"Gemma 3 1B Inference CLI v{VERSION}")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint directory or merged model")
    parser.add_argument("--merged", action="store_true", help="Set this flag if loading a merged model")
    parser.add_argument("--interactive", action="store_true", help="Interactive prompt mode")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum new tokens")
    parser.add_argument("--temp", type=float, default=0.7, help="Generation temperature")
    args = parser.parse_args()

    trace_id = generate_trace_id()
    setup_logging()
    logger.info("inference_start", version=VERSION, trace_id=trace_id)

    checkpoint_path = args.checkpoint
    is_merged = args.merged

    if not checkpoint_path:
        if os.path.exists(DEFAULT_MERGED_DIR):
            checkpoint_path = DEFAULT_MERGED_DIR
            is_merged = True
        elif os.path.exists(DEFAULT_CHECKPOINT_DIR):
            checkpoint_path = DEFAULT_CHECKPOINT_DIR
        else:
            logger.error("No model checkpoint found.")
            sys.exit(1)

    try:
        with trace_context(trace_id, "load_model"):
            model, tokenizer = load_inference_model(checkpoint_path, is_merged=is_merged)

        if args.interactive:
            print("Interactive mode started. Type 'exit' to quit.\n")
            while True:
                prompt = input("Prompt > ")
                if prompt.strip().lower() == "exit":
                    break
                response = generate_text(model, tokenizer, prompt, max_new_tokens=args.max_tokens, temperature=args.temp)
                print(f"\nResponse:\n{response}\n")
        elif args.prompt:
            with trace_context(trace_id, "generate"):
                response = generate_text(model, tokenizer, args.prompt, max_new_tokens=args.max_tokens, temperature=args.temp)
                print(f"\nResponse:\n{response}\n")
        else:
            parser.print_help()

    except Exception as exc:
        handle_failure("main", exc, trace_id)
        sys.exit(1)


if __name__ == "__main__":
    main()
