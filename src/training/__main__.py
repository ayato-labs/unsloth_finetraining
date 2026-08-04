import os
import sys

from unsloth import FastLanguageModel  # noqa: I001 - must be imported before transformers/peft

from src.common.config import EVAL_SIZE, GRADIENT_ACCUM_STEPS, MERGED_OUTPUT_DIR, OUTPUT_DIR, TRAIN_BATCH_SIZE, VERSION
from src.common.logger import generate_trace_id, handle_failure, logger, setup_logging, trace_context
from src.common.telemetry import check_gpu_availability
from src.training.dataset import load_dataset_mmap, split_dataset_mmap
from src.training.model import load_base_model, setup_peft_model
from src.training.trainer import build_training_args, cleanup_temp_cache, setup_trainer


def main() -> None:
    trace_id = generate_trace_id()
    setup_logging()
    logger.info("training_start", version=VERSION, trace_id=trace_id)

    try:
        check_gpu_availability(trace_id)
        model, tokenizer = load_base_model(trace_id)
        model = setup_peft_model(model, trace_id)

        dataset = load_dataset_mmap(trace_id)
        train_samples = max(1, len(dataset) - EVAL_SIZE)
        train_dataset, eval_dataset = split_dataset_mmap(dataset, tokenizer, trace_id)

        max_steps = max(1, train_samples // (TRAIN_BATCH_SIZE * GRADIENT_ACCUM_STEPS))

        training_args = build_training_args(max_steps=max_steps)
        trainer = setup_trainer(model, tokenizer, train_dataset, eval_dataset, training_args, trace_id)

        with trace_context(trace_id, "train"):
            checkpoint = None
            if os.path.exists(OUTPUT_DIR):
                checkpoints = [
                    os.path.join(OUTPUT_DIR, d)
                    for d in os.listdir(OUTPUT_DIR)
                    if d.startswith("checkpoint-")
                ]
                if checkpoints:
                    checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                    logger.info("resuming_from_checkpoint", checkpoint=checkpoint)

            trainer.train(resume_from_checkpoint=checkpoint)

        with trace_context(trace_id, "save_model"):
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info("model_saved", output_dir=OUTPUT_DIR)

        with trace_context(trace_id, "save_pretrained_merged"):
            model.save_pretrained_merged(MERGED_OUTPUT_DIR, tokenizer, save_method="merged_16bit")
            logger.info("merged_model_saved", output_dir=MERGED_OUTPUT_DIR)

    except Exception as exc:
        handle_failure("main", exc, trace_id)
        sys.exit(1)
    finally:
        cleanup_temp_cache()

    logger.info("training_complete", trace_id=trace_id)


if __name__ == "__main__":
    main()
