import math
import os
import sys
import tomllib
import uuid
import shutil
import subprocess
import statistics
from collections import deque
from contextlib import contextmanager
import torch
import psutil
from loguru import logger
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer
from tqdm.auto import tqdm as _base_tqdm

# バージョンを pyproject.toml から読み取り
with open("pyproject.toml", "rb") as f:
    _pyproject = tomllib.load(f)
VERSION = _pyproject["project"]["version"]

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"
    "garbage_collection_threshold:0.7,"
    "max_split_size_mb:256"
)

TEMP_CACHE_DIR = os.path.abspath(".cache_temp_datasets")
os.environ["HF_DATASETS_CACHE"] = TEMP_CACHE_DIR

MODEL_ID = "google/gemma-3-1b-it"
DATA_PATH = "data/dataset.jsonl"
OUTPUT_DIR = "gemma3-finetuned"
MERGED_OUTPUT_DIR = "gemma3-merged"
MAX_SEQ_LENGTH = 1024

ETA_WINDOW_FRACTION = 0.05
ETA_USE_MEDIAN = False


class RecentWindowTqdm(_base_tqdm):
    def __init__(self, *args, **kwargs):
        self._step_times: deque[float] = deque()
        self._last_update_at: float | None = None
        self._median = statistics.median
        self._mean = statistics.mean
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        now = self._time()
        if self._last_update_at is not None and self.n > self.initial:
            dt = now - self._last_update_at
            if dt > 0:
                window = max(1, int(self.total * ETA_WINDOW_FRACTION)) if self.total else 100
                self._step_times.append(dt / max(n, 1))
                while len(self._step_times) > window:
                    self._step_times.popleft()
        self._last_update_at = now
        return super().update(n)

    @property
    def format_dict(self):
        d = super().format_dict
        if self._step_times:
            recent = self._median(self._step_times) if ETA_USE_MEDIAN else self._mean(self._step_times)
            if recent > 0:
                d["rate"] = 1.0 / recent
        return d


def patch_progress_bar() -> None:
    import transformers.trainer_callback as trainer_callback
    trainer_callback.tqdm = RecentWindowTqdm
    logger.info("progress_bar_patched", window_fraction=ETA_WINDOW_FRACTION, use_median=ETA_USE_MEDIAN)


def setup_logging() -> None:
    os.makedirs("logs", exist_ok=True)
    logger.remove()
    logger.configure(extra={"trace_id": "system"})

    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>trace_id={extra[trace_id]}</cyan> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )
    logger.add(
        sys.stdout,
        format=console_format,
        level="INFO",
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    logger.add(
        "logs/train_{time:YYYYMMDD}.log",
        format="{message}",
        serialize=True,
        level="INFO",
        rotation="100 MB",
        retention="7 days",
        compression="gz",
        backtrace=True,
        diagnose=True,
    )


def generate_trace_id() -> str:
    return uuid.uuid4().hex[:16]


@contextmanager
def trace_context(trace_id: str, operation: str):
    with logger.contextualize(trace_id=trace_id, operation=operation):
        logger.info("start")
        try:
            yield
            logger.info("success")
        except Exception as exc:
            logger.exception(f"failure in {operation}")
            raise


def handle_failure(operation: str, exc: Exception, trace_id: str, **context) -> None:
    logger.bind(trace_id=trace_id, operation=operation, **context).exception(
        f"failure in {operation}: {type(exc).__name__}: {exc}"
    )


def get_gpu_telemetry() -> str:
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=temperature.gpu,clocks.sm,power.draw,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip()
    except Exception as exc:
        handle_failure("get_gpu_telemetry", exc, generate_trace_id())
        return "nvidia-smi unavailable"


class TelemetryCallback(TrainerCallback):
    def __init__(self, interval: int = 10, defrag_interval: int = 500):
        self.interval = interval
        self.defrag_interval = defrag_interval

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.interval != 0:
            return

        if state.global_step > 0 and state.global_step % self.defrag_interval == 0:
            torch.cuda.empty_cache()
            logger.info("vram_defrag", step=state.global_step)

        vram_free_mib = torch.cuda.mem_get_info(0)[0] / 1024**2
        vram_total_mib = torch.cuda.mem_get_info(0)[1] / 1024**2
        ram = psutil.virtual_memory()
        logger.info(
            "telemetry",
            step=state.global_step,
            vram_used_mib=round(vram_total_mib - vram_free_mib),
            vram_total_mib=round(vram_total_mib),
            ram_used_gb=round(ram.used / 1024**3, 1),
            ram_total_gb=round(ram.total / 1024**3, 1),
            gpu_info=get_gpu_telemetry(),
        )


def check_gpu_availability() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. This script requires an NVIDIA GPU for training."
        )


def load_model_and_tokenizer(trace_id: str):
    with trace_context(trace_id, "load_model_and_tokenizer"):
        model, tokenizer = FastLanguageModel.from_pretrained(
            MODEL_ID,
            max_seq_length=MAX_SEQ_LENGTH,
            dtype=None,
            load_in_4bit=True,
        )
        logger.info("model_loaded", model_id=MODEL_ID, max_seq_length=MAX_SEQ_LENGTH)
        return model, tokenizer


def prepare_tokenizer(model, tokenizer, trace_id: str):
    with trace_context(trace_id, "prepare_tokenizer"):
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        EXTRA_SPECIAL_TOKENS = [
            "<|start_of_metadata|>",
            "<|end_of_metadata|>",
            "<|start_of_story|>",
            "<|end_of_story|>",
        ]
        added = tokenizer.add_special_tokens({"additional_special_tokens": EXTRA_SPECIAL_TOKENS})
        logger.info("special_tokens_added", count=added)
        if added > 0:
            model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
            logger.info("token_embeddings_resized", vocab_size=len(tokenizer))
        return model, tokenizer


def setup_peft_model(model, trace_id: str):
    with trace_context(trace_id, "setup_peft_model"):
        model = FastLanguageModel.get_peft_model(
            model,
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            modules_to_save=["embed_tokens", "lm_head"],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        model.print_trainable_parameters()
        logger.info("peft_model_ready")
        return model


def load_dataset_mmap(trace_id: str):
    with trace_context(trace_id, "load_dataset_mmap"):
        dataset = load_dataset("json", data_files=DATA_PATH, split="train")
        logger.info("dataset_mmap_ready", total_samples=len(dataset))
        return dataset


def split_dataset_mmap(dataset, trace_id: str, eval_size: int = 500):
    with trace_context(trace_id, "split_dataset_mmap"):
        train_size = len(dataset) - eval_size
        train_dataset = dataset.select(range(train_size))
        eval_dataset = dataset.select(range(train_size, len(dataset)))
        logger.info("dataset_split", train_samples=len(train_dataset), eval_samples=len(eval_dataset))
        return train_dataset, eval_dataset


def build_training_args():
    return SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=False,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=os.cpu_count() or 4,
    )


def cleanup_temp_cache() -> None:
    if os.path.exists(TEMP_CACHE_DIR):
        try:
            shutil.rmtree(TEMP_CACHE_DIR, ignore_errors=True)
            logger.info("temp_cache_cleaned", path=TEMP_CACHE_DIR)
        except Exception as exc:
            logger.warning(f"failed_to_clean_temp_cache: {exc}")


def main() -> None:
    trace_id = generate_trace_id()
    setup_logging()

    logger.info("training_start", version=VERSION, trace_id=trace_id)

    with trace_context(trace_id, "check_gpu"):
        check_gpu_availability()
        logger.info("gpu_info", name=torch.cuda.get_device_name(0), vram_gb=round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1))

    try:
        model, tokenizer = load_model_and_tokenizer(trace_id)
        model, tokenizer = prepare_tokenizer(model, tokenizer, trace_id)
        model = setup_peft_model(model, trace_id)

        dataset = load_dataset_mmap(trace_id)
        train_dataset, eval_dataset = split_dataset_mmap(dataset, trace_id, eval_size=500)

        training_args = build_training_args()

        patch_progress_bar()

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_num_proc=training_args.dataset_num_proc,
            processing_class=tokenizer,
            callbacks=[TelemetryCallback(interval=20)],
        )

        with trace_context(trace_id, "training"):
            checkpoint = None
            if os.path.exists(OUTPUT_DIR):
                checkpoints = [
                    os.path.join(OUTPUT_DIR, d)
                    for d in os.listdir(OUTPUT_DIR)
                    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
                ]
                if checkpoints:
                    checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
                    logger.info("resuming_from_checkpoint", checkpoint=checkpoint)

            trainer.train(resume_from_checkpoint=checkpoint)

        with trace_context(trace_id, "save_model"):
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info("model_saved", output_dir=OUTPUT_DIR)

        # Unsloth 推奨: マージ済みモデルの自動保存 (16-bit / 4-bit merged)
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
