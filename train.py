import math
import os
import sys
import traceback
import tomllib
import uuid
from contextlib import contextmanager
from functools import wraps

# バージョンを pyproject.toml から読み取り
with open("pyproject.toml", "rb") as f:
    _pyproject = tomllib.load(f)
VERSION = _pyproject["project"]["version"]

# GC が活性化メモリを早期に解放し、ダブルバッファ用の空き VRAM(>512MB) を維持する
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.7"
# ダブルバッファを無効化し、VRAM 逼迫時の逐次化・再試行による不安定化を避ける
os.environ["UNSLOTH_DISABLE_DOUBLE_BUFFER"] = "1"

import subprocess
import torch
import psutil
from loguru import logger
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

MODEL_ID = "google/gemma-3-1b-it"
DATA_PATH = "data/dataset.jsonl"
OUTPUT_DIR = "gemma3-finetuned"
MAX_SEQ_LENGTH = 2048


def setup_logging() -> None:
    """ターミナルには色付きで見やすい表示、ファイルには構造化ログ（JSON）を出力"""
    logger.remove()
    # デフォルトの extra フィールド（trace_id 未指定時用）
    logger.configure(extra={"trace_id": "system"})

    # ターミナル（標準出力）: 人間が見やすいフォーマット（色付き、trace_id表示）
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
    # ファイル出力: 検索・分析用の構造化 JSON 形式
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
    """リクエスト単位の一意な trace_id を生成"""
    return uuid.uuid4().hex[:16]



@contextmanager
def trace_context(trace_id: str, operation: str):
    """処理単位のコンテキスト管理（trace_id と操作名を付与）"""
    with logger.contextualize(trace_id=trace_id, operation=operation):
        logger.info("start")
        try:
            yield
            logger.info("success")
        except Exception as exc:
            logger.exception(f"failure in {operation}")
            raise


def handle_failure(operation: str, exc: Exception, trace_id: str, **context) -> None:
    """統一例外処理: コンテキストとスタックトレースを構造化出力"""
    logger.bind(trace_id=trace_id, operation=operation, **context).exception(
        f"failure in {operation}: {type(exc).__name__}: {exc}"
    )


def get_gpu_telemetry() -> str:
    """nvidia-smi から温度・SMクロック・消費電力・VRAM使用量を取得"""
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
    """一定ステップごとに VRAM / システムRAM / GPU温度・クロックを出力し、
    スローダウンの原因（サーマルスロットリング or メモリ逼迫）を特定する。"""

    def __init__(self, interval: int = 10):
        self.interval = interval

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.global_step % self.interval != 0:
            return
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
    """GPU 利用可能性を確認"""
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not found. This script requires an NVIDIA GPU for training. "
            "Check: 1) `nvidia-smi` works, 2) PyTorch CUDA build is installed."
        )


def load_model_and_tokenizer(trace_id: str):
    """モデルとトークナイザを読み込み"""
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
    """トークナイザの設定と特殊トークン追加"""
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
    """PEFT/LoRA モデルの設定"""
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


def load_dataset_streaming(trace_id: str):
    """学習データをストリーミング（IterableDataset）で読み込み（メモリ消費最小化）"""
    with trace_context(trace_id, "load_dataset_streaming"):
        stream = load_dataset("json", data_files=DATA_PATH, split="train", streaming=True)
        stream = stream.shuffle(seed=42, buffer_size=1000)
        logger.info("dataset_stream_ready")
        return stream


def split_dataset_streaming(stream_dataset, trace_id: str, eval_size: int = 500):
    """take() / skip() によるストリーミング分割"""
    with trace_context(trace_id, "split_dataset_streaming"):
        eval_dataset = stream_dataset.take(eval_size)
        train_dataset = stream_dataset.skip(eval_size)
        logger.info("dataset_split_streaming", eval_samples=eval_size)
        return train_dataset, eval_dataset


def build_training_args(max_steps: int = 1000):
    """SFTConfig 構築（ストリーミング時は max_steps を指定）"""
    warmup_steps = max(1, int(0.03 * max_steps))
    return SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=max_steps,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=False,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
    )


def formatting_func(example):
    return example["text"]


def main() -> None:
    """メイン処理"""
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

        stream_dataset = load_dataset_streaming(trace_id)
        train_dataset, eval_dataset = split_dataset_streaming(stream_dataset, trace_id, eval_size=500)

        # ストリーミング時は max_steps を指定して最適化
        training_args = build_training_args(max_steps=1000)

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=formatting_func,
            processing_class=tokenizer,
            data_collator=None,
            callbacks=[TelemetryCallback(interval=20)],
        )

        with trace_context(trace_id, "training"):
            trainer.train()

        with trace_context(trace_id, "save_model"):
            model.save_pretrained(OUTPUT_DIR)
            tokenizer.save_pretrained(OUTPUT_DIR)
            logger.info("model_saved", output_dir=OUTPUT_DIR)

    except Exception as exc:
        handle_failure("main", exc, trace_id)
        sys.exit(1)

    logger.info("training_complete", trace_id=trace_id)


if __name__ == "__main__":
    main()