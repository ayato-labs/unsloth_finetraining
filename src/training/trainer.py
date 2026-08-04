import os
import shutil
import statistics
from collections import deque
from tqdm.auto import tqdm as _base_tqdm
from trl import SFTConfig, SFTTrainer

from src.common.config import (
    EVAL_BATCH_SIZE,
    ETA_WINDOW_FRACTION,
    GRADIENT_ACCUM_STEPS,
    MAX_SEQ_LENGTH,
    OUTPUT_DIR,
    TEMP_CACHE_DIR,
    TRAIN_BATCH_SIZE,
)
from src.common.logger import logger
from src.common.telemetry import TelemetryCallback


class RecentWindowTqdm(_base_tqdm):
    def __init__(self, *args, **kwargs):
        total = kwargs.get("total") or 1
        self._window_size = max(5, int(total * ETA_WINDOW_FRACTION))
        self._step_times = deque(maxlen=self._window_size)
        self._last_n = 0
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        now = self._time()
        if self.last_print_t is not None and n > 0:
            delta_t = now - self.last_print_t
            self._step_times.append(delta_t / n)
        super().update(n)

    @property
    def format_dict(self):
        d = super().format_dict
        if hasattr(self, "_step_times") and len(self._step_times) >= 2:
            avg_step_time = statistics.mean(self._step_times)
            if avg_step_time > 0:
                d["rate"] = 1.0 / avg_step_time
        return d


def build_training_args(max_steps: int = -1):
    return SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        max_steps=max_steps,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        eval_strategy="no",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=False,
        warmup_steps=50,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        report_to="none",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        max_grad_norm=0.3,
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
        ignore_data_skip=True,
        dataset_num_proc=os.cpu_count() or 4,
    )


def setup_trainer(model, tokenizer, train_dataset, eval_dataset, training_args, trace_id: str):
    import transformers.trainer_callback
    transformers.trainer_callback.tqdm = RecentWindowTqdm

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[TelemetryCallback(trace_id=trace_id, interval=20)],
    )
    logger.info("trainer_initialized", trace_id=trace_id)
    return trainer


def cleanup_temp_cache() -> None:
    if os.path.exists(TEMP_CACHE_DIR):
        try:
            shutil.rmtree(TEMP_CACHE_DIR, ignore_errors=True)
            logger.info("temp_cache_cleaned", path=TEMP_CACHE_DIR)
        except Exception as exc:
            logger.warning(f"failed_to_clean_temp_cache: {exc}")
