import os
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from datasets import Features, IterableDataset, Sequence, Value, load_dataset

from src.common.config import DATA_PATH, EVAL_SIZE, MAX_SEQ_LENGTH, TOKENIZE_WINDOW_SIZE
from src.common.logger import logger, trace_context

_WORKER_TOKENIZER = None
_WORKER_EOS_TOKEN = None


def init_tokenizer_worker(tok, eos_token):
    global _WORKER_TOKENIZER, _WORKER_EOS_TOKEN
    _WORKER_TOKENIZER = tok
    _WORKER_EOS_TOKEN = eos_token


def tokenize_text_worker(text: str) -> dict:
    if not text.endswith(_WORKER_EOS_TOKEN):
        text += _WORKER_EOS_TOKEN
    tok = _WORKER_TOKENIZER(text)
    return {
        "input_ids": tok["input_ids"],
        "attention_mask": tok.get("attention_mask", [1] * len(tok["input_ids"])),
        "token_type_ids": tok.get("token_type_ids", [0] * len(tok["input_ids"])),
    }


def iter_tokenized_window(dataset, tok, num_workers: int, window_size: int = TOKENIZE_WINDOW_SIZE) -> Iterator[dict]:
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_tokenizer_worker,
        initargs=(tok, tok.eos_token),
    ) as executor:
        window = []
        for example in dataset:
            window.append(example["text"])
            if len(window) >= window_size:
                futures = [executor.submit(tokenize_text_worker, t) for t in window]
                yield from (f.result() for f in futures)
                window = []
        if window:
            futures = [executor.submit(tokenize_text_worker, t) for t in window]
            yield from (f.result() for f in futures)


def load_dataset_mmap(trace_id: str):
    with trace_context(trace_id, "load_dataset_mmap"):
        dataset = load_dataset("json", data_files=DATA_PATH, split="train")
        logger.info("dataset_mmap_ready", total_samples=len(dataset))
        return dataset


def split_dataset_mmap(dataset, tokenizer, trace_id: str):
    with trace_context(trace_id, "split_dataset_mmap"):
        total_samples = len(dataset)
        train_size = total_samples - EVAL_SIZE

        raw_train = dataset.select(range(train_size))
        raw_eval = dataset.select(range(train_size, total_samples))

        num_workers = max(1, (os.cpu_count() or 4) // 2)
        features = Features({
            "input_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int32")),
            "token_type_ids": Sequence(Value("int32")),
        })

        train_iterable = IterableDataset.from_generator(
            iter_tokenized_window,
            gen_kwargs={
                "dataset": raw_train,
                "tok": tokenizer,
                "num_workers": num_workers,
                "window_size": TOKENIZE_WINDOW_SIZE,
            },
            features=features,
        )

        eval_iterable = IterableDataset.from_generator(
            iter_tokenized_window,
            gen_kwargs={
                "dataset": raw_eval,
                "tok": tokenizer,
                "num_workers": num_workers,
                "window_size": TOKENIZE_WINDOW_SIZE,
            },
            features=features,
        )

        logger.info(
            "dataset_split_complete",
            train_samples=train_size,
            eval_samples=EVAL_SIZE,
            window_size=TOKENIZE_WINDOW_SIZE,
            num_workers=num_workers,
        )

        return train_iterable, eval_iterable
