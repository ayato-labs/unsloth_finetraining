import pickle
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Iterator

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it-unsloth-bnb-4bit")
print("tokenizer loaded, eos:", repr(tokenizer.eos_token), "pad:", repr(tokenizer.pad_token))

blob = pickle.dumps(tokenizer)
print("pickle size:", len(blob))
tokenizer2 = pickle.loads(blob)
print("pickle roundtrip ok, eos:", repr(tokenizer2.eos_token))


_WORKER_TOKENIZER = None
_WORKER_EOS_TOKEN = None


def init_tokenizer_worker(tok, eos_token):
    global _WORKER_TOKENIZER, _WORKER_EOS_TOKEN
    _WORKER_TOKENIZER = tok
    _WORKER_EOS_TOKEN = eos_token


def tokenize_text_worker(text):
    if not text.endswith(_WORKER_EOS_TOKEN):
        text += _WORKER_EOS_TOKEN
    return {"input_ids": _WORKER_TOKENIZER(text)["input_ids"]}


def iter_tokenized_window(dataset, tok, num_workers, window_size=16):
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_tokenizer_worker,
        initargs=(tok, tok.eos_token),
    ) as executor:
        window = []
        for example in dataset:
            window.append(example["text"])
            if len(window) >= window_size:
                yield from (f.result() for f in [executor.submit(tokenize_text_worker, t) for t in window])
                window = []
        if window:
            yield from (f.result() for f in [executor.submit(tokenize_text_worker, t) for t in window])


raw = [{"text": f"<|start_of_story|>これは{t}番目の日本語テスト文です。長めの文を書いてみます。<|end_of_story|>"} for t in range(100)]

total = 0
count = 0
for ex in iter_tokenized_window(raw, tokenizer, 4):
    total += len(ex["input_ids"])
    count += 1
print("docs:", count, "tokens:", total, "avg_tokens:", total / count)
print("first example tail:", iter_tokenized_window(raw, tokenizer, 4).__next__()["input_ids"][-3:])