import math
import os
import torch
from unsloth import FastLanguageModel
from datasets import Dataset, load_dataset
from trl import SFTConfig, SFTTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "google/gemma-3-1b-it"
DATA_PATH = "data/dataset.jsonl"
OUTPUT_DIR = "gemma3-finetuned"
MAX_SEQ_LENGTH = 2048
MAX_SAMPLES = 10000  # 学習に使う最大サンプル数（必要に応じて増減）

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not found. This script requires an NVIDIA GPU for training. "
        "Check: 1) `nvidia-smi` works, 2) PyTorch CUDA build is installed."
    )

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

EXTRA_SPECIAL_TOKENS = [
    "<|start_of_metadata|>",
    "<|end_of_metadata|>",
    "<|start_of_story|>",
    "<|end_of_story|>",
]

model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_ID,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # 自動（4bit量子化なので bf16 コンポーネントを自動選択）
    load_in_4bit=True,  # QLoRA: 4GB VRAM のための 4bit 量子化
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

added = tokenizer.add_special_tokens({"additional_special_tokens": EXTRA_SPECIAL_TOKENS})
print(f"Added {added} special tokens")
if added > 0:
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,  # 0 にすることで Unsloth の高速パッチが全て効く
    bias="none",
    use_gradient_checkpointing="unsloth",  # Unsloth 専用の省メモリ勾配チェックポイント
    random_state=42,
)
model.print_trainable_parameters()

# 5.4GB の JSONL 全体をパースせず、ストリーミングで先頭 MAX_SAMPLES 行だけ読む
stream = load_dataset("json", data_files=DATA_PATH, split="train", streaming=True)
stream = stream.shuffle(seed=42, buffer_size=1000)
dataset = Dataset.from_list(list(stream.take(MAX_SAMPLES)))

train_size = int(0.95 * len(dataset))
train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, len(dataset)))

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

# 総ステップ数から warmup を算出（warmup_ratio は transformers 5.2 で削除予定のため不使用）
per_device_batch_size = 1
gradient_accumulation_steps = 8
num_steps = math.ceil(len(train_dataset) / (per_device_batch_size * gradient_accumulation_steps))
warmup_steps = max(1, int(0.03 * num_steps))

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=per_device_batch_size,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    warmup_steps=warmup_steps,
    lr_scheduler_type="cosine",
    optim="adamw_8bit",
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=1,  # マルチプロセス化はここでは逆に遅い（計測で5倍差を確認済み）
)


def formatting_func(example):
    return example["text"]


trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    formatting_func=formatting_func,
    processing_class=tokenizer,
    data_collator=None,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete!")
