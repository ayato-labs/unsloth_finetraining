import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

MODEL_ID = "google/gemma-4-E2B-it"
DATA_PATH = "data/dataset.jsonl"
OUTPUT_DIR = "gemma4-finetuned"
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

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

added = tokenizer.add_special_tokens({"additional_special_tokens": EXTRA_SPECIAL_TOKENS})
print(f"Added {added} special tokens")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation="sdpa",
)

if added > 0:
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

# vision/audioエンコーダはフリーズ（テキスト層のみLoRA適用）
for n, p in model.named_parameters():
    if n.startswith(("model.vision_tower", "model.audio_tower")):
        p.requires_grad = False

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=(
        r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"
        r"|model\.language_model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)"
    ),
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

dataset = load_dataset("json", data_files=DATA_PATH, split=f"train[:{MAX_SAMPLES}]")
dataset = dataset.shuffle(seed=42)

train_size = int(0.95 * len(dataset))
train_dataset = dataset.select(range(train_size))
eval_dataset = dataset.select(range(train_size, len(dataset)))

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples: {len(eval_dataset)}")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
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
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    max_grad_norm=0.3,
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

tokenizer.model_max_length = MAX_SEQ_LENGTH
trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Training complete!")