import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
assert torch.cuda.is_available(), "CUDA not available"

print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

MODEL_ID = "google/gemma-4-E2B-it"
EXTRA_SPECIAL_TOKENS = [
    "<|start_of_metadata|>", "<|end_of_metadata|>",
    "<|start_of_story|>", "<|end_of_story|>",
]

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
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
print("Model loaded")
if added > 0:
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

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

ids = tokenizer("<|start_of_metadata|>テスト<|end_of_metadata|><|start_of_story|>本文<|end_of_story|>", return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model(**ids)
print("forward OK, logits:", out.logits.shape)

free, total = torch.cuda.mem_get_info()
print(f"VRAM used: {(total-free)/1024**3:.2f} GB / {total/1024**3:.2f} GB")
