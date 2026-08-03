import os
import torch
import torch.nn.functional as F
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

LOG = "device_report.txt"

def log(msg):
    with open(LOG, "a") as f:
        f.write(msg + "\n")
    print(msg)

open(LOG, "w").close()

def patched_forward(self, input_ids):
    if self.weight.device.type == "cpu":
        out = F.embedding(input_ids.cpu(), self.weight, self.padding_idx)
        out = out.to(input_ids.device)
    else:
        out = F.embedding(input_ids, self.weight, self.padding_idx)
    return out * self.embed_scale.to(self.weight.dtype)

Gemma4TextScaledWordEmbedding.forward = patched_forward

EXTRA = ["<|start_of_metadata|>", "<|end_of_metadata|>", "<|start_of_story|>", "<|end_of_story|>"]
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
tokenizer.pad_token = tokenizer.eos_token
added = tokenizer.add_special_tokens({"additional_special_tokens": EXTRA})
log(f"added tokens: {added}")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

log("loading...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-E2B-it",
    quantization_config=bnb_config,
    device_map="auto",
    max_memory={0: "1GB", "cpu": "12GB"},
    torch_dtype=torch.bfloat16,
    attn_implementation="sdpa",
)
log("loaded")

model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
log(f"vocab after resize: {model.config.text_config.vocab_size}")

lm = model.model.language_model
lm.embed_tokens.weight.requires_grad_(False)
lm.embed_tokens_per_layer.weight.requires_grad_(False)

lm.embed_tokens.to("cuda")
lm.layers.to("cuda")
lm.norm.to("cuda")
lm.rotary_emb.to("cuda")
lm.per_layer_model_projection.to("cuda")
lm.per_layer_projection_norm.to("cuda")

log(f"lm_head device: {model.lm_head.weight.device}")

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

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
log("peft model created")

tps = model.print_trainable_parameters()

free, total = torch.cuda.mem_get_info()
log(f"VRAM after setup: used {(total-free)/1024**3:.2f} / {total/1024**3:.2f} GB")

# 2048長でフォワード+backward
ids = torch.randint(0, 260000, (1, 2048), device="cuda")
out = model(input_ids=ids, labels=ids)
log(f"forward+loss OK: {out.loss.item():.4f}")
out.loss.backward()
log("backward OK")

free, total = torch.cuda.mem_get_info()
log(f"VRAM after training step: used {(total-free)/1024**3:.2f} / {total/1024**3:.2f} GB")

n_grad = sum(1 for p in model.parameters() if p.grad is not None)
n_total = sum(1 for p in model.parameters())
log(f"grads: {n_grad}/{n_total}")
