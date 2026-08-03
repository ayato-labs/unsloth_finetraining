import inspect
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextScaledWordEmbedding
from transformers import Gemma4Config

print(inspect.getsource(Gemma4TextScaledWordEmbedding.__init__))
cfg = Gemma4Config.from_pretrained("google/gemma-4-E2B-it")
tc = cfg.text_config
print("text hidden:", tc.hidden_size)
print("text layers:", tc.num_hidden_layers)
print("text vocab:", tc.vocab_size)
print("text hidden_per_layer_input:", getattr(tc, "hidden_size_per_layer_input", "N/A"))
print("text vocab_per_layer_input:", getattr(tc, "vocab_size_per_layer_input", "N/A"))
