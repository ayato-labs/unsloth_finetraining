import inspect
import trl

print("has max_seq_length:", hasattr(trl.SFTConfig, "max_seq_length"))
try:
    cfg = trl.SFTConfig(
        output_dir="x",
        max_seq_length=123,
        max_length=456,
    )
    print("constructed with max_seq_length, val:", getattr(cfg, "max_seq_length", None), "max_length:", cfg.max_length)
except Exception as e:
    print("ERR:", type(e).__name__, e)
