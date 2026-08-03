from train import build_training_args

try:
    cfg = build_training_args()
    print("OK max_length:", getattr(cfg, "max_length", "N/A"))
except Exception as e:
    print("ERR:", type(e).__name__, e)