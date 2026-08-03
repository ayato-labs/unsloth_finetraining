import os
import sys
import tomllib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# pyproject.toml からバージョン読み込み
PYPROJECT_PATH = ROOT_DIR / "pyproject.toml"
if PYPROJECT_PATH.exists():
    with open(PYPROJECT_PATH, "rb") as f:
        _pyproject = tomllib.load(f)
    VERSION = _pyproject["project"]["version"]
else:
    VERSION = "0.1.0"

# CUDAアロケータ環境設定
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
    "expandable_segments:True,"
    "garbage_collection_threshold:0.7,"
    "max_split_size_mb:256"
)

TEMP_CACHE_DIR = os.path.abspath(".cache_temp_datasets")
os.environ["HF_DATASETS_CACHE"] = TEMP_CACHE_DIR

# 各種デフォルトパス・ハイパーパラメータ
MODEL_ID = "google/gemma-3-1b-it"
DEFAULT_BASE_MODEL = "unsloth/gemma-3-1b-it-bnb-4bit"
DATA_PATH = "data/dataset.jsonl"
OUTPUT_DIR = "gemma3-finetuned"
DEFAULT_CHECKPOINT_DIR = OUTPUT_DIR
MERGED_OUTPUT_DIR = "gemma3-merged"
DEFAULT_MERGED_DIR = MERGED_OUTPUT_DIR
MAX_SEQ_LENGTH = 1024

EVAL_SIZE = 500
TOKENIZE_WINDOW_SIZE = 2048
PACKING_MARGIN = 1.02
TRAIN_BATCH_SIZE = 1
EVAL_BATCH_SIZE = 1
GRADIENT_ACCUM_STEPS = 8
ETA_WINDOW_FRACTION = 0.05
