#!/bin/bash
cd /home/saiha/Gemma4_fine
export PATH=/home/saiha/.local/bin:$PATH
export PYTHONUNBUFFERED=1
uv run python -m train > train_run2.log 2>&1
