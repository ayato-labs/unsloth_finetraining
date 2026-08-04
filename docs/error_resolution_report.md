# Error Resolution Report

## 1. Overview
This report details the root cause analysis and resolution for the state_dict size mismatch error and the `RecentWindowTqdm` TypeError encountered during training.

## 2. Issue 1: State Dict Size Mismatch Error
```text
2026-08-04 21:39:17 | ERROR | src.common.logger:handle_failure:32 - Failure in train: Error(s) in loading state_dict for PeftModelForCausalLM:
        size mismatch for base_model.model.model.embed_tokens.weight: copying a param with shape torch.Size([262152, 1152]) from checkpoint, the shape in current model is torch.Size([262144, 1152]).
        size mismatch for base_model.model.lm_head.weight: copying a param with shape torch.Size([262152, 1152]) from checkpoint, the shape in current model is torch.Size([262144, 1152]).
```
### Resolution:
Added explicit embedding resizing during base model loading in [src/training/model.py](file:///home/saiha/unsloth/src/training/model.py#L15):
```python
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
```

---

## 3. Issue 2: tqdm format_meter TypeError
```text
2026-08-04 21:47:20 | ERROR | src.common.logger:handle_failure:32 - Failure in train: tqdm.std.tqdm.format_meter() argument after ** must be a mapping, not method
2026-08-04 21:47:20 | ERROR | src.common.logger:handle_failure:32 - Failure in main: tqdm.std.tqdm.format_meter() argument after ** must be a mapping, not method
```

### Root Cause Analysis:
In `tqdm.std.tqdm`, `format_dict` is defined as a `@property`. When `RecentWindowTqdm` in `src/training/trainer.py` overrode `format_dict` as a standard method without the `@property` decorator (`def format_dict(self):`), `tqdm`'s internal call `format_meter(**self.format_dict)` passed the bound method object rather than the dictionary mapping, raising a `TypeError`. Furthermore, attributes were initialized after `super().__init__()`, missing the initial refresh call during `__init__`.

### Resolution:
1. Added `@property` decorator to `format_dict` in [src/training/trainer.py](file:///home/saiha/unsloth/src/training/trainer.py#L36).
2. Moved attribute initialization before `super().__init__()` and added a defensive `hasattr` check.

---

## 4. Verification
Ran pytest test suite:
- `test/test_model_resize.py`: PASSED
- `test/test_unsloth_optimization.py`: PASSED (2 tests)
- `test/test_tqdm.py`: PASSED
Total: 4 passed.
