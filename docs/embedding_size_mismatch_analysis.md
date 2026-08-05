# embed_tokens / lm_head サイズミスマッチエラーの解析と対処報告書

## 1. エラー概要
推論実行時に以下のエラーが発生し、モデルのロードが失敗していました。

```text
size mismatch for base_model.model.model.embed_tokens.weight: copying a param with shape torch.Size([262152, 1152]) from checkpoint, the shape in current model is torch.Size([262144, 1152]).
size mismatch for base_model.model.lm_head.weight: copying a param with shape torch.Size([262152, 1152]) from checkpoint, the shape in current model is torch.Size([262144, 1152]).
```

---

## 2. 根本原因の解析 (Root Cause)

1. **学習時（`src/training/model.py`）の処理**:
   ファインチューニング時、`load_base_model` 内で以下のコードが実行されていました。
   `model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)`
   これにより、トークン埋め込み層 (`embed_tokens`) および最終出力層 (`lm_head`) のサイズがデフォルトの **262144** から **262152** (8の倍数へパディング) に拡張され、その状態（262152 x 1152）でチェックポイントが保存されました。

2. **推論時（`src/inference/generator.py`）の処理（修正前）**:
   `load_inference_model` においてベースモデルをロードした際、埋め込み層のサイズ拡張（`resize_token_embeddings`）を行わずにデフォルトサイズ **262144** のまま `PeftModel.from_pretrained` を呼び出していました。

3. **ミスマッチの発生**:
   チェックポイントから 262152 サイズの重みをロードしようとしたため、ベースモデルの 262144 サイズと衝突し `size mismatch` エラーが発生しました。

---

## 3. 対処内容

[src/inference/generator.py](file:///wsl.localhost/Ubuntu/home/saiha/unsloth/src/inference/generator.py#L22-L29) の `load_inference_model` 関数において、LoRA重み（PEFTモデル）をロードする前に、ベースモデルのトークン埋め込みサイズを学習時と同様にリサイズする処理を追加しました。

```python
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
base_model, _ = FastLanguageModel.from_pretrained(
    model_name=DEFAULT_BASE_MODEL,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)
# 学習時と同様にトークンエロケーションサイズを拡張
base_model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
model = PeftModel.from_pretrained(base_model, checkpoint_path)
```
