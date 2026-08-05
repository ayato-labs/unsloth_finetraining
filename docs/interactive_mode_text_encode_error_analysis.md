# 対話モード時における TextEncodeInput エラーの解析と修正報告書

## 1. エラー概要
対話モード（Interactive mode）にて2回目のプロンプト（`シアーシャとはどんな人？`）を入力した際、以下のエラーが発生してプロセスが停止しました。

```text
2026-08-05 21:21:24 | ERROR | src.common.logger:handle_failure:32 - Failure in main: TextEncodeInput must be Union[TextInputSequence, Tuple[InputSequence, InputSequence]]
```

---

## 2. 根本原因の解析 (Root Cause)

1. **`apply_chat_template` の戻り値構造**:
   [src/inference/generator.py](file:///wsl.localhost/Ubuntu/home/saiha/unsloth/src/inference/generator.py) の `generate_text` 内で `tokenizer.apply_chat_template(..., return_tensors="pt")` を呼び出すと、戻り値は単一の `Tensor` ではなく `input_ids` と `attention_mask` を含む `BatchEncoding` (辞書型オブジェクト) となります。

2. **引数渡しの不備**:
   修正前のコードでは `model.generate(input_ids=inputs, ...)` と記述しており、`input_ids` パラメータに辞書型オブジェクト (`inputs`) がそのまま渡されていました。
   Hugging Face の `model.generate` 内部で `input_ids` が Tensor でないと判定された場合、内部でエンコード関数（`tokenizer.encode`）へ再転送を試みますが、辞書型を文字列としてエンコードしようとしたことで `TypeError: TextEncodeInput must be Union[...]` が発生しました。

3. **`attention_mask` の欠落警告**:
   `input_ids=inputs` として辞書を渡していたため、`attention_mask` が `model.generate` に正しく伝わらず、`The attention mask is not set and cannot be inferred from input...` という警告が毎回発生する原因にもなっていました。

---

## 3. 対処内容

[src/inference/generator.py](file:///wsl.localhost/Ubuntu/home/saiha/unsloth/src/inference/generator.py#L36-L55) の `generate_text` を以下のように修正しました。

1. `return_dict=True` を明示的に指定。
2. `model.generate(**inputs, ...)` として `input_ids` と `attention_mask` をキーワード展開して渡すように変更。
3. トークン長計算部を `inputs["input_ids"].shape[1]` に変更。

```python
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.7) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            use_cache=True,
        )

    prompt_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)
    return response
```
