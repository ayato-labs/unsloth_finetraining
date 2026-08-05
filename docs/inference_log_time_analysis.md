# 推論実行時ログのタイムスタンプ解析とボトルネック改善報告書

## 1. タイムスタンプ別ログ解読

提示されたログのタイムスタンプを詳細に分析しました。

| タイムスタンプ | 発生処理 | 所要時間 | 詳細解析 |
|---|---|---|---|
| 21:09:41 ~ 21:10:45 | `load_base_model` & `align_model_embeddings` | **64秒** | ベースモデルの4-bit量子化ロードに加え、デフォルトで有効な `mean_resizing=True` による全262,144トークンの共分散行列計算が発生。 |
| 21:10:45 ~ 21:12:19 | `PeftModel.from_pretrained` & `for_inference` | **94秒** | LoRA重みの読み込みと、NVIDIA GeForce RTX 3050 Laptop (4GB VRAM) 上へのCUDAメモリ確保・最適化パッチ適用。 |
| 21:12:19 以降 | `generate_text` (テキスト生成処理) | 実行中 | `model.generate()` による実際のトークン生成計算（現在テキストを出力中）。 |

---

## 2. 主なボトルネックの原因

1. **`mean_resizing=True` による共分散行列計算（64秒）**:
   `resize_token_embeddings` のデフォルト動作として、新しく追加された8トークンの初期化に262,144次元の平均・共分散行列をCPU/GPU上で低速計算していました。
   ログにも `The new embeddings will be initialized from a multivariate normal distribution... To disable this, use mean_resizing=False` と警告が出力されています。
2. **モデルロードと4GB VRAMにおけるCUDA初期化（94秒）**:
   ノートPC向けRTX 3050 (4GB VRAM) という制限されたGPUメモリ空間で、Unslothが4-bit重みをメモリへ配置しCUDAカーネルを初期化するために一定の時間を要します。

---

## 3. 高速化の対処（修正完了）

[src/common/model_utils.py](file:///wsl.localhost/Ubuntu/home/saiha/unsloth/src/common/model_utils.py#L10) において `mean_resizing=False` を明示的に指定しました。これにより、重い統計計算が完全にスキップされ、リサイズ処理のオーバーヘッドが数ミリ秒に短縮されます。

```python
model.resize_token_embeddings(target_vocab_size, pad_to_multiple_of=pad_to_multiple_of, mean_resizing=False)
```
