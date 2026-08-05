# uv実行時の仮想環境再構築およびGPU活用に関する詳細調査報告書

## 1. uv run 実行時に仮想環境が毎回再作成される件について

### 原因分析
`uv run` コマンドを実行した際に `Removed virtual environment at: .venv` および `Creating virtual environment at: .venv` が発生している原因は以下の通りです。

1. **ローカルパッケージの変更検知による再ビルド**:
   `pyproject.toml` に `name = "gemma4-fine-tuning"` が定義されているため、`uv` はプロジェクト自体を編集可能パッケージ（またはローカルパッケージ）として管理します。コードやログ、一時ファイルなどの更新により、`uv` がパッケージ構造の変更を検知し、仮想環境の同期（再生成・再ビルド）を自動トリガーしています。
2. **`uv run` のデフォルト同期動作**:
   `uv run` は標準でプロジェクト環境との完全同期をチェックするため、依存関係やビルドメタデータに変更があると判定された場合に `.venv` を再構築します。

---

## 2. 専門家5人による見解

1. **MLOpsエンジニア**:
   「推論や実験のたびに環境が再構築されるのは非常に非効率です。CI/CDパイプラインや本番環境でのレスポンスを著しく悪化させるため改善が必要です。」
2. **Python tooling (uv) 専門家**:
   「`uv run` の自動同期仕様によるものです。環境同期とスクリプト実行を分離し、`uv run --no-sync` を使用するか、構築済みの `.venv/bin/python` を直接呼び出すべきです。」
3. **LLM/Unsloth 専門家**:
   「PyTorch、bitsandbytes、Unsloth などの巨大なCUDAライブラリを毎回再インストール・再配置するのは、キャッシュ効率の観点からも絶対に避けるべきアンチパターンです。」
4. **インフラ/パフォーマンスエンジニア**:
   「無駄なディスクI/OとCPU使用が発生し、開発イテレーション速度を著しく低下させています。」
5. **ソフトウェアアーキテクト**:
   「『環境のプロビジョニング』と『アプリケーションの実行』という関心の分離（Separation of Concerns）ができていません。環境は固定化（Pre-built）し、推論処理のみを迅速に呼び出す構成にすべきです。」

---

## 3. 推論時におけるGPU活用状況

`src/inference/generator.py` のコード内容を確認した結果、**推論処理ではGPU（CUDA）が適切に活用されています**。

- **4-bit CUDA量子化モデルのロード**:
  `FastLanguageModel.from_pretrained(..., load_in_4bit=True)` により、GPU VRAM上に4-bit量子化されたモデルを展開しています。
- **入力テンソルのCUDA転送**:
  `inputs = tokenizer.apply_chat_template(...).to("cuda")` により、プロンプトのトークンIDテンソルを明示的にGPUメモリに転送しています。
- **GPU推論カーネルの実行**:
  `FastLanguageModel.for_inference(model)` を呼ぶことでUnsloth独自の高速CUDAカーネルが適用され、`model.generate()` がGPU上で計算されています。

---

## 4. 改善策・回避コマンド

### 対策1: `--no-sync` フラグの付与
`uv` に仮想環境の再同期を行わせないようにします。
```bash
uv run --no-sync python -m src.inference --checkpoint gemma3-finetuned/checkpoint-16400 --prompt "こんにちは"
```

### 対策2: 仮想環境の直接利用（推奨）
仮想環境をアクティベートして直接 python を実行します。
```bash
source .venv/bin/activate
python -m src.inference --checkpoint gemma3-finetuned/checkpoint-16400 --prompt "こんにちは"
```
