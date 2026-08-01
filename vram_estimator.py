"""Unified VRAM Estimation for LLM Training

全推定ロジックを一元管理。全 caller はこのモジュールのみを参照する。

Liger FusedLinearCrossEntropy の chunked backward ワークスペース
(logits_chunk + grad_weight accumulator) もピーク見積もりに含めることで、
WSL2 + Triton 環境での VRAM逼迫・TDR/device-not-ready を防止する。

実測値 (VramMeasurementTracker) を優先し、実測できない場合のみ計算式を使用する。

auto_calibrate():
    Chinchilla 計算時に自動呼び出し。対象アーキテクチャのモデルを GPU 上に構築し、
    measure_training_vram() を実行して vram_calibration.json を生成する。
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass
from typing import Literal


def detect_vram() -> float:
    """GPU VRAM容量を検出（GB単位）。

    torch.cuda が利用可能な場合は物理VRAMを返し、不可の場合は 4.0 GB をフォールバックとする。
    """
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    except Exception:
        pass
    return 4.0


Precision = Literal["bf16", "fp16", "fp32"]
OptimizerType = Literal["adamw_bnb_8bit", "paged_adamw", "adamw_torch_fused"]
Checkpointing = Literal["full", "selective", "none"]


_CALIBRATION_FILE = "vram_calibration.json"


@dataclass(frozen=True)
class VramConfig:
    n_params: int = 150_000_000
    hidden_size: int = 768
    intermediate_size: int = 0  # 0 → auto (4 * hidden_size)
    num_layers: int = 12
    vocab_size: int = 32000

    micro_batch_size: int = 1
    seq_len: int = 1024

    precision: Precision = "bf16"
    optimizer_type: OptimizerType = "adamw_bnb_8bit"
    checkpointing: Checkpointing = "full"

    use_liger_kernel: bool = True
    torch_compile: bool = False

    total_vram_gb: float = 4.0
    allocator_factor: float = 1.35


@dataclass(frozen=True)
class VramBreakdown:
    weights_gb: float
    gradients_gb: float
    optimizer_states_gb: float
    activations_gb: float
    liger_flce_workspace_gb: float
    cuda_context_gb: float
    compile_overhead_gb: float
    fragmentation_gb: float

    total_estimated_gb: float
    vram_limit_gb: float
    util_pct: float
    is_safe: bool

    def log_string(self) -> str:
        lines = [
            f"  - Physical GPU VRAM Limit: {self.vram_limit_gb:.2f} GB",
            f"  - Model Weights: {self.weights_gb:.4f} GB",
            f"  - Gradients: {self.gradients_gb:.4f} GB",
            f"  - Optimizer States: {self.optimizer_states_gb:.4f} GB",
            f"  - Activations: {self.activations_gb:.4f} GB",
            f"  - Liger FLCE Workspace: {self.liger_flce_workspace_gb:.4f} GB",
            f"  - CUDA Context & OS: {self.cuda_context_gb:.4f} GB",
            f"  - Compiler Overhead: {self.compile_overhead_gb:.4f} GB",
            f"  - Fragmentation ({'%.0f' % ((self.total_estimated_gb / max(self.total_estimated_gb - self.fragmentation_gb, 1e-9) - 1) * 100)}%): {self.fragmentation_gb:.4f} GB",
            f"  - Estimated Peak VRAM: {self.total_estimated_gb:.2f} GB / {self.vram_limit_gb:.2f} GB ({self.util_pct:.1f}%)",
            f"  - Safety: {'OK' if self.is_safe else 'DANGER - exceeds VRAM limit'}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class VramEstimate:
    """estimate_training_vram の戻り値。内訳＋バッチ分割計算用の数値を含む。"""

    breakdown: VramBreakdown
    fixed_reserved_gb: float
    activation_per_sample_gb: float

    @property
    def available_for_activations_gb(self) -> float:
        return max(0.0, self.breakdown.vram_limit_gb - self.fixed_reserved_gb)

    @property
    def max_safe_micro_batch(self) -> int:
        if self.activation_per_sample_gb <= 1e-12:
            return 0
        return max(1, int(self.available_for_activations_gb / self.activation_per_sample_gb))


# ====================================================================
#  Measurement & Calibration Utilities
# ====================================================================


@dataclass
class VramSnapshot:
    """現在のCUDAメモリ状態のスナップショット"""

    allocated_bytes: int = 0
    reserved_bytes: int = 0
    peak_allocated_bytes: int = 0


def _take_snapshot() -> VramSnapshot:
    try:
        import torch

        return VramSnapshot(
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
    except (ImportError, RuntimeError, AssertionError):
        return VramSnapshot()


def _reset_peak_stats() -> None:
    try:
        import torch

        torch.cuda.reset_peak_memory_stats()
    except (ImportError, RuntimeError):
        pass


class VramMeasurementTracker:
    """特定処理の前後でVRAM増分を実測するコンテキストマネージャ。

    使い方:
      with VramMeasurementTracker("forward") as t:
          outputs = model(input_ids)
      print(t.delta_mb, t.peak_mb)
    """

    def __init__(self, label: str):
        self.label = label
        self.before: VramSnapshot = VramSnapshot()
        self.after: VramSnapshot = VramSnapshot()

    @property
    def delta_bytes(self) -> int:
        return self.after.allocated_bytes - self.before.allocated_bytes

    @property
    def delta_mb(self) -> float:
        return self.delta_bytes / (1024**2)

    @property
    def delta_gb(self) -> float:
        return self.delta_bytes / (1024**3)

    @property
    def peak_bytes(self) -> int:
        return self.after.peak_allocated_bytes - self.before.allocated_bytes

    @property
    def peak_mb(self) -> float:
        return self.peak_bytes / (1024**2)

    def __enter__(self) -> VramMeasurementTracker:
        _reset_peak_stats()
        self.before = _take_snapshot()
        return self

    def __exit__(self, *args) -> None:
        self.after = _take_snapshot()


@dataclass
class VramCalibration:
    """実測にもとづくキャリブレーションデータ。

    allocator_factor:  (実際の reserved ) / (生の合計)
    activation_per_sample_gb:  forward増分 / micro_batch_size
    cuda_context_gb: baseline 時点の allocated  から逆算
    flce_peak_gb: backward 時のピーク増分
    compile_overhead_gb: compile 有効時の追加増分
    """

    allocator_factor: float = 1.35
    activation_per_sample_gb: float = 0.0
    cuda_context_gb: float = 0.7
    flce_peak_gb: float = 0.0
    compile_overhead_gb: float = 0.0

    model_name: str = ""
    precision: str = "bf16"
    optimizer_type: str = "adamw_bnb_8bit"
    seq_len: int = 1024
    micro_batch_size: int = 1
    n_params: int = 0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> VramCalibration:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def save(self, path: str = _CALIBRATION_FILE) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str = _CALIBRATION_FILE) -> VramCalibration | None:
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                return cls.from_dict(json.load(f))
        except (json.JSONDecodeError, KeyError):
            return None


def measure_training_vram(
    model,
    optimizer,
    input_ids,
    labels,
    micro_batch_size: int,
    seq_len: int,
    n_params: int,
    precision: str = "bf16",
    optimizer_type: str = "adamw_bnb_8bit",
    use_liger_kernel: bool = True,
    torch_compile: bool = False,
    label: str = "default",
    n_samples: int = 5,
    warmup: bool = True,
    agg: str = "median",
) -> VramCalibration:
    """n_samples回の実測をとり、中央値/平均値をキャリブレーションとして返す。

    warmup=True で最初に1ステップ捨てる (CUDA cache/compiler を温める)。
    agg="median" で中央値、 "average" で平均値。
    """
    from src.common.logger import logger

    try:
        import torch

        if not torch.cuda.is_available():
            return VramCalibration(n_params=n_params)
    except ImportError:
        return VramCalibration(n_params=n_params)

    # ---- warmup (キャッシュ・コンパイラを温める) ----
    if warmup:
        optimizer.zero_grad()
        with torch.no_grad():
            _ = model(input_ids=input_ids, labels=labels)
        # dummy backward + step to ensure optimizer state is settled
        dummy_loss = model(input_ids=input_ids, labels=labels).loss
        dummy_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # ---- baseline (warmup後の安定した状態) ----
    _reset_peak_stats()
    baseline = _take_snapshot()
    baseline_gb = baseline.allocated_bytes / (1024**3)
    af = baseline.reserved_bytes / max(baseline.allocated_bytes, 1)

    # ---- n_samples回の実測 ----
    act_deltas: list[float] = []
    peak_deltas: list[float] = []
    peak_afs: list[float] = []

    for i in range(n_samples):
        optimizer.zero_grad()
        _reset_peak_stats()
        before = _take_snapshot()

        # forward
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        after_forward = _take_snapshot()
        act_delta = (after_forward.allocated_bytes - before.allocated_bytes) / (1024**3)

        # backward
        loss.backward()
        after_backward = _take_snapshot()
        bwd_peak = (after_backward.peak_allocated_bytes - before.allocated_bytes) / (1024**3)

        # optimizer step (update model state, not measured)
        optimizer.step()

        act_deltas.append(act_delta)
        peak_deltas.append(bwd_peak)
        peak_afs.append(after_backward.reserved_bytes / max(after_backward.allocated_bytes, 1))

        logger.info(
            f"  VRAM Sample [{label} #{i + 1}]: "
            f"fwd_delta={act_delta:.6f} GB, "
            f"bwd_peak={bwd_peak:.6f} GB, "
            f"af_peak={peak_afs[-1]:.4f}"
        )

    # ---- 集約 ----
    if agg == "median":
        act_agg = statistics.median(act_deltas)
        peak_agg = statistics.median(peak_deltas)
        af_agg = statistics.median(peak_afs)
    else:
        act_agg = sum(act_deltas) / len(act_deltas)
        peak_agg = sum(peak_deltas) / len(peak_deltas)
        af_agg = sum(peak_afs) / len(peak_afs)

    act_per_sample_gb = act_agg / max(micro_batch_size, 1)

    # cuda context (baselineから逆算)
    bpp = 2 if precision in ("bf16", "fp16") else 4
    weights_gb = n_params * bpp / (1024**3)
    grads_gb = n_params * bpp / (1024**3)
    if "8bit" in optimizer_type:
        optim_gb = n_params * 2 / (1024**3)
    elif "paged" in optimizer_type:
        optim_gb = n_params * 4 / (1024**3)
    else:
        optim_gb = n_params * 8 / (1024**3)
    estimated_fixed_gb = weights_gb + grads_gb + optim_gb
    measured_cuda_gb = max(0.0, baseline_gb - estimated_fixed_gb)

    # flce peak = backward peak delta - activation delta
    flce_peak_gb = max(0.0, peak_agg - act_agg)

    logger.info(
        f"VRAM Measurement [{label}] ({agg.upper()} of {n_samples} samples):\n"
        f"  Baseline: {baseline_gb:.4f} GB\n"
        f"  Activation/sample: {act_per_sample_gb:.6f} GB\n"
        f"  Backward peak delta: {peak_agg:.6f} GB\n"
        f"  FLCE workspace: {flce_peak_gb:.6f} GB\n"
        f"  CUDA context: {measured_cuda_gb:.4f} GB\n"
        f"  Allocator factor (baseline): {af:.4f}\n"
        f"  Allocator factor (peak median): {af_agg:.4f}"
    )

    return VramCalibration(
        allocator_factor=round(af_agg, 4),
        activation_per_sample_gb=round(act_per_sample_gb, 6),
        cuda_context_gb=round(measured_cuda_gb, 4),
        flce_peak_gb=round(flce_peak_gb, 6),
        compile_overhead_gb=0.0,
        model_name=label,
        precision=precision,
        optimizer_type=optimizer_type,
        seq_len=seq_len,
        micro_batch_size=micro_batch_size,
        n_params=n_params,
    )


def _liger_flce_chunk_size(
    micro_batch_size: int,
    seq_len: int,
    vocab_size: int,
    hidden_size: int,
) -> int:
    """Liger FusedLinearCrossEntropy の内部チャンク分割式を再現し、
    1チャンクあたりのトークン数を返す。

    式:
      inc_factor = ceil(V / H)
      chunk_size = next_power_of_2(ceil(BT / inc_factor))
      MAX_FUSED_SIZE = 32768 (non-NPU) によるキャップ
    """
    BT = micro_batch_size * seq_len
    if BT <= 0:
        return 0
    inc_factor = max(1, (vocab_size + hidden_size - 1) // hidden_size)
    target_chunk = (BT + inc_factor - 1) // inc_factor
    next_pow2 = 1
    while next_pow2 < target_chunk:
        next_pow2 <<= 1
    return min(next_pow2, 32768)


def estimate_liger_flce_peak_gb(
    micro_batch_size: int,
    seq_len: int,
    vocab_size: int,
    hidden_size: int,
) -> float:
    """Liger FLCE chunked backward のピーク追加メモリ (GB) を算出。

    logits_chunk (fp32): [chunk_size, V] — 各チャンクで一時確保
    grad_weight (fp32): [V, H] — 全チャンクで累積保持
    """
    chunk_size = _liger_flce_chunk_size(micro_batch_size, seq_len, vocab_size, hidden_size)
    if chunk_size <= 0:
        return 0.0
    logits_chunk_gb = (chunk_size * vocab_size * 4.0) / (1024**3)
    grad_weight_gb = (vocab_size * hidden_size * 4.0) / (1024**3)
    return round(logits_chunk_gb + grad_weight_gb, 6)


def _activation_bytes_per_sample(
    seq_len: int,
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    bytes_per_param: int,
    checkpointing: str,
) -> float:
    """Gradient Checkpointing モードに応じた1サンプルあたりの活性化メモリ (bytes)。

    "none": 全活性値を保持 (no checkpointing)
    "selective": Attention出力のみ保持 (selective checkpointing)
    "full": ほぼ保持しない (full gradient checkpointing)
    """
    inter = intermediate_size if intermediate_size > 0 else 4 * hidden_size
    if checkpointing == "none":
        # 各層: attention(12×hidden) + MLP(5×inter) + residual/norms
        per_layer = 12 * hidden_size + 5 * inter
        return bytes_per_param * seq_len * (per_layer * num_layers + hidden_size)
    elif checkpointing == "selective":
        return bytes_per_param * seq_len * 2 * hidden_size * num_layers
    else:
        # "full": checkpoint の入力のみ (hidden × layers)
        return bytes_per_param * hidden_size * num_layers * 2


def _calc_wsl_overhead_gb() -> float:
    return 0.0  # WSL-specific overhead removed; formula-based estimate used instead


def estimate_training_vram_with_calibration(
    config: VramConfig,
    calibration: VramCalibration | None = None,
) -> VramEstimate:
    """実測キャリブレーション値を反映した推定。

    実測値があればそちらを優先し、不足分のみ計算式で補完する。
    """
    if calibration is None:
        calibration = VramCalibration.load()
    if calibration is None:
        return estimate_training_vram(config)

    bpp = 2 if config.precision in ("bf16", "fp16") else 4

    weights_gb = (config.n_params * bpp) / (1024**3)
    gradients_gb = (config.n_params * bpp) / (1024**3)

    if "8bit" in config.optimizer_type:
        optim_bytes = 2
    elif "paged" in config.optimizer_type:
        optim_bytes = 4
    else:
        optim_bytes = 8
    optimizer_gb = (config.n_params * optim_bytes) / (1024**3)

    cuda_context_gb = (
        calibration.cuda_context_gb
        if calibration.cuda_context_gb > 0
        else 0.7 + _calc_wsl_overhead_gb()
    )

    compile_gb = 0.0
    if config.torch_compile:
        compile_gb = 1.0
        if config.use_liger_kernel:
            compile_gb += 0.5

    raw_activation_per_sample_gb = (
        _activation_bytes_per_sample(
            config.seq_len,
            config.hidden_size,
            config.intermediate_size,
            config.num_layers,
            bpp,
            config.checkpointing,
        )
    ) / (1024**3)

    liger_flce_gb = (
        estimate_liger_flce_peak_gb(
            config.micro_batch_size,
            config.seq_len,
            config.vocab_size,
            config.hidden_size,
        )
        if config.use_liger_kernel
        else 0.0
    )

    af = (
        max(calibration.allocator_factor, 1.0)
        if calibration.allocator_factor > 0
        else config.allocator_factor
    )
    sum_before_frag = (
        weights_gb
        + gradients_gb
        + optimizer_gb
        + raw_activation_per_sample_gb * config.micro_batch_size
        + liger_flce_gb
        + cuda_context_gb
        + compile_gb
    )
    total_estimated_gb = sum_before_frag * af
    fragmentation_gb = total_estimated_gb - sum_before_frag

    fixed_before_frag = (
        weights_gb + gradients_gb + optimizer_gb + liger_flce_gb + cuda_context_gb + compile_gb
    )
    fixed_reserved_gb = fixed_before_frag * af
    activation_per_sample_gb = raw_activation_per_sample_gb * af

    vram_limit = config.total_vram_gb
    util_pct = (total_estimated_gb / vram_limit * 100) if vram_limit > 0 else 0.0
    is_safe = total_estimated_gb <= vram_limit

    breakdown = VramBreakdown(
        weights_gb=round(weights_gb, 4),
        gradients_gb=round(gradients_gb, 4),
        optimizer_states_gb=round(optimizer_gb, 4),
        activations_gb=round(raw_activation_per_sample_gb * config.micro_batch_size, 4),
        liger_flce_workspace_gb=round(liger_flce_gb, 4),
        cuda_context_gb=round(cuda_context_gb, 4),
        compile_overhead_gb=round(compile_gb, 4),
        fragmentation_gb=round(fragmentation_gb, 4),
        total_estimated_gb=round(total_estimated_gb, 4),
        vram_limit_gb=round(vram_limit, 2),
        util_pct=round(util_pct, 1),
        is_safe=is_safe,
    )
    return VramEstimate(
        breakdown=breakdown,
        fixed_reserved_gb=round(fixed_reserved_gb, 6),
        activation_per_sample_gb=round(activation_per_sample_gb, 6),
    )


def estimate_training_vram(config: VramConfig) -> VramEstimate:
    """統一VRAM推定エンジン。

    モデル重み・勾配・オプティマイザ・活性値・Liger FLCE ワークスペース・
    CUDAコンテキスト・コンパイラオーバーヘッド・アロケータ断片化をすべて考慮し、
    バッチ分割に必要な数値も併せて返す。
    """
    bytes_per_param = 2 if config.precision in ("bf16", "fp16") else 4

    # ---- Base components (independent of micro_batch_size) ----
    weights_gb = (config.n_params * bytes_per_param) / (1024**3)
    gradients_gb = (config.n_params * bytes_per_param) / (1024**3)

    if "8bit" in config.optimizer_type:
        optim_bytes = 2
    elif "paged" in config.optimizer_type:
        optim_bytes = 4
    else:
        optim_bytes = 8
    optimizer_gb = (config.n_params * optim_bytes) / (1024**3)

    cuda_context_gb = 0.7 + _calc_wsl_overhead_gb()

    compile_gb = 0.0
    if config.torch_compile:
        compile_gb = 1.0
        if config.use_liger_kernel:
            compile_gb += 0.5

    # ---- Components that depend on micro_batch_size ----
    raw_activation_per_sample_gb = (
        _activation_bytes_per_sample(
            config.seq_len,
            config.hidden_size,
            config.intermediate_size,
            config.num_layers,
            bytes_per_param,
            config.checkpointing,
        )
    ) / (1024**3)

    liger_flce_gb = (
        estimate_liger_flce_peak_gb(
            config.micro_batch_size,
            config.seq_len,
            config.vocab_size,
            config.hidden_size,
        )
        if config.use_liger_kernel
        else 0.0
    )

    # ---- Fragmentation penalty (CUDACachingAllocator) ----
    af = config.allocator_factor
    sum_before_frag = (
        weights_gb
        + gradients_gb
        + optimizer_gb
        + raw_activation_per_sample_gb * config.micro_batch_size
        + liger_flce_gb
        + cuda_context_gb
        + compile_gb
    )
    total_estimated_gb = sum_before_frag * af
    fragmentation_gb = total_estimated_gb - sum_before_frag

    # ---- Fixed reserved (everything except activations) ----
    fixed_before_frag = (
        weights_gb + gradients_gb + optimizer_gb + liger_flce_gb + cuda_context_gb + compile_gb
    )
    fixed_reserved_gb = fixed_before_frag * af

    # Per-sample activation with fragmentation
    activation_per_sample_gb = raw_activation_per_sample_gb * af

    # ---- Safety ----
    vram_limit = config.total_vram_gb
    util_pct = (total_estimated_gb / vram_limit * 100) if vram_limit > 0 else 0.0
    is_safe = total_estimated_gb <= vram_limit

    breakdown = VramBreakdown(
        weights_gb=round(weights_gb, 4),
        gradients_gb=round(gradients_gb, 4),
        optimizer_states_gb=round(optimizer_gb, 4),
        activations_gb=round(raw_activation_per_sample_gb * config.micro_batch_size, 4),
        liger_flce_workspace_gb=round(liger_flce_gb, 4),
        cuda_context_gb=round(cuda_context_gb, 4),
        compile_overhead_gb=round(compile_gb, 4),
        fragmentation_gb=round(fragmentation_gb, 4),
        total_estimated_gb=round(total_estimated_gb, 4),
        vram_limit_gb=round(vram_limit, 2),
        util_pct=round(util_pct, 1),
        is_safe=is_safe,
    )

    return VramEstimate(
        breakdown=breakdown,
        fixed_reserved_gb=round(fixed_reserved_gb, 6),
        activation_per_sample_gb=round(activation_per_sample_gb, 6),
    )


def auto_calibrate(
    hidden_size: int,
    intermediate_size: int,
    num_layers: int,
    vocab_size: int = 32000,
    seq_len: int = 1024,
    micro_batch_size: int = 1,
    n_params: int = 0,
    n_samples: int = 5,
    label: str = "auto",
) -> bool:
    """GPU 上でモデルをビルドして実測キャリブレーションを実行し、vram_calibration.json に保存する。

    Returns:
        True if calibration was successfully run and saved.
    """
    from src.common.logger import logger

    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("[auto_calibrate] CUDA not available, skipping VRAM calibration")
            return False
    except ImportError:
        return False

    if n_params == 0:
        n_params = _estimate_params(hidden_size, intermediate_size, num_layers, vocab_size)

    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    estimate_weights_gb = n_params * 2 / (1024**3)
    for bytes_per_param in [2, 2, 2]:
        estimate_weights_gb += n_params * bytes_per_param / (1024**3)
    estimate_weights_gb *= 1.3

    if estimate_weights_gb > total_gb * 0.85:
        logger.warning(
            f"[auto_calibrate] Model {n_params / 1e6:.0f}M too large "
            f"({estimate_weights_gb:.2f} GB estimated) for {total_gb:.0f} GB GPU, "
            f"skipping VRAM calibration. Falling back to formula."
        )
        return False

    try:
        from transformers import LlamaConfig, LlamaForCausalLM

        num_attention_heads = max(4, hidden_size // 64)
        # GQA 制約: kv_heads は heads の約数である必要がある (chinchilla_law と同一規則)
        num_key_value_heads = max(1, num_attention_heads // 4)
        while num_key_value_heads > 1 and num_attention_heads % num_key_value_heads != 0:
            num_key_value_heads -= 1

        cfg = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            max_position_embeddings=seq_len,
            use_cache=False,
            attn_implementation="sdpa",
        )
        model = LlamaForCausalLM(cfg).to("cuda", dtype=torch.bfloat16).train()
        actual_n = sum(p.numel() for p in model.parameters())
        logger.info(
            f"[auto_calibrate] Built {actual_n / 1e6:.1f}M model (H={hidden_size}, L={num_layers})"
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        dummy_ids = torch.randint(0, 100, (micro_batch_size, seq_len), device="cuda")
        dummy_labels = dummy_ids.clone()

        cal = measure_training_vram(
            model=model,
            optimizer=optimizer,
            input_ids=dummy_ids,
            labels=dummy_labels,
            micro_batch_size=micro_batch_size,
            seq_len=seq_len,
            n_params=actual_n,
            precision="bf16",
            optimizer_type="adamw_bnb_8bit",
            use_liger_kernel=True,
            torch_compile=False,
            label=label or f"{actual_n / 1e6:.0f}M",
            n_samples=n_samples,
            warmup=True,
            agg="median",
        )
        cal.save()
        logger.info("[auto_calibrate] Saved vram_calibration.json")

        model.cpu()
        del model, optimizer
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        return True

    except Exception as e:
        logger.warning(f"[auto_calibrate] Calibration failed: {e}")
        import gc

        gc.collect()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return False


def _estimate_params(
    hidden_size: int, intermediate_size: int, num_layers: int, vocab_size: int
) -> int:
    embed = vocab_size * hidden_size
    per_layer = 4 * hidden_size * hidden_size + 3 * hidden_size * intermediate_size
    return embed + num_layers * per_layer
