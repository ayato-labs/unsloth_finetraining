import psutil
import torch
from transformers import TrainerCallback

from src.common.logger import logger, handle_failure


def get_gpu_telemetry(trace_id: str) -> dict:
    telemetry = {
        "gpu_allocated_mb": 0.0,
        "gpu_reserved_mb": 0.0,
        "gpu_max_allocated_mb": 0.0,
        "cpu_percent": psutil.cpu_percent(),
        "ram_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
    }

    if torch.cuda.is_available():
        telemetry.update(
            {
                "gpu_allocated_mb": round(torch.cuda.memory_allocated() / (1024**2), 2),
                "gpu_reserved_mb": round(torch.cuda.memory_reserved() / (1024**2), 2),
                "gpu_max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 2),
            }
        )

    logger.info("telemetry", **telemetry, trace_id=trace_id)
    return telemetry


def check_gpu_availability(trace_id: str) -> None:
    if not torch.cuda.is_available():
        exc = RuntimeError("CUDA is not available. GPU is required for training.")
        handle_failure("check_gpu_availability", exc, trace_id)
        raise exc

    device_name = torch.cuda.get_device_name(0)
    total_memory = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    logger.info("gpu_detected", device=device_name, total_vram_gb=total_memory, trace_id=trace_id)


class TelemetryCallback(TrainerCallback):
    def __init__(self, trace_id: str, interval: int = 20):
        self.trace_id = trace_id
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.interval == 0:
            get_gpu_telemetry(self.trace_id)
