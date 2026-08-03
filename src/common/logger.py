import sys
import uuid
from contextlib import contextmanager
from loguru import logger


def generate_trace_id() -> str:
    return str(uuid.uuid4())[:8]


def setup_logging() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )
    logger.add(
        "logs/train_{time:YYYYMMDD}.log",
        rotation="10 MB",
        retention="10 days",
        level="DEBUG",
        serialize=True,
    )


def handle_failure(func_name: str, exc: Exception, trace_id: str) -> None:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "unknown"
    line_no = exc_tb.tb_lineno if exc_tb else 0

    logger.error(
        f"Failure in {func_name}: {exc}",
        cause=str(exc),
        file=file_name,
        line=line_no,
        trace_id=trace_id,
        exc_info=True,
    )


@contextmanager
def trace_context(trace_id: str, step: str):
    logger.info("step_start", step=step, trace_id=trace_id)
    try:
        yield
        logger.info("step_complete", step=step, trace_id=trace_id)
    except Exception as exc:
        handle_failure(step, exc, trace_id)
        raise
