import logging
import sys
from logging.handlers import QueueHandler
from typing import Any


def setup_worker_logger(level: str, log_queue: Any = None) -> None:
    """Send a worker process's logs to *log_queue*, or to stderr without one."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(lvl)
    if log_queue is None:
        handler: logging.Handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(processName)s | %(levelname)s | %(message)s")
        )
        root.addHandler(handler)
    else:
        root.addHandler(QueueHandler(log_queue))
