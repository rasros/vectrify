import logging
import multiprocessing as mp
import sys
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path


def setup_logger(
    level: str,
    log_file: Path | str | None = None,
    console: bool = True,
) -> None:
    """Configure root logging to stderr and, when given, to *log_file*.

    A log file is added alongside the console handler rather than replacing it:
    this is called a second time once the run directory exists, and dropping
    stderr there left piped or scripted runs with no output at all -- including
    the "no valid candidate found" and "best candidate written" lines. Pass
    ``console=False`` only when something else owns the terminal, i.e. the live
    dashboard.
    """
    lvl = getattr(logging, level.upper(), logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(lvl)

    if console:
        sh = logging.StreamHandler(sys.stderr)
        sh.setFormatter(fmt)
        root.addHandler(sh)

    if log_file is not None:
        fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)


def start_log_listener() -> tuple[mp.Queue, QueueListener]:
    """Creates a queue and a listener attached to the current root handlers."""
    q = mp.get_context("spawn").Queue(-1)
    root = logging.getLogger()
    listener = QueueListener(q, *root.handlers, respect_handler_level=True)
    listener.start()
    return q, listener


def setup_worker_logger(level: str, log_queue: mp.Queue) -> None:
    """Configures a worker process to push all logs to the queue."""
    lvl = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(lvl)
    root.addHandler(QueueHandler(log_queue))
