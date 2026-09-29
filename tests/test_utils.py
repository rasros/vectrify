import logging
import queue
from logging.handlers import QueueHandler

from vectrify.utils import setup_worker_logger


def _handlers() -> list[logging.Handler]:
    return list(logging.getLogger().handlers)


def test_worker_logs_go_to_the_queue_when_there_is_one():
    q: queue.Queue = queue.Queue()
    setup_worker_logger("INFO", q)
    assert [type(h) for h in _handlers()] == [QueueHandler]
    logging.getLogger("worker").warning("hello")
    assert q.get_nowait().getMessage() == "hello"


def test_worker_logs_go_to_stderr_without_a_queue(capsys):
    setup_worker_logger("INFO", None)
    assert [type(h) for h in _handlers()] == [logging.StreamHandler]
    logging.getLogger("worker").warning("hello")
    assert "hello" in capsys.readouterr().err
