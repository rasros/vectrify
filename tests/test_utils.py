import logging

from vectrify.utils import setup_logger


def _handler_types() -> set[str]:
    return {type(h).__name__ for h in logging.getLogger().handlers}


def test_console_only_without_a_log_file():
    setup_logger("INFO")
    assert "StreamHandler" in _handler_types()


def test_log_file_is_added_alongside_the_console(tmp_path):
    """Regression: a log file *replaced* the stderr handler. runner.py calls
    this again once the run dir exists, so piped runs lost every later message
    -- including 'no valid candidate found' and 'best candidate written'.
    """
    setup_logger("INFO", log_file=tmp_path / "search.log")
    kinds = _handler_types()
    assert "FileHandler" in kinds
    assert "StreamHandler" in kinds


def test_console_can_be_suppressed_for_the_dashboard(tmp_path):
    setup_logger("INFO", log_file=tmp_path / "search.log", console=False)
    kinds = _handler_types()
    assert "FileHandler" in kinds
    assert "StreamHandler" not in kinds


def test_log_file_actually_receives_records(tmp_path):
    path = tmp_path / "search.log"
    setup_logger("INFO", log_file=path)
    logging.getLogger("main").info("hello from the run")
    for h in logging.getLogger().handlers:
        h.flush()
    assert "hello from the run" in path.read_text(encoding="utf-8")
