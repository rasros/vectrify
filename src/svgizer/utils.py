import logging


def setup_logger(level: str) -> None:
    lvl = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s | %(processName)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )