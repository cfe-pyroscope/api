import logging
import sys


def setup_logging(level=logging.INFO):
    """
    Configure the global logging settings for the application.

    Sets the logging level, output format, and stream handler to print logs to stdout.

    Args:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG). Defaults to logging.INFO.

    Format:
        Logs are formatted as:
        'YYYY-MM-DD HH:MM:SS | LEVEL    | logger_name | message'
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

logger = logging.getLogger("uvicorn")
