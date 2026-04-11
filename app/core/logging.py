import logging
import sys

from app.core.config import get_settings

settings = get_settings()  # we need this to check debug mode for logging setup

class _Formatter(logging.Formatter):
    """
    Human-readable output for development.
    In production, swap this for a JSON formatter
    so tje log aggregator (Datadog, CloudWatch etc.)
    can parse structured fields.
    """

    """
    DEBUG     → fine-grained detail, only useful when hunting a bug
    INFO      → normal operation ("file received", "job started")
    WARNING   → something unexpected but recoverable
    ERROR     → something failed but the app is still running
    CRITICAL  → something failed badly, app may not recover
    """

    COLORS = {
        "DEBUG": "\033[36m",    # cyan
        "INFO": "\033[32m",     # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",    # red
        "CRITICAL": "\033[41m", # red background
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        level = f"{color}{record.levelname:<8}{self.RESET}"
        location = f"{record.name}:{record.lineno}"
        return f"{level} {location:<40} {record.getMessage()}"

def setup_logging() -> None:
    """
    Configure the root logger.
    Call this once at app startup in main.py - before anything else.
    """
    log_level = logging.DEBUG if settings.debug else logging.INFO

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_Formatter())

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers.clear()       # remove uvicorn's default handlers
    root.addHandler(handler)

    # These third-party loggers are noisy and rarely useful
    for noisy in ("uvicorn.access", "multipart", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger. Call this at the top of any file that needs logging:

        from app.core.logging import get_logger
        logger = get_logger(__name__)
    
    Passing __name__ means the logger is named after the module,
    so your output shows exactly which file the log came from.
    """

    return logging.getLogger(name)
