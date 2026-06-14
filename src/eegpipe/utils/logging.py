"""Rich-backed logger factory so every module logs consistently."""

from __future__ import annotations

import logging

_CONFIGURED = False


def get_logger(name: str = "eegpipe", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
            fmt = "%(message)s"
        except ImportError:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
        logging.basicConfig(level=level, format=fmt, handlers=[handler])
        # MNE is extremely chatty; quiet it unless the user opts in.
        logging.getLogger("mne").setLevel(logging.WARNING)
        _CONFIGURED = True
    return logging.getLogger(name)
