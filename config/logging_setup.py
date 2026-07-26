import logging

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """Configure Structlog once, at process startup, to emit structured
    (JSON) logs at and above log_level. Idempotent: calling it again
    simply replaces the prior configuration, which is convenient for
    tests. Library code (core/, engine/, optimizers/) logs through
    structlog.get_logger() regardless of whether this has been called;
    without it, structlog falls back to its own built-in defaults, so the
    engine works standalone even if the CLI (the only caller of this
    function) is never involved.
    """
    resolved_level = getattr(logging, log_level.upper(), logging.INFO)
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(resolved_level),
        logger_factory=structlog.PrintLoggerFactory(),
        # False, not True: module-level loggers (e.g. engine/pipeline.py's
        # `logger = structlog.get_logger()`) are created once and reused
        # for the life of the process. With caching enabled, that proxy's
        # *first* real log call permanently locks in whatever config was
        # active at that moment - a later configure_logging() call (e.g.
        # a second CLI invocation in the same process, or a test using
        # structlog.testing.capture_logs()) would then silently have no
        # effect on it. Verified this exact failure mode empirically
        # while writing the test suite.
        cache_logger_on_first_use=False,
    )
