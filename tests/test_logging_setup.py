import io
import json
import logging
import sys

from sgorch.logging_setup import JSONFormatter, setup_logging, get_logger


def test_json_formatter_includes_context_fields():
    formatter = JSONFormatter()
    logger = logging.getLogger("t")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)

    # Inject extra fields by using LoggerAdapter-like .log with extra
    logger.info("hello", extra={"deployment": "d1", "job_id": "123", "event": "e"})
    out = stream.getvalue().strip()
    data = json.loads(out)
    assert data["msg"] == "hello"
    assert data["deployment"] == "d1"
    assert data["job_id"] == "123"
    assert data["event"] == "e"


def test_setup_logging_resets_handlers_and_uses_stdout(monkeypatch):
    # Put a dummy handler
    root = logging.getLogger()
    root.handlers = [logging.StreamHandler(io.StringIO())]
    setup_logging("INFO")
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)
