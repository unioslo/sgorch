from sgorch.notify.log_only import LogOnlyNotifier
from sgorch.notify.base import Notification, NotificationLevel, NotificationCategory
from sgorch.notify.email import EmailNotifier
from sgorch.config import EmailConfig


def _note(level=NotificationLevel.INFO):
    return Notification(
        level=level,
        category=NotificationCategory.SYSTEM_ERROR,
        title="t",
        message="m",
    )


def test_log_only_should_send_respects_min_level():
    n = LogOnlyNotifier(min_level=NotificationLevel.WARNING)
    assert n.should_send(_note(NotificationLevel.ERROR)) is True
    assert n.should_send(_note(NotificationLevel.INFO)) is False


def test_log_only_send_does_not_crash(caplog):
    caplog.set_level("DEBUG")
    n = LogOnlyNotifier(min_level=NotificationLevel.DEBUG)
    assert n.send(_note(NotificationLevel.DEBUG)) is True
    assert n.send(_note(NotificationLevel.INFO)) is True
    assert n.send(_note(NotificationLevel.WARNING)) is True
    assert n.send(_note(NotificationLevel.ERROR)) is True
    assert n.send(_note(NotificationLevel.CRITICAL)) is True


def test_email_enabled_logic_and_send(monkeypatch, caplog):
    caplog.set_level("INFO")
    # missing config -> disabled
    n = EmailNotifier(EmailConfig())
    assert n.is_enabled() is False
    assert n.send(_note()) is False

    # valid minimal config -> enabled and send returns True (placeholder)
    cfg = EmailConfig(smtp_host="smtp", from_addr="from@example.com", to_addrs=["to@example.com"]) 
    n2 = EmailNotifier(cfg)
    assert n2.is_enabled() is True
    assert n2.send(_note()) is True
    # placeholder logs a warning; ensure no exception and a log entry exists
    assert any("Email sending not implemented" in r.message for r in caplog.records)

