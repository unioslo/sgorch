from sgorch.notify.base import CompositeNotifier, Notification, NotificationLevel, NotificationCategory, Notifier


class _Ok(Notifier):
    def __init__(self):
        self.sent = []
    def send(self, n: Notification) -> bool:
        self.sent.append(n)
        return True
    def is_enabled(self) -> bool:
        return True


class _No(Notifier):
    def send(self, n: Notification) -> bool:
        return False
    def is_enabled(self) -> bool:
        return True


class _Boom(Notifier):
    def send(self, n: Notification) -> bool:
        raise RuntimeError("boom")
    def is_enabled(self) -> bool:
        return True


def _note():
    return Notification(level=NotificationLevel.INFO, category=NotificationCategory.SYSTEM_ERROR, title="t", message="m")


def test_composite_fanout_and_partial_failure():
    ok1, ok2 = _Ok(), _Ok()
    comp = CompositeNotifier([ok1, ok2, _No()])
    assert comp.send(_note()) is False  # one false makes overall false
    assert len(ok1.sent) == 1 and len(ok2.sent) == 1


def test_composite_raises_are_caught():
    ok = _Ok()
    comp = CompositeNotifier([ok, _Boom()])
    assert comp.send(_note()) is False
    assert len(ok.sent) == 1

