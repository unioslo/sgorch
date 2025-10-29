"""
Lightweight SLURM error classification helpers.

We only distinguish controller outages for now so the reconciler can back off
without tearing workers down. The patterns are easy to extend if we later want
finer-grained categories (configuration vs resource vs state errors).

Based on the error messages from slurm: https://github.com/SchedMD/slurm/blob/master/src/common/slurm_errno.c
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional

from ..logging_setup import get_logger

logger = get_logger(__name__)


class SlurmError(RuntimeError):
    """Base class for SLURM related errors raised by SGOrch."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class SlurmUnavailableError(SlurmError):
    """Raised when the SLURM control plane appears to be unavailable."""


class SlurmOperation(Enum):
    """Operation context for error classification."""

    SUBMIT = "submit"
    LIST = "list"
    CANCEL = "cancel"
    STATUS = "status"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class _Pattern:
    regex: re.Pattern[str]
    description: str
    operations: tuple[SlurmOperation, ...]


def _compile_patterns() -> tuple[_Pattern, ...]:
    """Compile the small set of outage patterns we currently care about."""
    raw_patterns: Iterable[tuple[str, str, Iterable[SlurmOperation]]] = [
        (
            r"socket timed out",
            "Network timeout reaching SLURM controller",
            (SlurmOperation.SUBMIT, SlurmOperation.LIST),
        ),
        (
            r"communications connection failure",
            "Connection failure talking to SLURM controller",
            (SlurmOperation.SUBMIT, SlurmOperation.LIST),
        ),
        (
            r"connection (?:timed out|refused|reset by peer)",
            "Connection problem talking to SLURM controller",
            (SlurmOperation.SUBMIT, SlurmOperation.LIST, SlurmOperation.STATUS),
        ),
        (
            r"unable to contact slurm controller",
            "Unable to contact SLURM controller",
            (SlurmOperation.SUBMIT, SlurmOperation.LIST, SlurmOperation.STATUS),
        ),
        (
            r"slurm controller .*? not responding",
            "SLURM controller not responding",
            (SlurmOperation.SUBMIT, SlurmOperation.LIST, SlurmOperation.STATUS),
        ),
    ]

    patterns: list[_Pattern] = []
    for expr, description, operations in raw_patterns:
        try:
            regex = re.compile(expr, re.IGNORECASE)
        except re.error as exc:
            logger.warning(f"Failed to compile SLURM error pattern '{expr}': {exc}")
            continue
        patterns.append(
            _Pattern(
                regex=regex,
                description=description,
                operations=tuple(operations),
            )
        )
    return tuple(patterns)


_OUTAGE_PATTERNS = _compile_patterns()


def detect_slurm_unavailability(
    message: Optional[str],
    *,
    operation: SlurmOperation = SlurmOperation.UNKNOWN,
) -> Optional[str]:
    """Return a human readable description if the message looks like an outage."""
    if not message:
        return None
    normalized = message.strip().lower()
    if not normalized:
        return None

    for pattern in _OUTAGE_PATTERNS:
        if operation not in pattern.operations and SlurmOperation.UNKNOWN not in pattern.operations:
            continue
        if pattern.regex.search(normalized):
            return pattern.description
    return None


def raise_if_unavailable(
    message: Optional[str],
    *,
    operation: SlurmOperation = SlurmOperation.UNKNOWN,
) -> None:
    """Raise SlurmUnavailableError if the message indicates an outage."""
    detail = detect_slurm_unavailability(message, operation=operation)
    if detail:
        text = message.strip() if message else ""
        composed = f"{detail}: {text}" if text else detail
        raise SlurmUnavailableError(composed)
