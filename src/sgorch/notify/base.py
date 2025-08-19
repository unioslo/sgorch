from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class NotificationLevel(Enum):
    """Notification severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class NotificationCategory(Enum):
    """Categories of notifications."""
    WORKER_HEALTH = "worker_health"
    DEPLOYMENT_STATE = "deployment_state"
    SYSTEM_ERROR = "system_error"
    RESOURCE_ISSUE = "resource_issue"
    NETWORK_ISSUE = "network_issue"
    SLURM_ISSUE = "slurm_issue"


@dataclass
class Notification:
    """A notification message."""
    level: NotificationLevel
    category: NotificationCategory
    title: str
    message: str
    deployment: Optional[str] = None
    worker_url: Optional[str] = None
    job_id: Optional[str] = None
    node: Optional[str] = None
    details: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = {}


class Notifier(ABC):
    """Base interface for notification systems."""
    
    @abstractmethod
    def send(self, notification: Notification) -> bool:
        """
        Send a notification.
        
        Args:
            notification: The notification to send
            
        Returns:
            True if notification was sent successfully
        """
        pass
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if this notifier is enabled and configured."""
        pass
    
    def should_send(self, notification: Notification) -> bool:
        """Check if this notification should be sent (for filtering)."""
        return self.is_enabled()


class CompositeNotifier(Notifier):
    """Notifier that sends to multiple notification backends."""
    
    def __init__(self, notifiers: list[Notifier]):
        self.notifiers = notifiers
    
    def send(self, notification: Notification) -> bool:
        """Send to all enabled notifiers."""
        success = True
        
        for notifier in self.notifiers:
            try:
                if notifier.should_send(notification):
                    result = notifier.send(notification)
                    success = success and result
            except Exception:
                success = False
        
        return success
    
    def is_enabled(self) -> bool:
        """Return True if any notifier is enabled."""
        return any(notifier.is_enabled() for notifier in self.notifiers)


def create_worker_unhealthy_notification(
    deployment: str,
    worker_url: str,
    job_id: Optional[str] = None,
    reason: str = "Health check failed",
    details: Dict[str, Any] = None
) -> Notification:
    """Create a notification for an unhealthy worker."""
    return Notification(
        level=NotificationLevel.WARNING,
        category=NotificationCategory.WORKER_HEALTH,
        title=f"Worker unhealthy in {deployment}",
        message=f"Worker {worker_url} is unhealthy: {reason}",
        deployment=deployment,
        worker_url=worker_url,
        job_id=job_id,
        details=details or {}
    )


def create_deployment_scaling_notification(
    deployment: str,
    desired: int,
    actual: int,
    reason: str = ""
) -> Notification:
    """Create a notification for deployment scaling issues."""
    level = NotificationLevel.WARNING if actual < desired else NotificationLevel.INFO
    
    return Notification(
        level=level,
        category=NotificationCategory.DEPLOYMENT_STATE,
        title=f"Scaling issue in {deployment}",
        message=f"Deployment {deployment}: desired={desired}, actual={actual}. {reason}",
        deployment=deployment,
        details={"desired": desired, "actual": actual, "reason": reason}
    )


def create_system_error_notification(
    title: str,
    message: str,
    deployment: Optional[str] = None,
    details: Dict[str, Any] = None
) -> Notification:
    """Create a system error notification."""
    return Notification(
        level=NotificationLevel.ERROR,
        category=NotificationCategory.SYSTEM_ERROR,
        title=title,
        message=message,
        deployment=deployment,
        details=details or {}
    )


def create_resource_issue_notification(
    deployment: str,
    resource_type: str,
    issue: str,
    details: Dict[str, Any] = None
) -> Notification:
    """Create a resource issue notification."""
    return Notification(
        level=NotificationLevel.WARNING,
        category=NotificationCategory.RESOURCE_ISSUE,
        title=f"Resource issue in {deployment}",
        message=f"{resource_type} issue: {issue}",
        deployment=deployment,
        details=details or {"resource_type": resource_type}
    )