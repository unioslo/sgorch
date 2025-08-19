from ..logging_setup import get_logger
from .base import Notifier, Notification, NotificationLevel


logger = get_logger(__name__)


class LogOnlyNotifier(Notifier):
    """Notifier that only logs notifications (no external delivery)."""
    
    def __init__(self, min_level: NotificationLevel = NotificationLevel.INFO):
        self.min_level = min_level
    
    def send(self, notification: Notification) -> bool:
        """Log the notification at appropriate level."""
        try:
            # Build log message
            message = self._format_message(notification)
            
            # Create log context
            context = {
                "event": "notification",
                "category": notification.category.value,
                "notification_level": notification.level.value,
            }
            
            # Add optional fields
            if notification.deployment:
                context["deployment"] = notification.deployment
            if notification.worker_url:
                context["worker_url"] = notification.worker_url
            if notification.job_id:
                context["job_id"] = notification.job_id
            if notification.node:
                context["node"] = notification.node
            if notification.details:
                context["details"] = notification.details
            
            # Log at appropriate level
            if notification.level == NotificationLevel.DEBUG:
                logger.debug(message, **context)
            elif notification.level == NotificationLevel.INFO:
                logger.info(message, **context)
            elif notification.level == NotificationLevel.WARNING:
                logger.warning(message, **context)
            elif notification.level == NotificationLevel.ERROR:
                logger.error(message, **context)
            elif notification.level == NotificationLevel.CRITICAL:
                logger.critical(message, **context)
            else:
                logger.info(message, **context)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log notification: {e}")
            return False
    
    def is_enabled(self) -> bool:
        """Log-only notifier is always enabled."""
        return True
    
    def should_send(self, notification: Notification) -> bool:
        """Check if notification level meets minimum threshold."""
        level_order = {
            NotificationLevel.DEBUG: 0,
            NotificationLevel.INFO: 1,
            NotificationLevel.WARNING: 2,
            NotificationLevel.ERROR: 3,
            NotificationLevel.CRITICAL: 4,
        }
        
        return level_order.get(notification.level, 1) >= level_order.get(self.min_level, 1)
    
    def _format_message(self, notification: Notification) -> str:
        """Format notification as a log message."""
        parts = [f"[{notification.level.value.upper()}]", notification.title]
        
        if notification.message != notification.title:
            parts.append(f"- {notification.message}")
        
        return " ".join(parts)