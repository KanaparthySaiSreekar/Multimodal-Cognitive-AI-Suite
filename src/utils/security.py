"""Security utilities: rate limiting, audit logging, authentication."""

import hashlib
import hmac
import json
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, Optional

from .logger import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """Token bucket rate limiter for API requests."""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: Optional[int] = None,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            burst_size: Maximum burst size (defaults to requests_per_minute)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        self.refill_rate = requests_per_minute / 60.0  # Tokens per second

        # Track tokens per client
        self.tokens: Dict[str, float] = defaultdict(lambda: self.burst_size)
        self.last_update: Dict[str, float] = defaultdict(lambda: time.time())

        # Track request history
        self.request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if request is allowed for client.

        Args:
            client_id: Unique client identifier

        Returns:
            True if request is allowed
        """
        current_time = time.time()

        # Refill tokens based on time elapsed
        time_elapsed = current_time - self.last_update[client_id]
        self.tokens[client_id] = min(
            self.burst_size,
            self.tokens[client_id] + (time_elapsed * self.refill_rate),
        )
        self.last_update[client_id] = current_time

        # Check if client has tokens
        if self.tokens[client_id] >= 1.0:
            self.tokens[client_id] -= 1.0
            self.request_history[client_id].append(current_time)
            return True

        logger.warning(f"Rate limit exceeded for client: {client_id}")
        return False

    def get_wait_time(self, client_id: str) -> float:
        """
        Get wait time before next request allowed.

        Args:
            client_id: Client identifier

        Returns:
            Wait time in seconds
        """
        if self.tokens[client_id] >= 1.0:
            return 0.0

        tokens_needed = 1.0 - self.tokens[client_id]
        return tokens_needed / self.refill_rate

    def reset(self, client_id: str):
        """Reset rate limit for client."""
        self.tokens[client_id] = self.burst_size
        self.last_update[client_id] = time.time()

    def get_stats(self, client_id: str) -> Dict:
        """Get rate limit statistics for client."""
        history = list(self.request_history[client_id])

        if not history:
            return {
                "requests_last_minute": 0,
                "requests_last_hour": 0,
                "available_tokens": self.tokens[client_id],
            }

        current_time = time.time()
        one_minute_ago = current_time - 60
        one_hour_ago = current_time - 3600

        return {
            "requests_last_minute": sum(1 for t in history if t > one_minute_ago),
            "requests_last_hour": sum(1 for t in history if t > one_hour_ago),
            "available_tokens": self.tokens[client_id],
            "wait_time_seconds": self.get_wait_time(client_id),
        }


def rate_limit(requests_per_minute: int = 60):
    """
    Decorator for rate limiting functions.

    Args:
        requests_per_minute: Maximum requests per minute

    Example:
        @rate_limit(requests_per_minute=30)
        def my_api_function(client_id, data):
            ...
    """
    limiter = RateLimiter(requests_per_minute=requests_per_minute)

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(client_id: str, *args, **kwargs):
            if not limiter.is_allowed(client_id):
                wait_time = limiter.get_wait_time(client_id)
                raise RateLimitExceededError(
                    f"Rate limit exceeded. Wait {wait_time:.2f}s before next request.",
                    wait_time=wait_time,
                )

            return func(client_id, *args, **kwargs)

        wrapper.rate_limiter = limiter
        return wrapper

    return decorator


class AuditLogger:
    """Structured audit logging for security events."""

    def __init__(self, log_dir: str = "./audit_logs"):
        """
        Initialize audit logger.

        Args:
            log_dir: Directory for audit logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger("audit")

    def log_event(
        self,
        event_type: str,
        user_id: str,
        action: str,
        resource: str,
        status: str,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ):
        """
        Log security/audit event.

        Args:
            event_type: Type of event (authentication, authorization, data_access, etc.)
            user_id: User identifier
            action: Action performed
            resource: Resource accessed
            status: Status (success, failure, denied)
            correlation_id: Request correlation ID
            metadata: Additional metadata
        """
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "action": action,
            "resource": resource,
            "status": status,
            "correlation_id": correlation_id,
            "metadata": metadata or {},
        }

        # Log to structured logger
        self.logger.info(
            f"[AUDIT] {event_type}: {action} on {resource} by {user_id} - {status}",
            extra=event,
        )

        # Also write to daily audit log file
        self._write_to_file(event)

    def log_authentication(
        self, user_id: str, success: bool, method: str = "token", ip_address: Optional[str] = None
    ):
        """Log authentication attempt."""
        self.log_event(
            event_type="authentication",
            user_id=user_id,
            action="login",
            resource="system",
            status="success" if success else "failure",
            metadata={"method": method, "ip_address": ip_address},
        )

    def log_data_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        correlation_id: Optional[str] = None,
    ):
        """Log data access event."""
        self.log_event(
            event_type="data_access",
            user_id=user_id,
            action=action,
            resource=f"{resource_type}:{resource_id}",
            status="success",
            correlation_id=correlation_id,
        )

    def log_inference(
        self,
        user_id: str,
        model_type: str,
        input_size: int,
        latency_ms: float,
        correlation_id: Optional[str] = None,
    ):
        """Log model inference event."""
        self.log_event(
            event_type="inference",
            user_id=user_id,
            action="predict",
            resource=f"model:{model_type}",
            status="success",
            correlation_id=correlation_id,
            metadata={"input_size": input_size, "latency_ms": latency_ms},
        )

    def log_error(
        self,
        user_id: str,
        error_code: str,
        error_message: str,
        correlation_id: Optional[str] = None,
    ):
        """Log error event."""
        self.log_event(
            event_type="error",
            user_id=user_id,
            action="error",
            resource="system",
            status="failure",
            correlation_id=correlation_id,
            metadata={"error_code": error_code, "error_message": error_message},
        )

    def _write_to_file(self, event: Dict):
        """Write event to daily log file."""
        today = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"audit_{today}.jsonl"

        with open(log_file, "a") as f:
            f.write(json.dumps(event) + "\n")

    def search_logs(
        self,
        user_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> list:
        """
        Search audit logs.

        Args:
            user_id: Filter by user ID
            event_type: Filter by event type
            start_date: Start date for search
            end_date: End date for search

        Returns:
            List of matching events
        """
        events = []

        # Determine date range
        if start_date is None:
            start_date = datetime.utcnow() - timedelta(days=7)
        if end_date is None:
            end_date = datetime.utcnow()

        # Read log files in date range
        current_date = start_date
        while current_date <= end_date:
            log_file = self.log_dir / f"audit_{current_date.strftime('%Y-%m-%d')}.jsonl"

            if log_file.exists():
                with open(log_file, "r") as f:
                    for line in f:
                        event = json.loads(line)

                        # Apply filters
                        if user_id and event.get("user_id") != user_id:
                            continue
                        if event_type and event.get("event_type") != event_type:
                            continue

                        events.append(event)

            current_date += timedelta(days=1)

        return events


class TokenAuthenticator:
    """Simple token-based authentication."""

    def __init__(self, secret_key: str):
        """
        Initialize authenticator.

        Args:
            secret_key: Secret key for token generation
        """
        self.secret_key = secret_key.encode("utf-8")

    def generate_token(self, user_id: str, expiry_seconds: int = 3600) -> str:
        """
        Generate authentication token.

        Args:
            user_id: User identifier
            expiry_seconds: Token expiry time

        Returns:
            Authentication token
        """
        expiry = int(time.time()) + expiry_seconds
        payload = f"{user_id}:{expiry}"

        signature = hmac.new(
            self.secret_key, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        token = f"{payload}:{signature}"
        return token

    def verify_token(self, token: str) -> Optional[str]:
        """
        Verify authentication token.

        Args:
            token: Token to verify

        Returns:
            User ID if valid, None otherwise
        """
        try:
            payload, signature = token.rsplit(":", 1)
            user_id, expiry = payload.split(":")

            # Check signature
            expected_signature = hmac.new(
                self.secret_key, payload.encode("utf-8"), hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                logger.warning("Invalid token signature")
                return None

            # Check expiry
            if int(expiry) < time.time():
                logger.warning(f"Token expired for user: {user_id}")
                return None

            return user_id

        except (ValueError, AttributeError):
            logger.warning("Malformed token")
            return None


class RateLimitExceededError(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str, wait_time: float):
        super().__init__(message)
        self.wait_time = wait_time
