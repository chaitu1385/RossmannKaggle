"""
Audit event data model.

Defines the immutable record structure for all auditable platform actions.
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class AuditEvent:
    """
    One immutable audit log entry.

    Every state-changing action in the platform produces an AuditEvent.
    Events are append-only — they cannot be modified or deleted.
    """
    action: str                              # e.g. "create_override", "promote_model"
    resource_type: str                       # e.g. "override", "model_card", "forecast_run"
    resource_id: str                         # ID of the affected resource
    user_id: str = "system"
    user_role: str = "admin"
    user_email: str = ""
    status: str = "SUCCESS"                  # SUCCESS | DENIED | FAILED
    old_value: Optional[str] = None          # JSON string of previous state
    new_value: Optional[str] = None          # JSON string of new state
    ip_address: Optional[str] = None
    request_id: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    audit_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])

    def to_dict(self) -> dict:
        """Convert to a flat dict for DataFrame construction."""
        return {
            "audit_id": self.audit_id,
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "user_email": self.user_email,
            "user_role": self.user_role,
            "action": self.action,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "status": self.status,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "ip_address": self.ip_address,
            "request_id": self.request_id,
            "error_message": self.error_message,
        }
