"""Tests for RBAC and audit trail."""

from datetime import datetime, timezone

import polars as pl
import pytest

from src.auth.models import Permission, Role, User, ROLE_PERMISSIONS
from src.audit.schemas import AuditEvent
from src.audit.logger import AuditLogger


class TestRolePermissions:
    def test_admin_has_all_permissions(self):
        assert ROLE_PERMISSIONS[Role.ADMIN] == set(Permission)

    def test_viewer_limited_permissions(self):
        perms = ROLE_PERMISSIONS[Role.VIEWER]
        assert Permission.VIEW_FORECASTS in perms
        assert Permission.VIEW_METRICS in perms
        assert Permission.CREATE_OVERRIDE not in perms
        assert Permission.MANAGE_USERS not in perms

    def test_planner_can_create_overrides(self):
        perms = ROLE_PERMISSIONS[Role.PLANNER]
        assert Permission.CREATE_OVERRIDE in perms
        assert Permission.APPROVE_OVERRIDE not in perms

    def test_manager_can_approve_overrides(self):
        perms = ROLE_PERMISSIONS[Role.MANAGER]
        assert Permission.APPROVE_OVERRIDE in perms
        assert Permission.CREATE_OVERRIDE not in perms

    def test_data_scientist_cannot_manage_users(self):
        perms = ROLE_PERMISSIONS[Role.DATA_SCIENTIST]
        assert Permission.MANAGE_USERS not in perms
        assert Permission.RUN_BACKTEST in perms


class TestUserModel:
    def test_user_has_permission(self):
        user = User(user_id="u1", email="a@b.com", role=Role.ADMIN)
        assert user.has_permission(Permission.MANAGE_USERS) is True

    def test_inactive_user_denied(self):
        user = User(user_id="u1", email="a@b.com", role=Role.ADMIN, is_active=False)
        assert user.has_permission(Permission.VIEW_FORECASTS) is False

    def test_viewer_denied_write(self):
        user = User(user_id="u1", email="a@b.com", role=Role.VIEWER)
        assert user.has_permission(Permission.CREATE_OVERRIDE) is False

    def test_user_to_dict(self):
        user = User(user_id="u1", email="a@b.com", role=Role.PLANNER)
        d = user.to_dict()
        assert d["role"] == "planner"
        assert d["user_id"] == "u1"


class TestAuditEvent:
    def test_event_creation(self):
        event = AuditEvent(
            action="create_override",
            resource_type="override",
            resource_id="OVR-001",
            user_id="jane",
            user_role="planner",
        )
        assert event.status == "SUCCESS"
        assert len(event.audit_id) == 16
        assert event.action == "create_override"

    def test_event_to_dict(self):
        event = AuditEvent(
            action="delete_override",
            resource_type="override",
            resource_id="OVR-002",
        )
        d = event.to_dict()
        assert "audit_id" in d
        assert d["action"] == "delete_override"


class TestAuditLogger:
    def test_log_and_query(self, tmp_path):
        logger = AuditLogger(str(tmp_path / "audit"))
        event = AuditEvent(
            action="create_override",
            resource_type="override",
            resource_id="OVR-001",
            user_id="jane",
            user_role="planner",
        )
        logger.log(event)

        results = logger.query()
        assert len(results) == 1
        assert results["action"][0] == "create_override"
        assert results["user_id"][0] == "jane"

    def test_query_filters(self, tmp_path):
        logger = AuditLogger(str(tmp_path / "audit"))
        for action in ["create_override", "delete_override", "view_forecast"]:
            logger.log(AuditEvent(
                action=action,
                resource_type="override",
                resource_id=f"OVR-{action}",
                user_id="jane",
            ))

        results = logger.query(action="create_override")
        assert len(results) == 1

    def test_count_by_action(self, tmp_path):
        logger = AuditLogger(str(tmp_path / "audit"))
        for _ in range(3):
            logger.log(AuditEvent(action="create", resource_type="x", resource_id="1"))
        logger.log(AuditEvent(action="delete", resource_type="x", resource_id="2"))

        counts = logger.count_by_action()
        assert len(counts) == 2
