"""
FabricConfig — Microsoft Fabric workspace and Lakehouse settings.

These values can be supplied via:
  1. Environment variables (``FABRIC_WORKSPACE``, ``FABRIC_LAKEHOUSE``, …).
  2. Direct constructor arguments.
  3. The ``fabric_config.yaml`` file (loaded by ``src.config.loader``).

Usage
-----
>>> from src.fabric.config import FabricConfig
>>> cfg = FabricConfig.from_env()
>>> print(cfg.abfss_base)
'abfss://my-workspace@onelake.dfs.fabric.microsoft.com/my-lakehouse'
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FabricConfig:
    """
    Fabric workspace and storage settings.

    Attributes
    ----------
    workspace:
        Fabric workspace name or GUID.
    lakehouse:
        Lakehouse name or GUID within the workspace.
    onelake_host:
        OneLake DFS endpoint (default: ``onelake.dfs.fabric.microsoft.com``).
    tables_root:
        Sub-path under the Lakehouse root where managed Delta tables live
        (default: ``Tables``).
    files_root:
        Sub-path for unmanaged files (default: ``Files``).
    environment:
        ``"development"`` | ``"staging"`` | ``"production"``.
    enable_delta_log_retention:
        Whether to configure Delta log / data file retention settings.
    delta_log_retention_days:
        How many days to keep Delta transaction log entries.
    delta_data_retention_days:
        How many days to keep data files eligible for VACUUM.
    """

    workspace: str = ""
    lakehouse: str = ""
    onelake_host: str = "onelake.dfs.fabric.microsoft.com"
    tables_root: str = "Tables"
    files_root: str = "Files"
    environment: str = "development"
    enable_delta_log_retention: bool = True
    delta_log_retention_days: int = 30
    delta_data_retention_days: int = 7

    # ── derived properties ────────────────────────────────────────────────────

    @property
    def abfss_base(self) -> str:
        """ABFSS URI for the Lakehouse root."""
        return (
            f"abfss://{self.workspace}@{self.onelake_host}/{self.lakehouse}"
        )

    @property
    def tables_path(self) -> str:
        """ABFSS path to the managed Delta tables root."""
        return f"{self.abfss_base}/{self.tables_root}"

    @property
    def files_path(self) -> str:
        """ABFSS path to the unmanaged files root."""
        return f"{self.abfss_base}/{self.files_root}"

    def table_path(self, table_name: str) -> str:
        """Full ABFSS path for a specific Delta table."""
        return f"{self.tables_path}/{table_name}"

    def file_path(self, *parts: str) -> str:
        """Full ABFSS path for a file under the Files root."""
        return "/".join([self.files_path] + list(parts))

    # ── factories ─────────────────────────────────────────────────────────────

    @classmethod
    def from_env(cls) -> "FabricConfig":
        """
        Build a FabricConfig from environment variables.

        Environment variables
        ---------------------
        FABRIC_WORKSPACE            Workspace name / GUID.
        FABRIC_LAKEHOUSE            Lakehouse name / GUID.
        FABRIC_ONELAKE_HOST         Override OneLake host.
        FABRIC_ENVIRONMENT          development | staging | production.
        """
        return cls(
            workspace=os.environ.get("FABRIC_WORKSPACE", ""),
            lakehouse=os.environ.get("FABRIC_LAKEHOUSE", ""),
            onelake_host=os.environ.get(
                "FABRIC_ONELAKE_HOST", "onelake.dfs.fabric.microsoft.com"
            ),
            environment=os.environ.get("FABRIC_ENVIRONMENT", "development"),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "FabricConfig":
        """Build a FabricConfig from a plain dict (e.g. parsed YAML)."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})

    def __str__(self) -> str:
        return (
            f"FabricConfig(workspace={self.workspace!r}, "
            f"lakehouse={self.lakehouse!r}, env={self.environment!r})"
        )
