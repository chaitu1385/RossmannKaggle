"""
Microsoft Fabric integration layer for the forecasting product.

Modules
-------
config          FabricConfig dataclass — workspace / lakehouse settings.
lakehouse       Read and write Delta tables in a Fabric Lakehouse.
delta_writer    Upsert / overwrite helpers for Delta tables.
deployment      End-to-end deployment orchestrator with pre/post-run checks.
"""

from .config import FabricConfig  # noqa: F401
from .delta_writer import DeltaWriter  # noqa: F401
from .deployment import DeploymentConfig, DeploymentOrchestrator, DeploymentResult  # noqa: F401
from .lakehouse import FabricLakehouse  # noqa: F401
