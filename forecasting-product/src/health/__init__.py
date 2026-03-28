"""
Health Check — validate system state and module availability.

Checks: core module imports, data connectivity, config integrity,
and pipeline readiness.  Use before a pipeline run to catch issues early.

Usage::

    from src.health import run_health_check

    report = run_health_check()
    print(report["summary"])
    if not report["overall_ok"]:
        for check in report["checks"]:
            if not check["ok"]:
                print(f"  FAIL: {check['name']} — {check['message']}")
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Module import check
# ---------------------------------------------------------------------------

_CORE_MODULES = [
    "src.validation",
    "src.visualization",
    "src.presentation",
    "src.tieout",
    "src.lineage",
    "src.profiler",
    "src.stats",
    "src.errors",
    "src.observability",
]

_OPTIONAL_MODULES = [
    "src.ai.commentary",
    "src.metrics.drift",
    "src.evaluation.evaluator",
    "src.forecasting",
    "src.hierarchy",
]


def check_module_imports() -> dict:
    """Verify all core and optional modules can be imported.

    Returns:
        dict with ok, core (list), optional (list).
        Each entry has name, importable, message.
    """
    core_results = []
    for mod_name in _CORE_MODULES:
        try:
            importlib.import_module(mod_name)
            core_results.append({"name": mod_name, "importable": True, "message": "OK"})
        except Exception as e:
            core_results.append({"name": mod_name, "importable": False, "message": str(e)})

    optional_results = []
    for mod_name in _OPTIONAL_MODULES:
        try:
            importlib.import_module(mod_name)
            optional_results.append({"name": mod_name, "importable": True, "message": "OK"})
        except Exception as e:
            optional_results.append({"name": mod_name, "importable": False, "message": str(e)})

    all_core_ok = all(r["importable"] for r in core_results)

    return {
        "ok": all_core_ok,
        "core": core_results,
        "optional": optional_results,
    }


# ---------------------------------------------------------------------------
# Dependency check
# ---------------------------------------------------------------------------

_REQUIRED_PACKAGES = [
    "polars",
    "numpy",
]

_OPTIONAL_PACKAGES = [
    "scipy",
    "matplotlib",
    "plotly",
    "marp",
    "yaml",
]


def check_dependencies() -> dict:
    """Check that required and optional Python packages are installed.

    Returns:
        dict with ok, required (list), optional (list).
    """
    required = []
    for pkg in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
            required.append({"name": pkg, "installed": True, "message": "OK"})
        except ImportError:
            required.append({"name": pkg, "installed": False, "message": f"pip install {pkg}"})

    optional = []
    for pkg in _OPTIONAL_PACKAGES:
        try:
            importlib.import_module(pkg)
            optional.append({"name": pkg, "installed": True, "message": "OK"})
        except ImportError:
            optional.append({"name": pkg, "installed": False, "message": f"pip install {pkg}"})

    return {
        "ok": all(r["installed"] for r in required),
        "required": required,
        "optional": optional,
    }


# ---------------------------------------------------------------------------
# Config integrity
# ---------------------------------------------------------------------------

def check_config(config_path: Optional[str] = None) -> dict:
    """Validate pipeline configuration file exists and parses.

    Args:
        config_path: Path to config YAML.  Defaults to 'config/pipeline.yaml'.

    Returns:
        dict with ok, path, message.
    """
    path = Path(config_path) if config_path else Path("config/pipeline.yaml")

    if not path.exists():
        return {
            "ok": False,
            "path": str(path),
            "message": f"Config file not found: {path}",
        }

    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            return {
                "ok": False,
                "path": str(path),
                "message": "Config file parsed but is not a dict.",
            }
        return {
            "ok": True,
            "path": str(path),
            "message": f"Config loaded: {len(cfg)} top-level keys.",
        }
    except ImportError:
        return {
            "ok": True,
            "path": str(path),
            "message": "File exists but PyYAML not installed — cannot validate contents.",
        }
    except Exception as e:
        return {
            "ok": False,
            "path": str(path),
            "message": f"Parse error: {e}",
        }


# ---------------------------------------------------------------------------
# Data directory check
# ---------------------------------------------------------------------------

def check_data_directory(data_dir: Optional[str] = None) -> dict:
    """Check that the data directory exists and contains files.

    Args:
        data_dir: Path to data directory.  Defaults to 'data/'.

    Returns:
        dict with ok, path, file_count, message.
    """
    path = Path(data_dir) if data_dir else Path("data")

    if not path.exists():
        return {
            "ok": False,
            "path": str(path),
            "file_count": 0,
            "message": f"Data directory not found: {path}",
        }

    files = list(path.rglob("*"))
    data_files = [f for f in files if f.is_file() and f.suffix in (
        ".csv", ".parquet", ".xlsx", ".json",
    )]

    return {
        "ok": len(data_files) > 0,
        "path": str(path),
        "file_count": len(data_files),
        "message": f"Found {len(data_files)} data file(s) in {path}.",
    }


# ---------------------------------------------------------------------------
# Output directory check
# ---------------------------------------------------------------------------

def check_output_directories() -> dict:
    """Ensure working/ and outputs/ directories exist.

    Returns:
        dict with ok, directories (list of {path, exists}).
    """
    dirs = ["working", "outputs", "outputs/charts"]
    results = []
    for d in dirs:
        p = Path(d)
        exists = p.exists()
        if not exists:
            try:
                p.mkdir(parents=True, exist_ok=True)
                results.append({"path": d, "exists": True, "created": True})
            except Exception as e:
                results.append({"path": d, "exists": False, "created": False, "error": str(e)})
        else:
            results.append({"path": d, "exists": True, "created": False})

    return {
        "ok": all(r["exists"] for r in results),
        "directories": results,
    }


# ---------------------------------------------------------------------------
# Combined health check
# ---------------------------------------------------------------------------

def run_health_check(config_path: Optional[str] = None, data_dir: Optional[str] = None) -> dict:
    """Run all health checks and return a combined report.

    Args:
        config_path: Optional config file path.
        data_dir: Optional data directory path.

    Returns:
        dict with overall_ok, checks (list), summary.
    """
    checks: list[dict] = []

    # 1. Dependencies
    deps = check_dependencies()
    checks.append({
        "name": "dependencies",
        "ok": deps["ok"],
        "message": (
            "All required packages installed."
            if deps["ok"]
            else "Missing: " + ", ".join(
                r["name"] for r in deps["required"] if not r["installed"]
            )
        ),
        "detail": deps,
    })

    # 2. Module imports
    modules = check_module_imports()
    checks.append({
        "name": "module_imports",
        "ok": modules["ok"],
        "message": (
            "All core modules importable."
            if modules["ok"]
            else "Failed: " + ", ".join(
                r["name"] for r in modules["core"] if not r["importable"]
            )
        ),
        "detail": modules,
    })

    # 3. Config
    cfg = check_config(config_path)
    checks.append({
        "name": "config",
        "ok": cfg["ok"],
        "message": cfg["message"],
        "detail": cfg,
    })

    # 4. Data directory
    data = check_data_directory(data_dir)
    checks.append({
        "name": "data_directory",
        "ok": data["ok"],
        "message": data["message"],
        "detail": data,
    })

    # 5. Output directories
    out = check_output_directories()
    checks.append({
        "name": "output_directories",
        "ok": out["ok"],
        "message": "Output directories ready." if out["ok"] else "Could not create output dirs.",
        "detail": out,
    })

    passed = sum(1 for c in checks if c["ok"])
    total = len(checks)
    overall = all(c["ok"] for c in checks)

    return {
        "overall_ok": overall,
        "checks": checks,
        "summary": f"{passed}/{total} checks passed",
    }
