"""Orchestrator for the M5 daily-frequency end-to-end test suite.

Runs each phase in sequence:
  A) Generate M5 daily fixtures (if missing)
  B) Start FastAPI backend
  C) Run pytest backend integration tests
  D) Start Next.js frontend
  E) Run Playwright browser E2E tests
  F) Print summary

Usage
-----
    cd forecasting-product
    python -m tests.integration.run_m5_daily_e2e
    python -m tests.integration.run_m5_daily_e2e --skip-frontend   # backend only
    python -m tests.integration.run_m5_daily_e2e --skip-backend    # frontend only (servers must be running)
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # forecasting-product/
FIXTURE_CSV = ROOT / "tests" / "integration" / "fixtures" / "m5_daily_sample.csv"
CONFIG_YAML = ROOT / "tests" / "integration" / "fixtures" / "m5_daily_config.yaml"
FRONTEND_DIR = ROOT / "frontend"

BACKEND_PORT = 8000
FRONTEND_PORT = 3000


def _banner(msg: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {msg}")
    print(f"{'=' * 72}\n")


def _run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> int:
    """Run a command, streaming output.  Returns exit code."""
    proc = subprocess.run(cmd, cwd=cwd, timeout=timeout)
    return proc.returncode


def _start_server(cmd: list[str], cwd: Path, port: int, label: str) -> subprocess.Popen:
    """Start a server in the background and wait until the port is accepting."""
    _banner(f"Starting {label} on port {port} …")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Wait for the port to become available (max 60s)
    import socket

    for attempt in range(120):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                print(f"  {label} ready on port {port}")
                return proc
        except OSError:
            time.sleep(0.5)
            if proc.poll() is not None:
                stdout = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
                print(f"  ERROR: {label} exited prematurely (code {proc.returncode})")
                print(stdout[-2000:])
                sys.exit(1)

    print(f"  TIMEOUT: {label} did not start within 60 seconds")
    proc.terminate()
    sys.exit(1)


def _stop(proc: subprocess.Popen, label: str) -> None:
    if proc.poll() is None:
        print(f"  Stopping {label} (pid={proc.pid}) …")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ── Phases ────────────────────────────────────────────────────────────────────

def phase_a_prepare_data() -> bool:
    """Generate fixture CSV if it doesn't exist."""
    _banner("Phase A: Data Preparation")
    if FIXTURE_CSV.exists():
        print(f"  Fixture already exists: {FIXTURE_CSV}")
        return True

    code = _run(
        [sys.executable, "-m", "tests.integration.prepare_m5_daily"],
        cwd=ROOT,
        timeout=300,
    )
    if code != 0:
        print("  FAILED: data preparation failed")
        return False
    return FIXTURE_CSV.exists()


def phase_b_start_backend() -> subprocess.Popen:
    """Start the FastAPI backend."""
    return _start_server(
        [
            sys.executable, "scripts/serve.py",
            "--port", str(BACKEND_PORT),
            "--data-dir", str(ROOT / "data"),
        ],
        cwd=ROOT,
        port=BACKEND_PORT,
        label="FastAPI backend",
    )


def phase_c_backend_tests() -> int:
    """Run pytest backend integration tests."""
    _banner("Phase C: Backend Integration Tests")
    return _run(
        [
            sys.executable, "-m", "pytest",
            "tests/integration/test_m5_daily_backend.py",
            "-v", "--tb=short",
        ],
        cwd=ROOT,
        timeout=600,
    )


def phase_d_start_frontend() -> subprocess.Popen:
    """Start the Next.js frontend."""
    npm_cmd = "npm.cmd" if sys.platform == "win32" else "npm"
    return _start_server(
        [npm_cmd, "run", "dev"],
        cwd=FRONTEND_DIR,
        port=FRONTEND_PORT,
        label="Next.js frontend",
    )


def phase_e_frontend_tests() -> int:
    """Run Playwright E2E tests tagged @m5daily."""
    _banner("Phase E: Playwright E2E Tests (daily pipeline)")
    npx_cmd = "npx.cmd" if sys.platform == "win32" else "npx"
    return _run(
        [npx_cmd, "playwright", "test", "--grep", "@m5daily"],
        cwd=FRONTEND_DIR,
        timeout=300,
    )


def phase_f_summary(results: dict[str, int | str]) -> None:
    """Print a summary table."""
    _banner("Phase F: Summary")
    for phase, result in results.items():
        status = "PASS" if result == 0 else ("SKIP" if result == "skip" else f"FAIL (exit {result})")
        print(f"  {phase:40s} {status}")

    failed = [k for k, v in results.items() if v not in (0, "skip")]
    if failed:
        print(f"\n  {len(failed)} phase(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n  All phases passed ✓")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="M5 Daily E2E Test Orchestrator")
    parser.add_argument("--skip-frontend", action="store_true", help="Skip frontend/Playwright tests")
    parser.add_argument("--skip-backend", action="store_true", help="Skip backend tests (assume servers running)")
    args = parser.parse_args()

    results: dict[str, int | str] = {}
    backend_proc = None
    frontend_proc = None

    try:
        # Phase A: Data prep
        if not phase_a_prepare_data():
            results["A: Data Preparation"] = 1
            phase_f_summary(results)
            return
        results["A: Data Preparation"] = 0

        # Phase B+C: Backend
        if not args.skip_backend:
            backend_proc = phase_b_start_backend()
            results["B: Start Backend"] = 0

            results["C: Backend Integration Tests"] = phase_c_backend_tests()
        else:
            results["B: Start Backend"] = "skip"
            results["C: Backend Integration Tests"] = "skip"

        # Phase D+E: Frontend
        if not args.skip_frontend:
            frontend_proc = phase_d_start_frontend()
            results["D: Start Frontend"] = 0

            results["E: Playwright E2E Tests"] = phase_e_frontend_tests()
        else:
            results["D: Start Frontend"] = "skip"
            results["E: Playwright E2E Tests"] = "skip"

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        # Cleanup servers
        if frontend_proc:
            _stop(frontend_proc, "frontend")
        if backend_proc:
            _stop(backend_proc, "backend")

    phase_f_summary(results)


if __name__ == "__main__":
    main()
