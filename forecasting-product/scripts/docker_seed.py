"""Docker first-boot seeder — pre-runs backtest + forecast so the UI has data.

On first boot, this script:
  1. Generates the weekly M5 sample (daily → weekly aggregation)
  2. Starts the API server in the background
  3. Calls POST /pipeline/backtest with the weekly data + config
  4. Calls POST /pipeline/forecast with the champion model
  5. Writes a sentinel file so subsequent boots skip seeding
  6. Keeps the API server running (replaces itself with serve.py)

Usage (inside Docker):
    python forecasting-product/scripts/docker_seed.py
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("docker_seed")

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # forecasting-product/
DATA_DIR = Path(os.environ.get("API_DATA_DIR", str(_ROOT / "data")))
SENTINEL = DATA_DIR / ".seeded"
DEMO_LOB = os.environ.get("DEMO_LOB", "walmart_m5_weekly")
API_HOST = "0.0.0.0"
API_PORT = int(os.environ.get("API_PORT", "8000"))


def wait_for_api(host: str = "127.0.0.1", port: int = API_PORT, timeout: int = 120) -> bool:
    """Poll the /health endpoint until it responds or timeout."""
    import urllib.request
    import urllib.error

    url = f"http://{host}:{port}/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(2)
    return False


def seed_via_api(csv_path: Path, config_path: Path, lob: str) -> str | None:
    """Call the backtest + forecast API endpoints. Returns champion model name."""
    import urllib.request
    import urllib.error

    base = f"http://127.0.0.1:{API_PORT}"

    # ── Backtest ──────────────────────────────────────────────────────────
    logger.info("Seeding backtest for LOB=%s …", lob)
    csv_bytes = csv_path.read_bytes()
    config_bytes = config_path.read_bytes()

    # Build multipart form data
    boundary = "----DockerSeedBoundary"
    body = bytearray()

    # file field
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{csv_path.name}"\r\n'.encode()
    body += b"Content-Type: text/csv\r\n\r\n"
    body += csv_bytes
    body += b"\r\n"

    # config_file field
    body += f"--{boundary}\r\n".encode()
    body += f'Content-Disposition: form-data; name="config_file"; filename="{config_path.name}"\r\n'.encode()
    body += b"Content-Type: application/x-yaml\r\n\r\n"
    body += config_bytes
    body += b"\r\n"

    body += f"--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        f"{base}/pipeline/backtest?lob={lob}",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())
        champion = result.get("champion_model")
        best_wmape = result.get("best_wmape")
        logger.info("Backtest done — champion=%s, WMAPE=%s", champion, best_wmape)
    except (urllib.error.URLError, OSError) as exc:
        logger.error("Backtest API call failed: %s", exc)
        return None

    # ── Forecast ──────────────────────────────────────────────────────────
    logger.info("Seeding forecast with champion=%s …", champion or "naive_seasonal")
    body2 = bytearray()
    body2 += f"--{boundary}\r\n".encode()
    body2 += f'Content-Disposition: form-data; name="file"; filename="{csv_path.name}"\r\n'.encode()
    body2 += b"Content-Type: text/csv\r\n\r\n"
    body2 += csv_bytes
    body2 += b"\r\n"

    body2 += f"--{boundary}\r\n".encode()
    body2 += f'Content-Disposition: form-data; name="config_file"; filename="{config_path.name}"\r\n'.encode()
    body2 += b"Content-Type: application/x-yaml\r\n\r\n"
    body2 += config_bytes
    body2 += b"\r\n"

    body2 += f"--{boundary}--\r\n".encode()

    model = champion or "naive_seasonal"
    req2 = urllib.request.Request(
        f"{base}/pipeline/forecast?lob={lob}&model_id={model}",
        data=bytes(body2),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req2, timeout=600) as resp:
            result2 = json.loads(resp.read())
        logger.info("Forecast done — %d rows", result2.get("forecast_rows", 0))
    except (urllib.error.URLError, OSError) as exc:
        logger.error("Forecast API call failed: %s", exc)

    return champion


def main() -> None:
    if SENTINEL.exists():
        logger.info("Sentinel found (%s) — skipping seed, starting API directly.", SENTINEL)
        os.execvp(
            sys.executable,
            [sys.executable, str(_HERE / "serve.py"),
             "--host", API_HOST, "--port", str(API_PORT),
             "--data-dir", str(DATA_DIR)],
        )
        return  # unreachable after execvp

    # Step 1: Generate weekly M5 sample
    demo_csv = DATA_DIR / "demo" / "m5_weekly.csv"
    config_path = _ROOT / "configs" / "m5_weekly_config.yaml"

    if not demo_csv.exists():
        logger.info("Generating weekly M5 sample …")
        sys.path.insert(0, str(_ROOT))
        from scripts.prepare_m5_weekly import prepare_weekly
        daily_csv = _ROOT / "tests" / "integration" / "fixtures" / "m5_daily_sample.csv"
        if not daily_csv.exists():
            logger.error("Daily fixture not found at %s — cannot seed.", daily_csv)
            # Fall through to start API anyway
        else:
            prepare_weekly(daily_csv, demo_csv, n_series=10)

    # Step 2: Start API server in background
    logger.info("Starting API server on %s:%d …", API_HOST, API_PORT)
    api_proc = subprocess.Popen(
        [sys.executable, str(_HERE / "serve.py"),
         "--host", API_HOST, "--port", str(API_PORT),
         "--data-dir", str(DATA_DIR)],
        cwd=str(_ROOT),
    )

    if not wait_for_api(timeout=120):
        logger.error("API server did not become healthy within 120s")
        api_proc.terminate()
        sys.exit(1)

    logger.info("API server is healthy.")

    # Step 3: Seed via API
    if demo_csv.exists() and config_path.exists():
        champion = seed_via_api(demo_csv, config_path, DEMO_LOB)
        if champion:
            # Write sentinel
            SENTINEL.parent.mkdir(parents=True, exist_ok=True)
            SENTINEL.write_text(f"seeded:{DEMO_LOB}:champion={champion}\n")
            logger.info("Seed complete — sentinel written to %s", SENTINEL)
        else:
            logger.warning("Seeding failed — API will run without pre-loaded data.")
    else:
        logger.warning("Demo CSV or config missing — skipping seed.")

    # Step 4: Wait for the API process (keep container alive)
    logger.info("Seeding finished. API server running on :%d", API_PORT)
    try:
        api_proc.wait()
    except KeyboardInterrupt:
        api_proc.terminate()
        api_proc.wait()


if __name__ == "__main__":
    main()
