"""
serve.py — Start the forecasting platform REST API server.

Usage
-----
python scripts/serve.py
python scripts/serve.py --port 8080 --data-dir data/ --metrics-dir data/metrics/
python scripts/serve.py --host 0.0.0.0 --port 8000 --reload   # dev mode

Environment variable overrides
-------------------------------
API_DATA_DIR      Path to data directory.
API_METRICS_DIR   Path to metrics store.
API_VERSION       Version string embedded in /health.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("serve")


def parse_args():
    p = argparse.ArgumentParser(description="Forecasting Platform REST API server")
    p.add_argument("--host",        default="0.0.0.0",  help="Bind host")
    p.add_argument("--port",        type=int, default=8000, help="Bind port")
    p.add_argument("--data-dir",    default=os.environ.get("API_DATA_DIR", "data/"),
                   help="Root data directory for forecasts")
    p.add_argument("--metrics-dir", default=os.environ.get("API_METRICS_DIR", "data/metrics/"),
                   help="Metrics store directory")
    p.add_argument("--reload",      action="store_true", help="Enable hot-reload (dev mode)")
    p.add_argument("--workers",     type=int, default=1,
                   help="Number of Uvicorn worker processes (production)")
    return p.parse_args()


def main():
    args = parse_args()

    # Patch environment so create_app picks up the right directories
    os.environ["API_DATA_DIR"]    = args.data_dir
    os.environ["API_METRICS_DIR"] = args.metrics_dir

    from src.api.app import create_app
    import uvicorn

    app = create_app(data_dir=args.data_dir, metrics_dir=args.metrics_dir)

    logger.info("Starting Forecasting Platform API on %s:%d", args.host, args.port)
    logger.info("  data_dir    = %s", args.data_dir)
    logger.info("  metrics_dir = %s", args.metrics_dir)
    logger.info("  Swagger UI  = http://%s:%d/docs", args.host, args.port)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )


if __name__ == "__main__":
    main()
