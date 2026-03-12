"""
spark_deploy.py — Unified production deployment pipeline for Spark / Fabric.

Chains the full forecast cycle in a single command:
  1. Pre-flight validation (data freshness, series count)
  2. Backtest (if --retrain or no existing champion)
  3. Champion selection
  4. Forecast (39-week horizon by default)
  5. Post-run checks (forecast row count)
  6. Audit log (deploy_log Delta table)

Usage (local / dev)
-------------------
python scripts/spark_deploy.py \\
    --config       configs/platform_config.yaml \\
    --fabric-config configs/fabric_config.yaml \\
    --data-dir     data/ \\
    --lob          rossmann

Usage (Fabric / spark-submit)
-----------------------------
spark-submit \\
    --master yarn --deploy-mode cluster \\
    scripts/spark_deploy.py \\
    --config         configs/platform_config.yaml \\
    --fabric-config  configs/fabric_config.yaml \\
    --lob            rossmann \\
    --workspace      my-fabric-ws \\
    --lakehouse      my-lakehouse \\
    --write-lakehouse \\
    --retrain

Force options
-------------
  --retrain           Always re-run backtest even if champion exists.
  --model lgbm_direct Skip backtest and use this model directly.
  --horizon 52        Override forecast horizon weeks.
"""

import argparse
import logging
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("spark_deploy")


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified Fabric deployment pipeline: backtest → forecast"
    )
    p.add_argument("--config",         default="configs/platform_config.yaml")
    p.add_argument("--fabric-config",  default="configs/fabric_config.yaml")
    p.add_argument("--lob",            default="rossmann")
    p.add_argument("--data-dir",       default="data/")
    p.add_argument("--workspace",      default="")
    p.add_argument("--lakehouse",      default="")
    p.add_argument("--write-lakehouse", action="store_true",
                   help="Write outputs to Fabric Lakehouse")
    p.add_argument("--retrain",        action="store_true",
                   help="Force backtest even if a champion already exists")
    p.add_argument("--model",          default="",
                   help="Skip backtest and use this champion model directly")
    p.add_argument("--models",         nargs="*",
                   help="Models to evaluate during backtest (default: from config)")
    p.add_argument("--horizon",        type=int, default=0,
                   help="Forecast horizon weeks (0 = from config)")
    p.add_argument("--write-mode",     default="upsert",
                   choices=["upsert", "overwrite_partition", "append"])
    p.add_argument("--max-staleness",  type=int, default=14,
                   help="Max allowed data staleness in days (0 = skip check)")
    p.add_argument("--min-series",     type=int, default=1,
                   help="Min distinct series required (0 = skip check)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Spark session ─────────────────────────────────────────────────────────
    from src.spark.session import get_or_create_spark
    spark = get_or_create_spark(app_name=f"DeployPipeline-{args.lob}")
    logger.info("SparkSession ready (version=%s)", spark.version)

    # ── Platform + fabric config ──────────────────────────────────────────────
    import yaml
    from src.config.loader import load_config
    from src.spark.series_builder import SparkSeriesBuilder

    config = load_config(args.config)
    config.lob = args.lob

    with open(args.fabric_config) as _f:
        fabric_yaml = yaml.safe_load(_f)

    # ── Load actuals ──────────────────────────────────────────────────────────
    import os
    ws = args.workspace or os.environ.get("FABRIC_WORKSPACE", "")
    lh_name = args.lakehouse or os.environ.get("FABRIC_LAKEHOUSE", "")

    if args.write_lakehouse and ws:
        from src.fabric.config import FabricConfig
        from src.fabric.lakehouse import FabricLakehouse
        fabric_cfg = FabricConfig(workspace=ws, lakehouse=lh_name)
        lh = FabricLakehouse(spark, fabric_cfg)
        actuals_raw = lh.read_table("actuals")
    else:
        from src.spark.loader import SparkDataLoader
        loader = SparkDataLoader(spark, args.data_dir)
        train_sdf, _, store_sdf = loader.read_rossmann_all()
        actuals_raw = train_sdf.join(store_sdf, on="Store", how="left")

    builder = SparkSeriesBuilder.from_config(fabric_yaml["series_builder"])
    actuals_sdf = builder.build(actuals_raw)
    logger.info(
        "Actuals: %d rows, %d series",
        actuals_sdf.count(),
        actuals_sdf.select("series_id").distinct().count(),
    )

    # ── Deploy ────────────────────────────────────────────────────────────────
    from src.fabric.deployment import DeploymentConfig, DeploymentOrchestrator

    deploy_cfg = DeploymentConfig(
        lob=args.lob,
        workspace=ws if args.write_lakehouse else "",
        lakehouse=lh_name if args.write_lakehouse else "",
        force_retrain=args.retrain or bool(args.model == ""),
        models=args.models,
        horizon_weeks=args.horizon,
        max_staleness_days=args.max_staleness,
        min_series_count=args.min_series,
        write_mode=args.write_mode,
    )

    # If a specific model was passed, inject it by monkey-patching _resolve_champion
    # so the orchestrator skips backtest entirely.
    orch = DeploymentOrchestrator(spark, config=config, deploy_config=deploy_cfg)

    if args.model:
        # Bypass backtest: the user knows which champion to use.
        original_resolve = orch._resolve_champion
        orch._resolve_champion = lambda _sdf: (args.model, False)
        logger.info("Champion overridden via --model: %s", args.model)

    result = orch.run(actuals_sdf=actuals_sdf)

    logger.info(
        "Deployment complete — run_id=%s champion=%s rows=%d retrained=%s warnings=%d",
        result.run_id,
        result.champion_model,
        result.n_forecast_rows,
        result.retrained,
        len(result.preflight_warnings),
    )

    spark.stop()
    return result


if __name__ == "__main__":
    main()
