"""
SKUMappingPipeline — top-level orchestrator.

Wires together:
  ProductMasterLoader → [methods...]  → CandidateFusion → MappingWriter

Phase 1 methods : AttributeMatchingMethod, NamingConventionMethod
Phase 2 methods : + CurveFittingMethod, TemporalCovementMethod
"""

import logging
from typing import List, Optional

import polars as pl

from .data.loader import ProductMasterLoader
from .fusion.scorer import CandidateFusion, _FULL_WEIGHTS
from .methods.base import BaseMethod
from .methods.attribute_matching import AttributeMatchingMethod
from .methods.naming_parsing import NamingConventionMethod
from .methods.curve_fitting import CurveFittingMethod
from .methods.temporal_comovement import TemporalCovementMethod
from .output.writer import MappingWriter

logger = logging.getLogger(__name__)


def build_phase2_pipeline(
    sales_df: Optional[pl.DataFrame] = None,
    launch_window_days: int = 180,
    min_base_similarity: float = 0.70,
    min_confidence: str = "Low",
    window_weeks: int = 13,
) -> "SKUMappingPipeline":
    """
    Phase 2 pipeline: all four methods (attribute + naming + curve + temporal).

    Both ``CurveFittingMethod`` and ``TemporalCovementMethod`` require sales
    history data.  If ``sales_df`` is None they silently return no candidates
    and the pipeline degrades to Phase 1 behaviour.

    Parameters
    ----------
    sales_df:
        Weekly sales history DataFrame with columns
        ``[sku_id, week, quantity]``.
    launch_window_days, min_base_similarity, min_confidence:
        Same as ``build_phase1_pipeline``.
    window_weeks:
        Half-width of the transition window for curve and temporal methods.
    """
    methods: List[BaseMethod] = [
        AttributeMatchingMethod(launch_window_days=launch_window_days),
        NamingConventionMethod(min_base_similarity=min_base_similarity),
        CurveFittingMethod(
            sales_df=sales_df,
            window_weeks=window_weeks,
            launch_window_days=launch_window_days,
        ),
        TemporalCovementMethod(
            sales_df=sales_df,
            window_weeks=window_weeks,
            launch_window_days=launch_window_days,
        ),
    ]
    # _FULL_WEIGHTS covers all 4 methods; fusion auto-normalises to whichever
    # returned candidates for each specific pair.
    fusion = CandidateFusion(weights=_FULL_WEIGHTS, min_confidence=min_confidence)
    return SKUMappingPipeline(methods=methods, fusion=fusion)


def build_phase1_pipeline(
    launch_window_days: int = 180,
    min_base_similarity: float = 0.70,
    min_confidence: str = "Low",
) -> "SKUMappingPipeline":
    """
    Convenience factory that creates the Phase 1 MVP pipeline.

    Parameters
    ----------
    launch_window_days:
        Maximum days between old-SKU launch and new-SKU launch to form a
        candidate pair in attribute matching.
    min_base_similarity:
        Minimum rapidfuzz WRatio score (0–1) for the naming method.
    min_confidence:
        Drop all candidates below this confidence level from the output.
        Accepts ``"High"``, ``"Medium"``, ``"Low"``, or ``"Very Low"``.
    """
    methods: List[BaseMethod] = [
        AttributeMatchingMethod(launch_window_days=launch_window_days),
        NamingConventionMethod(min_base_similarity=min_base_similarity),
    ]
    fusion = CandidateFusion(min_confidence=min_confidence)
    return SKUMappingPipeline(methods=methods, fusion=fusion)


class SKUMappingPipeline:
    """
    End-to-end SKU mapping discovery pipeline.

    Usage
    -----
    >>> pipeline = build_phase1_pipeline()
    >>> df = pipeline.run_from_csv("data/product_master.csv", "output/mappings.csv")

    Or with a pre-loaded Polars DataFrame:

    >>> from src.sku_mapping.data.mock_generator import generate_product_master
    >>> pm = generate_product_master()
    >>> df = pipeline.run(pm, "output/mappings.csv")
    """

    def __init__(
        self,
        methods: List[BaseMethod],
        fusion: Optional[CandidateFusion] = None,
        writer: Optional[MappingWriter] = None,
    ):
        self.methods = methods
        self.fusion = fusion or CandidateFusion()
        self.writer = writer or MappingWriter()
        self._loader = ProductMasterLoader()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        product_master: pl.DataFrame,
        output_path: Optional[str] = None,
    ) -> pl.DataFrame:
        """
        Run the pipeline on a pre-loaded product master DataFrame.

        Parameters
        ----------
        product_master:
            Polars DataFrame conforming to ``PRODUCT_MASTER_SCHEMA``.
        output_path:
            If provided, the output CSV is written to this path.

        Returns
        -------
        pl.DataFrame
            The full mapping table (design doc §5 schema).
        """
        product_master = self._loader.load_dataframe(product_master)

        logger.info(
            "Pipeline starting: %d SKUs, %d methods",
            len(product_master),
            len(self.methods),
        )

        # ── Discover candidates ──────────────────────────────────────────────
        all_candidates = []
        for method in self.methods:
            candidates = method.run(product_master)
            logger.info(
                "Method '%s': %d candidates", method.name, len(candidates)
            )
            all_candidates.extend(candidates)

        logger.info("Total raw candidates: %d", len(all_candidates))

        # ── Fuse ────────────────────────────────────────────────────────────
        records = self.fusion.fuse(all_candidates, product_master)
        logger.info("After fusion: %d unique mappings", len(records))

        # ── Summarise confidence breakdown ───────────────────────────────────
        if records:
            from collections import Counter
            breakdown = Counter(r.confidence_level for r in records)
            for level in ("High", "Medium", "Low", "Very Low"):
                if level in breakdown:
                    logger.info("  %s: %d", level, breakdown[level])

        # ── Write output ─────────────────────────────────────────────────────
        output_df = self.writer.to_polars(records)

        if output_path:
            self.writer.save_csv(output_df, output_path)
            logger.info("Saved mapping table to %s", output_path)

        return output_df

    def run_from_csv(
        self,
        product_master_path: str,
        output_path: Optional[str] = None,
    ) -> pl.DataFrame:
        """Load product master from CSV, then run the pipeline."""
        logger.info("Loading product master from %s", product_master_path)
        product_master = self._loader.load_csv(product_master_path)
        return self.run(product_master, output_path=output_path)
