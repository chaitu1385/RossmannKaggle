"""
SKU Mapping Discovery Pipeline (Phase 1 — MVP)
===============================================

Algorithmic first-pass for product transition mappings where no system of
record exists.  Feeds into the forecasting platform's Layer 1 (data
preparation) for synthetic history construction.

Quick start
-----------
>>> from src.sku_mapping import build_phase1_pipeline
>>> from src.sku_mapping.data.mock_generator import generate_product_master
>>>
>>> pipeline = build_phase1_pipeline()
>>> product_master = generate_product_master()
>>> mapping_df = pipeline.run(product_master, output_path="output/mappings.csv")
"""

from .data.schemas import MappingCandidate, MappingRecord
from .pipeline import SKUMappingPipeline, build_phase1_pipeline

__all__ = [
    "SKUMappingPipeline",
    "build_phase1_pipeline",
    "MappingCandidate",
    "MappingRecord",
]
