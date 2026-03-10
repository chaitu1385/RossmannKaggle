"""Abstract base class for all SKU mapping discovery methods."""

from abc import ABC, abstractmethod
from typing import List

import polars as pl

from ..data.schemas import MappingCandidate


class BaseMethod(ABC):
    """
    Every discovery method must implement ``run()``.

    A method receives the full product master DataFrame and returns a flat
    list of ``MappingCandidate`` objects.  It is responsible only for its
    own logic — candidate fusion and scoring happen downstream.
    """

    #: Short identifier used in output column ``methods_matched``.
    name: str = "base"

    @abstractmethod
    def run(self, product_master: pl.DataFrame) -> List[MappingCandidate]:
        """
        Discover candidate (old_sku, new_sku) pairs from ``product_master``.

        Parameters
        ----------
        product_master:
            Polars DataFrame conforming to ``PRODUCT_MASTER_SCHEMA``.

        Returns
        -------
        List[MappingCandidate]
            Flat list of candidates with method-specific scores in [0, 1].
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
