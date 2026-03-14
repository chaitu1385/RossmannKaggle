"""Load and validate product master data into Polars DataFrames."""

from pathlib import Path
from typing import Union

import polars as pl


class ProductMasterLoader:
    """Loads product master data and normalises it to the pipeline's schema."""

    #: Columns that must be present for the pipeline to run at all.
    REQUIRED_COLUMNS = frozenset(
        {"sku_id", "product_family", "launch_date", "country", "segment", "status"}
    )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_csv(self, path: Union[str, Path]) -> pl.DataFrame:
        """
        Load from a CSV file.

        The ``country`` column may be a comma-separated string
        (``"USA,GBR,DEU"``) — it is automatically split into a
        ``List[Utf8]`` column.  Date columns must be in ``YYYY-MM-DD``
        format.
        """
        df = pl.read_csv(
            str(path),
            schema_overrides={"launch_date": pl.Utf8, "eol_date": pl.Utf8},
            null_values=["", "NA", "N/A", "None", "null"],
        )
        df = self._parse_dates(df)
        df = self._parse_country(df)
        self._validate(df)
        return df

    def load_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Accept a pre-built Polars DataFrame (e.g., from the mock generator).

        Runs validation only; no CSV parsing.
        """
        self._validate(df)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _parse_dates(self, df: pl.DataFrame) -> pl.DataFrame:
        for col in ("launch_date", "eol_date"):
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col)
                    .str.to_date(format="%Y-%m-%d", strict=False)
                    .alias(col)
                )
        return df

    def _parse_country(self, df: pl.DataFrame) -> pl.DataFrame:
        """Split 'USA,GBR,DEU' strings into proper List[Utf8] columns."""
        if "country" in df.columns and df["country"].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col("country")
                .str.split(",")
                .list.eval(pl.element().str.strip_chars())
                .alias("country")
            )
        return df

    def _validate(self, df: pl.DataFrame) -> None:
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"Product master is missing required columns: {sorted(missing)}"
            )
