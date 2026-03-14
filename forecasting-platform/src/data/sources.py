"""
Data source connectors for the ingestion layer.

Supports file-based (CSV, Parquet, Delta), database (via SQLAlchemy),
and REST API sources.  Each connector implements ``read() -> pl.DataFrame``
and ``probe() -> bool`` for connectivity checks.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import polars as pl

logger = logging.getLogger(__name__)


class BaseSource(ABC):
    """Abstract data source."""

    @abstractmethod
    def read(self) -> pl.DataFrame:
        """Read data from the source and return a Polars DataFrame."""

    @abstractmethod
    def probe(self) -> bool:
        """Return True if the source is reachable / exists."""

    @property
    @abstractmethod
    def source_type(self) -> str:
        """Short identifier for the source type."""


class FileSource(BaseSource):
    """
    Reads CSV, Parquet, or Delta files.

    Format is auto-detected from extension unless explicitly provided.
    Supports glob patterns for directories of Parquet files.
    """

    _FORMAT_MAP = {
        ".csv": "csv",
        ".tsv": "csv",
        ".parquet": "parquet",
        ".pq": "parquet",
    }

    def __init__(
        self,
        path: str,
        format: Optional[str] = None,
        csv_separator: str = ",",
        csv_has_header: bool = True,
        columns: Optional[List[str]] = None,
    ):
        self.path = Path(path)
        self._format = format
        self.csv_separator = csv_separator
        self.csv_has_header = csv_has_header
        self.columns = columns

    @property
    def source_type(self) -> str:
        return "file"

    def _resolve_format(self) -> str:
        if self._format:
            return self._format
        if self.path.is_dir():
            return "parquet"  # directory of parquet files
        return self._FORMAT_MAP.get(self.path.suffix.lower(), "csv")

    def probe(self) -> bool:
        return self.path.exists()

    def read(self) -> pl.DataFrame:
        fmt = self._resolve_format()
        logger.info("Reading %s source: %s", fmt, self.path)

        if fmt == "csv":
            df = pl.read_csv(
                self.path,
                separator=self.csv_separator,
                has_header=self.csv_has_header,
                try_parse_dates=True,
                columns=self.columns,
            )
        elif fmt == "parquet":
            if self.path.is_dir():
                df = pl.read_parquet(
                    str(self.path / "*.parquet"),
                    columns=self.columns,
                )
            else:
                df = pl.read_parquet(
                    self.path,
                    columns=self.columns,
                )
        elif fmt == "delta":
            df = pl.read_delta(str(self.path), columns=self.columns)
        else:
            raise ValueError(f"Unsupported file format: {fmt}")

        logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), self.path)
        return df


class DatabaseSource(BaseSource):
    """
    Reads data via SQL query using a connection string.

    Requires ``connectorx`` (fast, zero-copy) or ``sqlalchemy`` as fallback.
    Connection strings are resolved from environment variables when prefixed
    with ``$`` (e.g. ``$DB_CONNECTION_STRING``).
    """

    def __init__(
        self,
        connection_string: str,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ):
        self._raw_connection_string = connection_string
        self.query = query
        self.params = params or {}

    @property
    def source_type(self) -> str:
        return "database"

    @property
    def connection_string(self) -> str:
        cs = self._raw_connection_string
        if cs.startswith("$"):
            env_var = cs[1:]
            resolved = os.environ.get(env_var)
            if not resolved:
                raise ValueError(
                    f"Environment variable {env_var!r} not set "
                    f"(referenced by connection_string={cs!r})"
                )
            return resolved
        return cs

    def probe(self) -> bool:
        try:
            cs = self.connection_string
            return bool(cs)
        except ValueError:
            return False

    def read(self) -> pl.DataFrame:
        cs = self.connection_string
        query = self.query
        logger.info("Executing database query: %.80s...", query)

        try:
            df = pl.read_database_uri(query=query, uri=cs)
        except Exception:
            # Fallback: try via SQLAlchemy + pandas
            try:
                import pandas as pd
                from sqlalchemy import create_engine, text

                engine = create_engine(cs)
                with engine.connect() as conn:
                    pdf = pd.read_sql(text(query), conn, params=self.params)
                df = pl.from_pandas(pdf)
            except ImportError:
                raise ImportError(
                    "Database source requires 'connectorx' or 'sqlalchemy'. "
                    "Install with: pip install connectorx  or  pip install sqlalchemy"
                )

        logger.info("Database query returned %d rows, %d columns", len(df), len(df.columns))
        return df


class APISource(BaseSource):
    """
    Reads data from a REST API endpoint.

    Expects JSON response with a list of records (or a key pointing to one).
    Supports basic pagination via ``next_url`` or ``offset`` strategies.
    """

    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        data_key: Optional[str] = None,
        pagination: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
    ):
        self.url = url
        self.headers = headers or {}
        self.data_key = data_key
        self.pagination = pagination
        self.timeout = timeout

    @property
    def source_type(self) -> str:
        return "api"

    def probe(self) -> bool:
        try:
            import urllib.request

            req = urllib.request.Request(self.url, method="HEAD")
            for k, v in self.headers.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=self.timeout):
                return True
        except Exception:
            return False

    def read(self) -> pl.DataFrame:
        import json
        import urllib.request

        logger.info("Fetching API source: %s", self.url)

        all_records: List[Dict] = []
        url = self.url

        while url:
            req = urllib.request.Request(url)
            for k, v in self.headers.items():
                req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = json.loads(resp.read().decode())

            if self.data_key:
                records = body.get(self.data_key, [])
            elif isinstance(body, list):
                records = body
            else:
                records = [body]

            all_records.extend(records)

            # Pagination
            url = None
            if self.pagination and self.pagination.get("strategy") == "next_url":
                next_key = self.pagination.get("next_key", "next")
                url = body.get(next_key) if isinstance(body, dict) else None

        if not all_records:
            return pl.DataFrame()

        df = pl.DataFrame(all_records)
        logger.info("API returned %d records", len(df))
        return df


def build_source(config: Dict[str, Any]) -> BaseSource:
    """
    Factory: build a source connector from a config dict.

    Expected keys:
      - type: "file" | "database" | "api"
      - (type-specific keys)
    """
    source_type = config.get("type", "file")

    if source_type == "file":
        return FileSource(
            path=config["path"],
            format=config.get("format"),
            csv_separator=config.get("csv_separator", ","),
            csv_has_header=config.get("csv_has_header", True),
            columns=config.get("columns"),
        )
    elif source_type == "database":
        return DatabaseSource(
            connection_string=config["connection_string"],
            query=config["query"],
            params=config.get("params"),
        )
    elif source_type == "api":
        return APISource(
            url=config["url"],
            headers=config.get("headers"),
            data_key=config.get("data_key"),
            pagination=config.get("pagination"),
            timeout=config.get("timeout", 30),
        )
    else:
        raise ValueError(f"Unknown source type: {source_type!r}")
