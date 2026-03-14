# Implementation Plan — Tier 1 Features

> **Status:** Tier 1 complete. See `Future_Ideas.md` for Tier 2/3 roadmap.
> **See also:** Data Ingestion and CI/CD plans below.

---

## Tier 1 Features (COMPLETED)

1. **External Regressors / Promotional Calendar** — ML models use promotions, holidays, price indices
2. **RBAC + Audit Trail** — JWT auth, 5 roles, 11 permissions, append-only audit log
3. **FVA Analysis** — Layer-by-layer accuracy attribution (naive → stat → ML → override)

---

## Infrastructure Plan A: Data Ingestion Robustness

### Goal
Replace the basic file-based `DataLoader` with a production-grade ingestion layer that supports multiple sources (files + databases), schema validation, configurable data quality scoring, and scheduled execution.

### Current State
- `src/data/loader.py` — 44-line class that reads `train.csv`, `test.csv`, `store.csv` via pandas
- `src/data/preprocessor.py` — basic null-filling and filtering (Open==1, Sales>0)
- No schema enforcement at load time (except `ProductMasterLoader` which checks required columns)
- No data quality scoring, outlier detection, or profiling
- `validate_regressors()` in `regressors.py` is the most sophisticated validation (5-point checks)
- No database connectivity

### Files to Create
1. `src/data/sources.py` — Source connectors (file, database, API)
2. `src/data/schema_validator.py` — Schema enforcement + type checking
3. `src/data/quality.py` — Data quality scoring engine
4. `src/data/ingestion.py` — Orchestrator (load → validate → score → preprocess)
5. `src/config/ingestion_config.py` — Ingestion configuration dataclass (or extend schema.py)
6. `tests/test_data_ingestion.py` — Tests

### Files to Modify
7. `src/config/schema.py` — Add `IngestionConfig` with source definitions, quality thresholds
8. `src/data/__init__.py` — Export new classes
9. `src/pipeline/forecast.py` — Use new ingestion layer instead of raw DataLoader
10. `src/pipeline/backtest.py` — Same

### Design

**Source connectors (`src/data/sources.py`):**
```python
class BaseSource(ABC):
    def read(self) -> pl.DataFrame: ...
    def probe(self) -> bool: ...       # connectivity check

class FileSource(BaseSource):
    """CSV, Parquet, Delta — auto-detected from extension."""
    def __init__(self, path: str, format: Optional[str] = None): ...

class DatabaseSource(BaseSource):
    """SQL query execution via SQLAlchemy or connector string."""
    def __init__(self, connection_string: str, query: str, params: dict = {}): ...

class APISource(BaseSource):
    """REST API with pagination support."""
    def __init__(self, url: str, headers: dict, pagination: Optional[dict] = None): ...
```

**Schema validator (`src/data/schema_validator.py`):**
```python
@dataclass
class ColumnSpec:
    name: str
    dtype: str                    # "Utf8", "Float64", "Date", etc.
    required: bool = True
    nullable: bool = False
    allowed_values: Optional[set] = None
    min_value: Optional[float] = None
    max_value: Optional[float] = None

class SchemaValidator:
    def __init__(self, columns: List[ColumnSpec]): ...
    def validate(self, df: pl.DataFrame) -> ValidationResult: ...

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]            # blocking issues
    warnings: List[str]          # non-blocking issues
```

**Data quality scoring (`src/data/quality.py`):**
```python
@dataclass
class QualityCheckConfig:
    name: str
    severity: str = "warn"       # "block" | "warn" | "info"
    threshold: Optional[float] = None

class DataQualityScorer:
    def __init__(self, checks: List[QualityCheckConfig]): ...
    def score(self, df: pl.DataFrame) -> QualityReport: ...

@dataclass
class QualityReport:
    overall_score: float          # 0-100
    passed: bool                  # True if no "block" checks failed
    check_results: List[CheckResult]

@dataclass
class CheckResult:
    check_name: str
    passed: bool
    severity: str
    score: float
    detail: str
```

Built-in checks:
- `completeness` — % non-null per column (threshold: e.g. 95%)
- `uniqueness` — duplicate row detection
- `freshness` — max date within expected recency window
- `volume` — row count within expected range (±X% of historical mean)
- `outlier` — IQR-based outlier % (configurable multiplier)
- `schema_drift` — new/missing columns vs expected schema
- `value_range` — values within configured min/max bounds

**Ingestion config (addition to `src/config/schema.py`):**
```yaml
ingestion:
  sources:
    actuals:
      type: file                  # file | database | api
      path: data/actuals/
      format: parquet             # csv | parquet | delta
    store_metadata:
      type: database
      connection_string: ${DB_CONNECTION_STRING}
      query: "SELECT * FROM store_dim WHERE active = 1"

  schema:
    actuals:
      columns:
        - {name: series_id, dtype: Utf8, required: true}
        - {name: week, dtype: Date, required: true}
        - {name: quantity, dtype: Float64, required: true, min_value: 0}
        - {name: lob, dtype: Utf8, required: true}

  quality:
    checks:
      - {name: completeness, severity: block, threshold: 95}
      - {name: freshness, severity: block, threshold: 7}   # max 7 days stale
      - {name: volume, severity: warn, threshold: 20}       # ±20% of expected
      - {name: outlier, severity: warn, threshold: 5}       # max 5% outliers
      - {name: uniqueness, severity: block}
      - {name: schema_drift, severity: warn}
```

**Orchestrator (`src/data/ingestion.py`):**
```python
class IngestionPipeline:
    def __init__(self, config: IngestionConfig): ...

    def run(self, source_name: str) -> IngestionResult:
        """Load → validate schema → score quality → return result."""
        ...

    def run_all(self) -> Dict[str, IngestionResult]: ...

@dataclass
class IngestionResult:
    source_name: str
    data: pl.DataFrame
    schema_result: ValidationResult
    quality_report: QualityReport
    ingested_at: datetime
    row_count: int
    blocked: bool                 # True if any "block" check failed
```

### Implementation Order
1. Source connectors (FileSource first, DatabaseSource second)
2. Schema validator
3. Data quality scorer (built-in checks)
4. Ingestion orchestrator
5. Config schema extension
6. Pipeline integration (forecast + backtest)
7. Tests

### Key Decisions
- **Configurable per check**: Each quality check has a `severity` field (`block`/`warn`/`info`). The pipeline halts only on `block` failures.
- **Database support**: SQLAlchemy for broad DB compatibility. Connection strings via environment variables (never in config files).
- **Backward compatible**: If no `ingestion` section in config, falls back to current `DataLoader` behavior.
- **Quality reports**: Written as JSON alongside ingested data for audit trail.

---

## Infrastructure Plan B: CI/CD & Containerization

### Goal
Add reproducible builds, automated testing, and deployment pipelines. Support both GitHub Actions and Azure DevOps. Container-first deployment targeting Azure and generic Docker.

### Current State
- No Dockerfile, docker-compose, CI pipeline, or Makefile
- `setup.py` exists (basic package metadata)
- `requirements.txt` exists (comprehensive)
- Tests run via `pytest` manually
- Deployment is Fabric notebook-driven (no container story)

### Files to Create

**Docker:**
1. `forecasting-platform/Dockerfile` — Multi-stage build (test → production)
2. `forecasting-platform/Dockerfile.dev` — Development image with all optional deps
3. `forecasting-platform/docker-compose.yml` — Local dev stack (API + optional DuckDB)
4. `forecasting-platform/.dockerignore` — Exclude tests, docs, data, .git

**CI — GitHub Actions:**
5. `.github/workflows/ci.yml` — Lint + test + build on push/PR
6. `.github/workflows/deploy.yml` — Build image + push to registry + deploy (on tag/release)

**CI — Azure DevOps:**
7. `azure-pipelines.yml` — Equivalent pipeline for Azure DevOps

**Build tooling:**
8. `forecasting-platform/Makefile` — Common commands (test, lint, build, run)
9. `forecasting-platform/pyproject.toml` — Modern Python packaging (replace/augment setup.py)

### Design

**Dockerfile (multi-stage):**
```dockerfile
# Stage 1: Base with dependencies
FROM python:3.11-slim AS base
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY setup.py pyproject.toml ./

# Stage 2: Test runner (CI only, not shipped)
FROM base AS test
COPY tests/ tests/
RUN pip install pytest pytest-cov
RUN python -m pytest --ignore=tests/test_metrics.py --ignore=tests/test_feature_engineering.py -v

# Stage 3: Production image
FROM base AS production
RUN pip install --no-cache-dir gunicorn uvicorn[standard]
EXPOSE 8000
ENV API_DATA_DIR=/app/data API_METRICS_DIR=/app/data/metrics
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**GitHub Actions CI (`.github/workflows/ci.yml`):**
```yaml
name: CI
on:
  push:
    branches: [master, main, 'claude/**']
  pull_request:
    branches: [master, main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install ruff
      - run: ruff check forecasting-platform/src/

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11"}
      - run: pip install -r forecasting-platform/requirements.txt
      - run: pip install pytest pytest-cov
      - working-directory: forecasting-platform
        run: python -m pytest --ignore=tests/test_metrics.py --ignore=tests/test_feature_engineering.py -v --cov=src --cov-report=xml
      - uses: actions/upload-artifact@v4
        with: {name: coverage, path: forecasting-platform/coverage.xml}

  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t forecasting-platform:${{ github.sha }} forecasting-platform/
```

**GitHub Actions Deploy (`.github/workflows/deploy.yml`):**
```yaml
name: Deploy
on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      # Option A: Azure Container Registry
      - uses: azure/docker-login@v2
        with:
          login-server: ${{ secrets.ACR_LOGIN_SERVER }}
          username: ${{ secrets.ACR_USERNAME }}
          password: ${{ secrets.ACR_PASSWORD }}
      - run: |
          docker build -t ${{ secrets.ACR_LOGIN_SERVER }}/forecasting-platform:${{ github.ref_name }} forecasting-platform/
          docker push ${{ secrets.ACR_LOGIN_SERVER }}/forecasting-platform:${{ github.ref_name }}

      # Option B: GitHub Container Registry (uncomment to use instead)
      # - uses: docker/login-action@v3
      #   with:
      #     registry: ghcr.io
      #     username: ${{ github.actor }}
      #     password: ${{ secrets.GITHUB_TOKEN }}
      # - run: |
      #     docker build -t ghcr.io/${{ github.repository }}:${{ github.ref_name }} forecasting-platform/
      #     docker push ghcr.io/${{ github.repository }}:${{ github.ref_name }}
```

**Azure DevOps Pipeline (`azure-pipelines.yml`):**
```yaml
trigger:
  branches:
    include: [master, main]
  tags:
    include: ['v*']

pool:
  vmImage: 'ubuntu-latest'

stages:
  - stage: Test
    jobs:
      - job: LintAndTest
        steps:
          - task: UsePythonVersion@0
            inputs: {versionSpec: '3.11'}
          - script: |
              pip install -r forecasting-platform/requirements.txt
              pip install pytest pytest-cov ruff
          - script: ruff check forecasting-platform/src/
            displayName: Lint
          - script: |
              cd forecasting-platform
              python -m pytest --ignore=tests/test_metrics.py --ignore=tests/test_feature_engineering.py -v --cov=src
            displayName: Test

  - stage: Build
    dependsOn: Test
    jobs:
      - job: DockerBuild
        steps:
          - task: Docker@2
            inputs:
              containerRegistry: $(ACR_SERVICE_CONNECTION)
              repository: forecasting-platform
              command: buildAndPush
              Dockerfile: forecasting-platform/Dockerfile
              tags: |
                $(Build.BuildId)
                latest
```

**Makefile:**
```makefile
.PHONY: test lint build run clean

test:
	cd forecasting-platform && python -m pytest --ignore=tests/test_metrics.py --ignore=tests/test_feature_engineering.py -v

lint:
	ruff check forecasting-platform/src/

build:
	docker build -t forecasting-platform:latest forecasting-platform/

run:
	docker run -p 8000:8000 -v $(PWD)/data:/app/data forecasting-platform:latest

dev:
	docker compose -f forecasting-platform/docker-compose.yml up --build

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
```

**docker-compose.yml (local dev):**
```yaml
version: "3.9"
services:
  api:
    build:
      context: .
      dockerfile: Dockerfile.dev
    ports:
      - "8000:8000"
    volumes:
      - ./src:/app/src
      - ./data:/app/data
    environment:
      - API_DATA_DIR=/app/data
      - API_METRICS_DIR=/app/data/metrics
      - AUTH_ENABLED=false
    command: uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### Implementation Order
1. Dockerfile + .dockerignore (production image)
2. Makefile (local dev shortcuts)
3. docker-compose.yml + Dockerfile.dev (local dev)
4. GitHub Actions CI (lint + test + build)
5. GitHub Actions Deploy (tag-triggered)
6. Azure DevOps pipeline (equivalent)
7. pyproject.toml (modern packaging)

### Key Decisions
- **Multi-stage Docker**: Test stage runs in CI but isn't shipped to prod (smaller image)
- **Both CI providers**: GitHub Actions as primary, Azure DevOps as alternative — same test/build logic
- **Configurable deploy targets**: Azure ACR and GHCR as options; Azure AKS/ACA deployment via Helm or `az container` (documented, not automated yet)
- **No secrets in repo**: Connection strings, JWT secrets, registry credentials all via environment variables / CI secrets
- **Backward compatible**: Existing manual `pytest` workflow still works; CI is additive
