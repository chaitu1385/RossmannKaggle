# M5 Integration Baselines

This directory holds **blessed accuracy baselines** for the M5 end-to-end
integration tests. Each baseline is the ground truth against which PRs are
gated: no magic numbers, no hand-waved thresholds.

## Files

| File | Frequency | Fixture |
|------|-----------|---------|
| `m5_daily_baseline.json` | Daily | `tests/integration/fixtures/m5_daily_sample.csv` |

Weekly and monthly baselines will be added when the corresponding pytest
integration suites are added.

## Workflow

1. **A test compares the observed pipeline output against the baseline**
   (champion WMAPE, per-model WMAPE, FVA vs naive, calibration). Tolerances
   live in `tests/integration/baseline.py`.
2. **Fixture + config hashes are verified first** — if they drift, the
   baseline is considered stale and the test fails with a clear pointer to
   the bless script.
3. **To intentionally update** (new model, improved features):
   ```bash
   python scripts/bless_m5_baseline.py --frequency daily
   git diff tests/integration/baselines/m5_daily_baseline.json
   ```
   The diff is a reviewable record of "what this platform now delivers".

## Field reference

| Field | Meaning |
|-------|---------|
| `frequency` | `daily` \| `weekly` \| `monthly` |
| `fixture_sha256` | SHA-256 of the CSV fixture at bless time |
| `config_sha256` | SHA-256 of the YAML config at bless time |
| `seed` | Deterministic-seed value used by the pipeline |
| `per_model_wmape` | `{model_id: wmape}` for every ranked model |
| `champion_model` / `champion_wmape` | Winner of the walk-forward backtest |
| `naive_wmape` | Seasonal-naive WMAPE, used for FVA |
| `fva_vs_naive` | `naive_wmape - champion_wmape` (must stay positive) |
| `calibration_p50_coverage`, `calibration_p90_coverage` | Optional PI coverage |
| `blessed_at` | ISO-8601 UTC timestamp |
| `notes` | Free-form rationale for this bless |
