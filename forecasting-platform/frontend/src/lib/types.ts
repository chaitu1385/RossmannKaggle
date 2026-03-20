// ──────────────────────────────────────────────────────────────────────────────
// TypeScript interfaces matching FastAPI response schemas (src/api/schemas.py)
// ──────────────────────────────────────────────────────────────────────────────

// ── Auth ─────────────────────────────────────────────────────────────────────

export type Role = "admin" | "data_scientist" | "planner" | "manager" | "viewer";

export interface TokenResponse {
  access_token: string;
  token_type: string;
}

export interface User {
  username: string;
  role: Role;
  permissions: string[];
}

// ── Health ────────────────────────────────────────────────────────────────────

export interface HealthResponse {
  status: string;
  version: string;
}

// ── Forecast ─────────────────────────────────────────────────────────────────

export interface ForecastPoint {
  series_id: string;
  week: string;
  forecast: number;
  model?: string;
  lob?: string;
}

export interface ForecastResponse {
  lob: string;
  series_count: number;
  forecast_origin?: string;
  points: ForecastPoint[];
}

// ── Metrics / Leaderboard ────────────────────────────────────────────────────

export interface LeaderboardEntry {
  model: string;
  wmape: number;
  normalized_bias: number;
  rank: number;
  n_series: number;
}

export interface LeaderboardResponse {
  lob: string;
  run_type: string;
  entries: LeaderboardEntry[];
}

// ── Drift ────────────────────────────────────────────────────────────────────

export interface DriftAlertItem {
  series_id: string;
  metric: string;
  severity: "warning" | "critical";
  current_value: number;
  baseline_value: number;
  message: string;
}

export interface DriftResponse {
  lob: string;
  n_critical: number;
  n_warning: number;
  alerts: DriftAlertItem[];
}

// ── Analysis ─────────────────────────────────────────────────────────────────

export interface AnalysisResponse {
  lob_name: string;
  time_column: string;
  target_column: string;
  id_columns: string[];
  n_series: number;
  n_rows: number;
  date_range_start: string;
  date_range_end: string;
  frequency: string;
  overall_forecastability: number;
  forecastability_distribution: Record<string, number>;
  demand_classes: Record<string, number>;
  detected_hierarchies: Record<string, unknown>[];
  recommended_config_yaml: string;
  config_reasoning: string[];
  hypotheses: string[];
  llm_narrative?: string;
  llm_risk_factors?: string[];
}

// ── AI: NL Query ─────────────────────────────────────────────────────────────

export interface NLQueryRequest {
  series_id: string;
  question: string;
  lob: string;
}

export interface NLQueryResponse {
  answer: string;
  supporting_data: Record<string, unknown>;
  confidence: "high" | "medium" | "low";
  sources_used: string[];
}

// ── AI: Triage ───────────────────────────────────────────────────────────────

export interface TriageRequest {
  lob: string;
  run_type?: string;
  severity_filter?: string;
  max_alerts?: number;
}

export interface TriagedAlertItem {
  series_id: string;
  metric: string;
  severity: string;
  business_impact_score: number;
  suggested_action: string;
  reasoning: string;
  original_message: string;
}

export interface TriageResponse {
  lob: string;
  executive_summary: string;
  total_alerts: number;
  critical_count: number;
  warning_count: number;
  ranked_alerts: TriagedAlertItem[];
}

// ── AI: Config Tuner ─────────────────────────────────────────────────────────

export interface ConfigTuneRequest {
  lob: string;
  run_type?: string;
}

export interface ConfigRecommendationItem {
  field_path: string;
  current_value: unknown;
  recommended_value: unknown;
  reasoning: string;
  expected_impact: string;
  risk: "low" | "medium" | "high";
}

export interface ConfigTuneResponse {
  lob: string;
  recommendations: ConfigRecommendationItem[];
  overall_assessment: string;
  risk_summary: string;
}

// ── AI: Commentary ───────────────────────────────────────────────────────────

export interface CommentaryRequest {
  lob: string;
  run_type?: string;
  period_start?: string;
  period_end?: string;
}

export interface KeyMetricItem {
  name: string;
  value: number;
  unit: string;
  trend: "improving" | "stable" | "degrading";
}

export interface CommentaryResponse {
  lob: string;
  executive_summary: string;
  key_metrics: KeyMetricItem[];
  exceptions: string[];
  action_items: string[];
}

// ── Audit ────────────────────────────────────────────────────────────────────

export interface AuditEvent {
  audit_id: string;
  timestamp: string;
  user_id: string;
  user_email: string;
  user_role: string;
  action: string;
  resource_type: string;
  resource_id: string;
  status: string;
  old_value?: string;
  new_value?: string;
  ip_address: string;
  request_id: string;
  error_message?: string;
}

export interface AuditResponse {
  count: number;
  events: AuditEvent[];
}
