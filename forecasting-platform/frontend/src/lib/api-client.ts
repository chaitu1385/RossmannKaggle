// ──────────────────────────────────────────────────────────────────────────────
// Centralized API client for FastAPI backend
// ──────────────────────────────────────────────────────────────────────────────

import type {
  HealthResponse,
  TokenResponse,
  ForecastResponse,
  LeaderboardResponse,
  DriftResponse,
  AnalysisResponse,
  NLQueryRequest,
  NLQueryResponse,
  TriageRequest,
  TriageResponse,
  ConfigTuneRequest,
  ConfigTuneResponse,
  CommentaryRequest,
  CommentaryResponse,
  AuditResponse,
} from "./types";

const BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

function getToken(): string | null {
  if (typeof window === "undefined") return null;
  return localStorage.getItem("access_token");
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  // Don't set Content-Type for FormData (browser sets boundary automatically)
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new ApiError(res.status, text);
  }

  return res.json() as Promise<T>;
}

// ── Endpoints ────────────────────────────────────────────────────────────────

export const api = {
  // Health
  health: () => request<HealthResponse>("/health"),

  // Auth
  getToken: (username: string, role: string = "viewer") =>
    request<TokenResponse>(
      `/auth/token?username=${encodeURIComponent(username)}&role=${encodeURIComponent(role)}`,
      { method: "POST" },
    ),

  // Forecasts
  getForecasts: (lob: string, params?: { series_id?: string; horizon?: number }) => {
    const searchParams = new URLSearchParams();
    if (params?.series_id) searchParams.set("series_id", params.series_id);
    if (params?.horizon) searchParams.set("horizon", String(params.horizon));
    const qs = searchParams.toString();
    return request<ForecastResponse>(`/forecast/${encodeURIComponent(lob)}${qs ? `?${qs}` : ""}`);
  },

  getForecastSeries: (lob: string, seriesId: string) =>
    request<ForecastResponse>(
      `/forecast/${encodeURIComponent(lob)}/${encodeURIComponent(seriesId)}`,
    ),

  // Metrics
  getLeaderboard: (lob: string, runType: string = "backtest") =>
    request<LeaderboardResponse>(
      `/metrics/leaderboard/${encodeURIComponent(lob)}?run_type=${encodeURIComponent(runType)}`,
    ),

  getDrift: (
    lob: string,
    params?: { run_type?: string; baseline_weeks?: number; recent_weeks?: number },
  ) => {
    const searchParams = new URLSearchParams();
    searchParams.set("run_type", params?.run_type || "backtest");
    if (params?.baseline_weeks) searchParams.set("baseline_weeks", String(params.baseline_weeks));
    if (params?.recent_weeks) searchParams.set("recent_weeks", String(params.recent_weeks));
    return request<DriftResponse>(
      `/metrics/drift/${encodeURIComponent(lob)}?${searchParams.toString()}`,
    );
  },

  // Analysis (file upload)
  analyze: (file: File, lobName: string = "analyzed", llmEnabled: boolean = false) => {
    const formData = new FormData();
    formData.append("file", file);
    return request<AnalysisResponse>(
      `/analyze?lob_name=${encodeURIComponent(lobName)}&llm_enabled=${llmEnabled}`,
      { method: "POST", body: formData },
    );
  },

  // AI endpoints
  aiExplain: (body: NLQueryRequest) =>
    request<NLQueryResponse>("/ai/explain", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  aiTriage: (body: TriageRequest) =>
    request<TriageResponse>("/ai/triage", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  aiRecommendConfig: (body: ConfigTuneRequest) =>
    request<ConfigTuneResponse>("/ai/recommend-config", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  aiCommentary: (body: CommentaryRequest) =>
    request<CommentaryResponse>("/ai/commentary", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  // Audit
  getAuditLog: (params?: { action?: string; resource_type?: string; limit?: number }) => {
    const searchParams = new URLSearchParams();
    if (params?.action) searchParams.set("action", params.action);
    if (params?.resource_type) searchParams.set("resource_type", params.resource_type);
    if (params?.limit) searchParams.set("limit", String(params.limit));
    const qs = searchParams.toString();
    return request<AuditResponse>(`/audit${qs ? `?${qs}` : ""}`);
  },
};

export { ApiError };
