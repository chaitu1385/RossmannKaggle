import type { Page } from "@playwright/test";

// ──────────────────────────────────────────────────────────────────────────────
// Authentication helpers for Playwright tests.
// Seeds localStorage with a mock JWT and user object so tests can bypass
// the login form when the test scenario doesn't involve login itself.
// ──────────────────────────────────────────────────────────────────────────────

const MOCK_TOKEN =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0X3VzZXIiLCJyb2xlIjoiYWRtaW4iLCJleHAiOjk5OTk5OTk5OTl9.mock";

interface SeedAuthOptions {
  username?: string;
  role?: string;
  permissions?: string[];
}

const ALL_PERMISSIONS = [
  "VIEW_FORECASTS",
  "VIEW_METRICS",
  "VIEW_AUDIT_LOG",
  "CREATE_OVERRIDE",
  "DELETE_OVERRIDE",
  "APPROVE_OVERRIDE",
  "RUN_BACKTEST",
  "RUN_PIPELINE",
  "PROMOTE_MODEL",
  "MODIFY_CONFIG",
  "MANAGE_USERS",
];

const ROLE_PERMISSIONS: Record<string, string[]> = {
  admin: ALL_PERMISSIONS,
  data_scientist: [
    "VIEW_FORECASTS",
    "VIEW_METRICS",
    "VIEW_AUDIT_LOG",
    "RUN_BACKTEST",
    "RUN_PIPELINE",
    "PROMOTE_MODEL",
    "MODIFY_CONFIG",
  ],
  planner: ["VIEW_FORECASTS", "VIEW_METRICS", "CREATE_OVERRIDE", "DELETE_OVERRIDE"],
  manager: ["VIEW_FORECASTS", "VIEW_METRICS", "VIEW_AUDIT_LOG", "APPROVE_OVERRIDE"],
  viewer: ["VIEW_FORECASTS", "VIEW_METRICS"],
};

/**
 * Seed localStorage with auth tokens so the page loads as an authenticated user.
 * Must be called BEFORE navigating to any page.
 */
export async function seedAuth(page: Page, opts: SeedAuthOptions = {}) {
  const { username = "test_admin", role = "admin" } = opts;
  const permissions = opts.permissions ?? ROLE_PERMISSIONS[role] ?? ALL_PERMISSIONS;

  // Navigate to the origin first so localStorage is accessible
  await page.goto("/");
  await page.evaluate(
    ({ token, user }) => {
      localStorage.setItem("access_token", token);
      localStorage.setItem("user", JSON.stringify(user));
    },
    {
      token: MOCK_TOKEN,
      user: { username, role, permissions },
    },
  );
}

/**
 * Log in via the login form UI. Useful for testing the login flow itself.
 */
export async function loginViaUI(
  page: Page,
  username: string,
  role: string,
) {
  await page.goto("/login");
  await page.fill("#username", username);
  await page.selectOption("#role", role);
  await page.click('button[type="submit"]');
  // Wait for redirect to home
  await page.waitForURL("/", { timeout: 10_000 });
}

/**
 * Log in as admin via localStorage seeding, then navigate to a target page.
 */
export async function loginAsAdmin(page: Page, targetPath = "/") {
  await seedAuth(page, { username: "test_admin", role: "admin" });
  await page.goto(targetPath);
}

/**
 * Log in as planner via localStorage seeding, then navigate to a target page.
 */
export async function loginAsPlanner(page: Page, targetPath = "/") {
  await seedAuth(page, { username: "test_planner", role: "planner" });
  await page.goto(targetPath);
}

/**
 * Log in as data scientist via localStorage seeding.
 */
export async function loginAsDataScientist(page: Page, targetPath = "/") {
  await seedAuth(page, { username: "test_ds", role: "data_scientist" });
  await page.goto(targetPath);
}

/**
 * Clear auth state from localStorage.
 */
export async function clearAuth(page: Page) {
  await page.evaluate(() => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("user");
  });
}
