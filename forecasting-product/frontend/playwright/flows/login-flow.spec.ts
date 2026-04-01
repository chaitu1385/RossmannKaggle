import { test, expect } from "@playwright/test";
import { mockApiRoutes } from "../helpers/mock-routes";
import { clearAuth } from "../helpers/auth";

test.describe("Login Flow", () => {
  test.beforeEach(async ({ page }) => {
    await mockApiRoutes(page);
  });

  test("successful login as admin redirects to home", async ({ page }) => {
    await page.goto("/login");
    await page.fill("#username", "admin_user");
    await page.selectOption("#role", "admin");
    await page.click('button[type="submit"]');

    // Should redirect to home page
    await page.waitForURL("/", { timeout: 10_000 });
    await expect(page.locator("main h1")).toContainText("Sales Forecasting Product");
  });

  test("successful login as data scientist", async ({ page }) => {
    await page.goto("/login");
    await page.fill("#username", "ds_user");
    await page.selectOption("#role", "data_scientist");
    await page.click('button[type="submit"]');

    await page.waitForURL("/", { timeout: 10_000 });
  });

  test("successful login as planner", async ({ page }) => {
    await page.goto("/login");
    await page.fill("#username", "planner1");
    await page.selectOption("#role", "planner");
    await page.click('button[type="submit"]');

    await page.waitForURL("/", { timeout: 10_000 });
  });

  test("login with empty username shows error", async ({ page }) => {
    await page.goto("/login");
    // Leave username empty, submit
    await page.click('button[type="submit"]');

    // Error message should appear
    await expect(page.locator("text=Username is required")).toBeVisible();
    // Should still be on login page
    expect(page.url()).toContain("/login");
  });

  test("login button shows loading state during submission", async ({ page }) => {
    // Delay the token response to observe loading state
    await page.route("http://localhost:8000/auth/token**", async (route) => {
      await new Promise((r) => setTimeout(r, 500));
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({
          access_token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0IiwiZXhwIjo5OTk5OTk5OTk5fQ.mock",
          token_type: "bearer",
        }),
      });
    });

    await page.goto("/login");
    await page.fill("#username", "test_user");
    await page.click('button[type="submit"]');

    // Button should show "Signing in..." text
    await expect(page.locator("button[type='submit']")).toContainText("Signing in");
  });

  test("all five roles are available in the role selector", async ({ page }) => {
    await page.goto("/login");
    const options = page.locator("#role option");
    await expect(options).toHaveCount(5);

    const values = await options.evaluateAll((els) =>
      els.map((el) => (el as HTMLOptionElement).value),
    );
    expect(values).toEqual([
      "admin",
      "data_scientist",
      "planner",
      "manager",
      "viewer",
    ]);
  });

  test("login persists auth to localStorage", async ({ page }) => {
    await page.goto("/login");
    await page.fill("#username", "persist_user");
    await page.selectOption("#role", "admin");
    await page.click('button[type="submit"]');
    await page.waitForURL("/", { timeout: 10_000 });

    // Verify localStorage has the token and user
    const token = await page.evaluate(() => localStorage.getItem("access_token"));
    const userStr = await page.evaluate(() => localStorage.getItem("user"));
    expect(token).toBeTruthy();
    expect(userStr).toBeTruthy();

    const user = JSON.parse(userStr!);
    expect(user.username).toBe("persist_user");
    expect(user.role).toBe("admin");
  });

  test("clearing auth logs user out", async ({ page }) => {
    // First login
    await page.goto("/login");
    await page.fill("#username", "logout_user");
    await page.selectOption("#role", "admin");
    await page.click('button[type="submit"]');
    await page.waitForURL("/", { timeout: 10_000 });

    // Clear auth
    await clearAuth(page);

    // Verify localStorage is empty
    const token = await page.evaluate(() => localStorage.getItem("access_token"));
    expect(token).toBeNull();
  });
});
