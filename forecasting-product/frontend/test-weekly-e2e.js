/**
 * E2E test — Visit all UI pages with weekly_retail LOB data and capture screenshots.
 * 
 * Prerequisites:
 *   - Backend running on :8000 with backtest + forecast data for "weekly_retail"
 *   - Frontend running on :3000
 *   - npx playwright install chromium (already done)
 *
 * Run: node test-weekly-e2e.js
 */
const { chromium } = require('playwright');
const path = require('path');

const BASE = 'http://localhost:3000';
const RESULTS_DIR = path.join(__dirname, 'test-results', 'weekly');
const LOB = 'weekly_retail';

async function ensureDir(dir) {
  const fs = require('fs');
  fs.mkdirSync(dir, { recursive: true });
}

async function screenshot(page, name) {
  await page.screenshot({ path: path.join(RESULTS_DIR, `${name}.png`), fullPage: true });
  console.log(`  📸 ${name}.png`);
}

async function main() {
  await ensureDir(RESULTS_DIR);
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
  const page = await context.newPage();

  // ──────────────── 1. DATA ONBOARDING ────────────────
  console.log('\n1. Data Onboarding — Upload + Analyze');
  await page.goto(`${BASE}/data-onboarding`);
  await page.waitForLoadState('networkidle');

  // Set LOB name
  const lobInput = page.locator('input[type="text"]').first();
  await lobInput.fill(LOB);

  // Upload CSV — this auto-triggers analysis via onFileSelect
  const csvPath = path.resolve(__dirname, '..', 'data', 'demo', 'weekly_retail.csv');
  const fileInput = page.locator('input[type="file"]').first();
  await fileInput.setInputFiles(csvPath);
  console.log('  File uploaded, analysis auto-started...');

  // Wait for analysis results to appear (auto-triggered by file select)
  await page.waitForSelector('text=/Schema Detection|Forecastability|series/i', { timeout: 60000 });
  await page.waitForTimeout(1500);
  await screenshot(page, '01-onboarding-analyzed-top');

  // Scroll to see forecastability + demand classification
  await page.evaluate(() => window.scrollBy(0, 600));
  await page.waitForTimeout(500);
  await screenshot(page, '02-onboarding-forecastability');

  // Scroll more to see config reasoning + hypotheses
  await page.evaluate(() => window.scrollBy(0, 600));
  await page.waitForTimeout(500);
  await screenshot(page, '03-onboarding-config');

  // Scroll to bottom — Pipeline Execution panel
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(500);
  await screenshot(page, '04-onboarding-pipeline-panel');

  // ──────────────── 1b. RUN BACKTEST FROM UI ────────────────
  console.log('\n1b. Pipeline Execution — Run Backtest from UI');
  // The pipeline panel now auto-receives the uploaded file
  // Scroll to the pipeline panel and wait for file to be ready
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(1000);
  
  const backtestBtn = page.getByRole('button', { name: /run backtest/i });
  if (await backtestBtn.isVisible()) {
    // Wait for the button to become enabled (file should be auto-loaded)
    try {
      await backtestBtn.waitFor({ state: 'attached', timeout: 5000 });
      const isDisabled = await backtestBtn.isDisabled();
      if (isDisabled) {
        console.log('  ⚠️ Run Backtest button still disabled — file may not be passed to Pipeline panel');
        await screenshot(page, '05-pipeline-btn-disabled');
      } else {
        await backtestBtn.click();
        console.log('  Running backtest... (this may take a while)');
        try {
          await page.waitForSelector('text=/completed|champion|Backtest Result/i', { timeout: 180000 });
          await page.waitForTimeout(1500);
          await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
          await page.waitForTimeout(500);
          await screenshot(page, '05-pipeline-backtest-done');
        } catch (e) {
          console.log('  ⚠️ Backtest completion not detected in time');
          await screenshot(page, '05-pipeline-backtest-timeout');
        }
      }
    } catch (e) {
      console.log('  ⚠️ Run Backtest button not ready');
    }
  } else {
    console.log('  ⚠️ Run Backtest button not found');
  }

  // ──────────────── 2. BACKTEST RESULTS ────────────────
  console.log('\n2. Backtest Results');
  await page.goto(`${BASE}/backtest`);
  await page.waitForLoadState('networkidle');

  // Enter LOB name and load
  const btLob = page.locator('input[type="text"]').first();
  await btLob.fill(LOB);
  const loadBtn = page.getByRole('button', { name: 'Load', exact: true });
  if (await loadBtn.isVisible()) {
    await loadBtn.click();
    // Wait for the leaderboard data to render (Recharts + metric cards)
    try {
      await page.waitForSelector('text=/Model Leaderboard/i', { timeout: 5000 });
      // Give Recharts time to render the chart
      await page.waitForTimeout(3000);
    } catch (e) {
      console.log('  ⚠️ Leaderboard data not detected');
    }
  }
  await screenshot(page, '06-backtest-loaded');

  // Scroll for full leaderboard
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(500);
  await screenshot(page, '07-backtest-bottom');

  // ──────────────── 3. FORECAST VIEWER ────────────────
  console.log('\n3. Forecast Viewer');
  await page.goto(`${BASE}/forecast`);
  await page.waitForLoadState('networkidle');

  const fcLob = page.locator('input[type="text"]').first();
  await fcLob.fill(LOB);
  const loadFcBtn = page.getByRole('button', { name: 'Load Forecasts', exact: true });
  if (await loadFcBtn.isVisible()) {
    await loadFcBtn.click();
    try {
      // Wait for forecast data to load and render (Recharts needs time)
      await page.waitForSelector('text=/Series Count|Forecast Origin|forecast/i', { timeout: 15000 });
      await page.waitForTimeout(3000);
    } catch (e) {
      console.log('  ⚠️ Forecast data not detected');
    }
  }
  await screenshot(page, '08-forecast-loaded');

  // Scroll for more content
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(500);
  await screenshot(page, '09-forecast-bottom');

  // ──────────────── 4. SERIES EXPLORER ────────────────
  console.log('\n4. Series Explorer');
  await page.goto(`${BASE}/series-explorer`);
  await page.waitForLoadState('networkidle');
  await screenshot(page, '10-series-explorer');

  // ──────────────── 5. HIERARCHY ────────────────
  console.log('\n5. Hierarchy Manager');
  await page.goto(`${BASE}/hierarchy`);
  await page.waitForLoadState('networkidle');
  await screenshot(page, '11-hierarchy');

  // ──────────────── 6. SKU TRANSITIONS ────────────────
  console.log('\n6. SKU Transitions');
  await page.goto(`${BASE}/sku-transitions`);
  await page.waitForLoadState('networkidle');
  await screenshot(page, '12-sku-transitions');

  // ──────────────── 7. PLATFORM HEALTH ────────────────
  console.log('\n7. Platform Health');
  await page.goto(`${BASE}/health`);
  await page.waitForLoadState('networkidle');
  await screenshot(page, '13-health');

  // ──────────────── 8. S&OP MEETING ────────────────
  console.log('\n8. S&OP Meeting');
  await page.goto(`${BASE}/sop`);
  await page.waitForLoadState('networkidle');
  await screenshot(page, '14-sop');

  // ──────────────── DONE ────────────────
  await browser.close();
  console.log(`\n✅ All screenshots saved to ${RESULTS_DIR}`);
}

main().catch(e => { console.error(e); process.exit(1); });
