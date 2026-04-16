/**
 * E2E test — Visit all UI pages with Walmart M5 weekly data and capture screenshots.
 *
 * Prerequisites:
 *   - Backend running on :8000
 *   - Frontend running on :3000
 *   - Weekly M5 demo CSV at ../data/demo/m5_weekly.csv
 *   - npx playwright install chromium
 *
 * Run: node test-m5-weekly-e2e.js
 */
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const BASE = 'http://localhost:3000';
const API = 'http://localhost:8000';
const RESULTS_DIR = path.join(__dirname, 'test-results', 'm5-weekly');
const LOB = 'walmart_m5_weekly';
const CSV_PATH = path.resolve(__dirname, '..', 'data', 'demo', 'm5_weekly.csv');
const CONFIG_PATH = path.resolve(__dirname, '..', 'configs', 'm5_weekly_config.yaml');

function ensureDir(dir) {
  fs.mkdirSync(dir, { recursive: true });
}

async function screenshot(page, name) {
  await page.screenshot({ path: path.join(RESULTS_DIR, `${name}.png`), fullPage: true });
  console.log(`  📸 ${name}.png`);
}

async function main() {
  ensureDir(RESULTS_DIR);

  // Verify files exist
  if (!fs.existsSync(CSV_PATH)) {
    console.error(`❌ Weekly CSV not found: ${CSV_PATH}`);
    console.error('   Run: python scripts/prepare_m5_weekly.py');
    process.exit(1);
  }

  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
  const page = await context.newPage();
  let errors = 0;

  // ──────────────── 1. DATA ONBOARDING ────────────────
  console.log('\n1. Data Onboarding — Upload M5 Weekly');
  await page.goto(`${BASE}/data-onboarding`);
  await page.waitForLoadState('networkidle');

  const lobInput = page.locator('input[type="text"]').first();
  await lobInput.fill(LOB);

  const fileInput = page.locator('input[type="file"]').first();
  await fileInput.setInputFiles(CSV_PATH);
  console.log('  Uploaded m5_weekly.csv');

  try {
    await page.waitForSelector('text=/Schema Detection|Forecastability|series/i', { timeout: 60000 });
    await page.waitForTimeout(2000);
    await screenshot(page, '01-onboarding-analyzed');

    await page.evaluate(() => window.scrollBy(0, 600));
    await page.waitForTimeout(500);
    await screenshot(page, '02-onboarding-forecastability');

    await page.evaluate(() => window.scrollBy(0, 600));
    await page.waitForTimeout(500);
    await screenshot(page, '03-onboarding-config');
  } catch (e) {
    console.log('  ⚠️ Analysis timeout:', e.message);
    errors++;
  }

  // ──────────────── 1b. RUN BACKTEST FROM UI ────────────────
  console.log('\n1b. Pipeline — Run Backtest');
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(1000);

  const backtestBtn = page.getByRole('button', { name: /run backtest/i });
  if (await backtestBtn.isVisible()) {
    const isDisabled = await backtestBtn.isDisabled();
    if (!isDisabled) {
      await backtestBtn.click();
      console.log('  Running backtest (may take a few minutes)...');
      try {
        await page.waitForSelector('text=/completed|champion|Backtest Result/i', { timeout: 300000 });
        await page.waitForTimeout(2000);
        await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
        await screenshot(page, '04-pipeline-backtest-done');
      } catch (e) {
        console.log('  ⚠️ Backtest timeout');
        await screenshot(page, '04-pipeline-backtest-timeout');
        errors++;
      }
    } else {
      console.log('  ⚠️ Backtest button disabled');
      await screenshot(page, '04-pipeline-btn-disabled');
    }
  }

  // ──────────────── 2. BACKTEST RESULTS ────────────────
  console.log('\n2. Backtest Results');
  await page.goto(`${BASE}/backtest`);
  await page.waitForLoadState('networkidle');

  const btLob = page.locator('input[type="text"]').first();
  await btLob.fill(LOB);
  const loadBtn = page.getByRole('button', { name: 'Load', exact: true });
  if (await loadBtn.isVisible()) {
    await loadBtn.click();
    try {
      await page.waitForSelector('text=/Model Leaderboard/i', { timeout: 10000 });
      await page.waitForTimeout(3000);
    } catch (e) {
      console.log('  ⚠️ Leaderboard not detected');
    }
  }
  await screenshot(page, '05-backtest-loaded');
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(500);
  await screenshot(page, '06-backtest-bottom');

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
      await page.waitForSelector('text=/Series Count|Forecast Origin|forecast/i', { timeout: 15000 });
      await page.waitForTimeout(3000);
    } catch (e) {
      console.log('  ⚠️ Forecast data not detected');
    }
  }
  await screenshot(page, '07-forecast-loaded');
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(500);
  await screenshot(page, '08-forecast-bottom');

  // ──────────────── 4. SERIES EXPLORER ────────────────
  console.log('\n4. Series Explorer');
  await page.goto(`${BASE}/series-explorer`);
  await page.waitForLoadState('networkidle');

  const seLob = page.locator('input[type="text"]').first();
  await seLob.fill(LOB);
  const loadSe = page.getByRole('button', { name: 'Load', exact: true });
  if (await loadSe.isVisible()) {
    await loadSe.click();
    try {
      await page.waitForSelector('text=/Demand Classification|SBC|series/i', { timeout: 15000 });
      await page.waitForTimeout(2000);
    } catch (e) {
      console.log('  ⚠️ Series explorer data not loaded');
    }
  }
  await screenshot(page, '09-series-explorer');

  // ──────────────── 5. HIERARCHY ────────────────
  console.log('\n5. Hierarchy Manager');
  await page.goto(`${BASE}/hierarchy`);
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(1500);
  await screenshot(page, '10-hierarchy');

  // ──────────────── 6. SKU TRANSITIONS ────────────────
  console.log('\n6. SKU Transitions');
  await page.goto(`${BASE}/sku-transitions`);
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(1500);
  await screenshot(page, '11-sku-transitions');

  // ──────────────── 7. PLATFORM HEALTH ────────────────
  console.log('\n7. Platform Health');
  await page.goto(`${BASE}/health`);
  await page.waitForLoadState('networkidle');
  await page.waitForTimeout(1500);
  await screenshot(page, '12-health');

  // ──────────────── 8. S&OP MEETING ────────────────
  console.log('\n8. S&OP Meeting');
  await page.goto(`${BASE}/sop-meeting`);
  await page.waitForLoadState('networkidle');

  // Enter LOB for governance/export panels
  const sopLob = page.locator('input[type="text"]').first();
  if (sopLob) {
    try {
      await sopLob.fill(LOB);
      const loadSop = page.getByRole('button', { name: 'Load', exact: true });
      if (await loadSop.isVisible()) await loadSop.click();
    } catch (e) { /* some pages may not have LOB input */ }
  }
  await page.waitForTimeout(2000);
  await screenshot(page, '13-sop-meeting');

  // Test BI Export buttons
  const exportBtns = page.locator('button:has-text("Export")');
  const exportCount = await exportBtns.count();
  console.log(`  Found ${exportCount} export buttons`);
  if (exportCount > 0) {
    await exportBtns.first().click();
    await page.waitForTimeout(2000);
    await screenshot(page, '14-sop-export-result');
  }

  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(500);
  await screenshot(page, '15-sop-bottom');

  // ──────────────── SUMMARY ────────────────
  await browser.close();
  console.log(`\n${'='.repeat(50)}`);
  console.log(`M5 Weekly E2E — ${errors === 0 ? '✅ PASS' : `⚠️ ${errors} issue(s)`}`);
  console.log(`Screenshots: ${RESULTS_DIR}`);
  console.log(`${'='.repeat(50)}\n`);
  process.exit(errors > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
