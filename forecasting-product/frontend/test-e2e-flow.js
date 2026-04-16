e
const { chromium } = require('playwright');
const path = require('path');

const M5_CSV = path.resolve(__dirname, '../tests/integration/fixtures/m5_daily_sample.csv');

(async () => {
  const browser = await chromium.launch({ headless: false });
  const page = await browser.newPage();

  // 1. Homepage
  console.log('1. Opening homepage...');
  await page.goto('http://localhost:3000');
  await page.waitForLoadState('networkidle');
  console.log('   Title:', await page.title());

  // 2. Data Onboarding
  console.log('2. Navigating to Data Onboarding...');
  await page.goto('http://localhost:3000/data-onboarding');
  await page.waitForLoadState('networkidle');
  await page.screenshot({ path: 'test-results/01-onboarding-empty.png', fullPage: true });

  // 3. Fill LOB name
  console.log('3. Filling LOB name...');
  const lobInput = page.locator('input').first();
  const allInputs = await page.locator('input').all();
  console.log('   Found', allInputs.length, 'inputs');
  for (let i = 0; i < allInputs.length; i++) {
    const ph = await allInputs[i].getAttribute('placeholder');
    const type = await allInputs[i].getAttribute('type');
    console.log(`   Input ${i}: type=${type}, placeholder=${ph}`);
  }

  // Find the LOB text input (not file, not checkbox)
  const textInputs = page.locator('input[type="text"], input:not([type])');
  const textCount = await textInputs.count();
  console.log('   Text inputs found:', textCount);
  if (textCount > 0) {
    await textInputs.first().fill('walmart_m5_daily');
    console.log('   LOB name entered');
  }

  // 4. Upload file
  console.log('4. Uploading CSV...');
  const fileInput = page.locator('input[type="file"]');
  const fileCount = await fileInput.count();
  console.log('   File inputs found:', fileCount);
  if (fileCount > 0) {
    await fileInput.first().setInputFiles(M5_CSV);
    console.log('   File selected, waiting for analyze...');

    try {
      const response = await page.waitForResponse(
        resp => resp.url().includes('/analyze'),
        { timeout: 120000 }
      );
      console.log('   Analyze status:', response.status());
      await page.waitForTimeout(3000);
    } catch (e) {
      console.log('   WARNING: No /analyze response within 120s:', e.message);
    }
  }

  await page.screenshot({ path: 'test-results/02-onboarding-analyzed.png', fullPage: true });

  // 5. Check page state
  const headings = await page.locator('h2, h3').allTextContents();
  console.log('5. Sections visible:', headings);

  const errors = await page.locator('[class*="error" i], [class*="Error"], [role="alert"]').allTextContents();
  if (errors.length > 0) {
    console.log('   ERRORS:', errors.slice(0, 5));
  } else {
    console.log('   No visible errors');
  }

  // 6. Look for PipelineExecutionPanel (backtest/forecast trigger)
  console.log('6. Looking for pipeline execution...');
  const buttons = await page.locator('button').allTextContents();
  console.log('   Buttons:', buttons.filter(b => b.trim()));

  // 7. Navigate to backtest page
  console.log('7. Navigating to Backtest page...');
  await page.goto('http://localhost:3000/backtest');
  await page.waitForLoadState('networkidle');
  await page.screenshot({ path: 'test-results/03-backtest-empty.png', fullPage: true });

  // Try entering LOB and loading
  const btInputs = page.locator('input[type="text"], input:not([type])');
  if (await btInputs.count() > 0) {
    await btInputs.first().fill('walmart_m5_daily');
    // Look for a Load button
    const loadBtn = page.locator('button:has-text("Load"), button:has-text("load"), button:has-text("Search")');
    if (await loadBtn.count() > 0) {
      await loadBtn.first().click();
      console.log('   Clicked Load button');
      try {
        await page.waitForResponse(
          resp => resp.url().includes('/leaderboard'),
          { timeout: 30000 }
        );
        console.log('   Leaderboard API responded');
        await page.waitForTimeout(2000);
      } catch (e) {
        console.log('   No leaderboard response:', e.message);
      }
    }
  }
  await page.screenshot({ path: 'test-results/04-backtest-loaded.png', fullPage: true });

  // 8. Navigate to forecast page
  console.log('8. Navigating to Forecast page...');
  await page.goto('http://localhost:3000/forecast');
  await page.waitForLoadState('networkidle');
  await page.screenshot({ path: 'test-results/05-forecast-empty.png', fullPage: true });

  const fcHeadings = await page.locator('h2, h3').allTextContents();
  console.log('   Forecast sections:', fcHeadings);

  // 9. Check hierarchy page
  console.log('9. Navigating to Hierarchy page...');
  await page.goto('http://localhost:3000/hierarchy');
  await page.waitForLoadState('networkidle');
  await page.screenshot({ path: 'test-results/06-hierarchy.png', fullPage: true });

  // 10. Check remaining pages exist
  for (const pg of ['/series-explorer', '/sku-transitions', '/health', '/sop']) {
    await page.goto('http://localhost:3000' + pg);
    await page.waitForLoadState('networkidle');
    const name = pg.replace('/', '');
    await page.screenshot({ path: `test-results/07-${name}.png`, fullPage: true });
    const h = await page.locator('h1, h2').first().textContent().catch(() => 'N/A');
    console.log(`   ${pg}: "${h}"`);
  }

  await browser.close();
  console.log('\nDone! Screenshots saved to test-results/');
})();
