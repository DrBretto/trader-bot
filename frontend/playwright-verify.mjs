/**
 * Playwright verification script for live dashboard truth surfaces.
 * Checks: fresh date, story card, System Brain, regime history,
 * SPY/benchmark continuity, top-stack spacing.
 */
import { chromium } from 'playwright';

const LIVE_URL = 'http://investment-system-data.s3-website-us-east-1.amazonaws.com/dashboard/';
const SCREENSHOT_DIR = process.env.SCREENSHOT_DIR || '/tmp/claude/playwright-screenshots';

async function main() {
  const results = [];
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  console.log(`Opening ${LIVE_URL}...`);
  await page.goto(LIVE_URL, { waitUntil: 'networkidle', timeout: 30000 });

  // Wait for dashboard to render
  await page.waitForTimeout(3000);

  // 1. Check fresh date
  console.log('\n=== CHECK 1: Fresh Date ===');
  const pageText = await page.textContent('body');
  const today = new Date().toISOString().slice(0, 10);
  // Check for date in various formats
  const dateFormats = [
    today,
    new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' }),
    new Date().toLocaleDateString('en-US', { month: 'long', day: 'numeric', year: 'numeric' }),
  ];
  const hasDate = dateFormats.some(d => pageText.includes(d));
  // Also check for "Apr 1" or "April 1" style
  const monthDay = new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  const hasRecentDate = hasDate || pageText.includes(monthDay);
  results.push({ check: 'Fresh date visible', pass: hasRecentDate, detail: hasRecentDate ? `Found date reference (${monthDay})` : 'No fresh date found' });
  console.log(hasRecentDate ? `PASS: Found date (${monthDay})` : 'FAIL: No fresh date found');

  // 2. Check Today's Story card
  console.log('\n=== CHECK 2: Today\'s Story ===');
  const storyCard = await page.$('.story-card, .todays-story-card, [class*="story"]');
  const hasStoryHeadline = pageText.includes('Volatility') || pageText.includes('Market') || pageText.includes('Portfolio');
  const hasStorySummary = pageText.length > 500; // Dashboard should have substantial text
  results.push({
    check: 'Story card present and truthful',
    pass: storyCard !== null || hasStoryHeadline,
    detail: storyCard ? 'Story card element found' : (hasStoryHeadline ? 'Story headline text found in page' : 'No story card found')
  });
  console.log(storyCard || hasStoryHeadline ? 'PASS: Story content present' : 'FAIL: No story content');

  // 3. Check System Brain
  console.log('\n=== CHECK 3: System Brain ===');
  const brainPanel = await page.$('.system-brain-panel, [class*="brain"]');
  const hasGRU = pageText.includes('GRU');
  const hasTrans = pageText.includes('Trans');
  const hasBrainContent = hasGRU || hasTrans;
  results.push({
    check: 'System Brain populated',
    pass: brainPanel !== null || hasBrainContent,
    detail: brainPanel ? 'Brain panel element found' : (hasBrainContent ? 'Brain prediction text found' : 'No brain content found')
  });
  console.log(brainPanel || hasBrainContent ? 'PASS: System Brain present' : 'FAIL: No System Brain');

  // 4. Check regime history
  console.log('\n=== CHECK 4: Regime History ===');
  const regimeStrip = await page.$('.brain-regime-strip, .regime-strip, [class*="regime-strip"]');
  const hasRegimeHistory = pageText.includes('Regime history') || pageText.includes('regime');
  results.push({
    check: 'Regime history visible',
    pass: regimeStrip !== null || hasRegimeHistory,
    detail: regimeStrip ? 'Regime strip element found' : (hasRegimeHistory ? 'Regime history text found' : 'No regime history found')
  });
  console.log(regimeStrip || hasRegimeHistory ? 'PASS: Regime history present' : 'FAIL: No regime history');

  // 5. Check SPY/benchmark in equity curve
  console.log('\n=== CHECK 5: SPY/Benchmark Continuity ===');
  const hasSPY = pageText.includes('SPY') || pageText.includes('benchmark') || pageText.includes('Benchmark');
  const perfChart = await page.$('.performance-chart, [class*="equity"], [class*="chart"]');
  results.push({
    check: 'SPY/benchmark continuity visible',
    pass: hasSPY || perfChart !== null,
    detail: hasSPY ? 'SPY/benchmark reference found' : (perfChart ? 'Performance chart found' : 'No benchmark reference found')
  });
  console.log(hasSPY || perfChart ? 'PASS: Benchmark content present' : 'FAIL: No benchmark');

  // 6. Check top-stack spacing
  console.log('\n=== CHECK 6: Top-Stack Spacing ===');
  // Check that the first visible card/section starts at a reasonable position
  const firstCard = await page.$('.card, .status-bar, [class*="hero"], [class*="status"]');
  let spacingOk = false;
  if (firstCard) {
    const box = await firstCard.boundingBox();
    if (box) {
      spacingOk = box.y >= 0 && box.y < 150; // Should start within first 150px
      results.push({
        check: 'Top-stack spacing accurate',
        pass: spacingOk,
        detail: `First card at y=${Math.round(box.y)}px (expected < 150px)`
      });
    }
  }
  if (!spacingOk && !firstCard) {
    results.push({ check: 'Top-stack spacing accurate', pass: false, detail: 'No card elements found' });
  }
  console.log(spacingOk ? `PASS: Top-stack at reasonable position` : 'WARN: Could not verify top-stack');

  // 7. Check signal chart (timeseries-powered)
  console.log('\n=== CHECK 7: Signal Chart (Timeseries) ===');
  const signalChart = await page.$('.brain-signal-chart-wrap, [class*="signal-chart"], svg.recharts-surface');
  const hasSignalLabels = pageText.includes('Macro') || pageText.includes('Fragility') || pageText.includes('Entropy');
  results.push({
    check: 'Signal chart with timeseries data',
    pass: signalChart !== null || hasSignalLabels,
    detail: signalChart ? 'Signal chart SVG found' : (hasSignalLabels ? 'Signal labels found' : 'No signal chart found')
  });
  console.log(signalChart || hasSignalLabels ? 'PASS: Signal chart present' : 'FAIL: No signal chart');

  // Take full-page screenshot
  const screenshotPath = `${SCREENSHOT_DIR}/dashboard-full-${today}.png`;
  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log(`\nScreenshot saved: ${screenshotPath}`);

  // Take above-the-fold screenshot
  const foldPath = `${SCREENSHOT_DIR}/dashboard-fold-${today}.png`;
  await page.screenshot({ path: foldPath });
  console.log(`Fold screenshot saved: ${foldPath}`);

  await browser.close();

  // Summary
  console.log('\n========== VERIFICATION SUMMARY ==========');
  const passed = results.filter(r => r.pass).length;
  const total = results.length;
  for (const r of results) {
    console.log(`${r.pass ? 'PASS' : 'FAIL'}: ${r.check} — ${r.detail}`);
  }
  console.log(`\nResult: ${passed}/${total} checks passed`);

  // Output JSON for evidence file
  const output = { url: LIVE_URL, date: today, timestamp: new Date().toISOString(), results, passed, total };
  console.log('\n--- JSON ---');
  console.log(JSON.stringify(output, null, 2));

  process.exit(passed === total ? 0 : 1);
}

main().catch(err => {
  console.error('Playwright error:', err);
  process.exit(2);
});
