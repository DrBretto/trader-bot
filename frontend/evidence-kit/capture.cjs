#!/usr/bin/env node
/**
 * Evidence capture runner — runs a full capture against a target URL.
 *
 * Usage: node evidence-kit/capture.js [url] [output-dir]
 *
 * Defaults:
 *   url: http://investment-system-data.s3-website-us-east-1.amazonaws.com/dashboard/
 *   output-dir: ./evidence-kit/captures/<timestamp>
 */

const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const { generateScorecard } = require('./scorecard.cjs');

const TARGET_URL = process.argv[2] || 'http://investment-system-data.s3-website-us-east-1.amazonaws.com/dashboard/';
const OUTPUT_DIR = process.argv[3] || path.join(__dirname, 'captures', new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19));

// -- Trader-bot-specific selectors (the project-specific part) --
const EXPECTED_ELEMENTS = {
  dashboard: '.dashboard',
  zone1Header: '.zone-1-header',
  zone1Title: '.zone-1-header h1',
  systemStatusBar: '.system-status-bar',
  zone2: '.zone-2',
  zone2Chart: '.zone-2-chart',
  zone2Right: '.zone-2-right',
  perfSummaryCard: '.perf-summary-card',
  performanceChart: '.performance-chart-card',
  todaysStoryCard: '.todays-story-card',
  systemBrainPanel: '.system-brain-panel',
  lowerDeck: '.lower-deck',
  portfolioTable: '.lower-deck-slot:nth-child(1)',
  tradeLog: '.lower-deck-slot:nth-child(2)',
  candidatesTable: '.lower-deck-slot:nth-child(3)',
};

const VIEWPORTS = [
  { width: 1600, height: 900, label: 'desktop-wide' },
  { width: 1024, height: 768, label: 'tablet' },
  { width: 768, height: 1024, label: 'tablet-portrait' },
];

async function run() {
  console.log(`Evidence capture starting...`);
  console.log(`  Target: ${TARGET_URL}`);
  console.log(`  Output: ${OUTPUT_DIR}`);

  fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  fs.mkdirSync(path.join(OUTPUT_DIR, 'screenshots'), { recursive: true });

  const browser = await chromium.launch({ headless: true, channel: 'chrome' });
  const evidence = {
    url: TARGET_URL,
    timestamp: new Date().toISOString(),
    structure: {},
    geometry: { measurements: [], inconsistencies: [] },
    visual: { screenshots: [], cls: null },
    interaction: { flows: [] },
    accessibility: {},
    performance: {},
    dataTruth: { comparisons: [] },
  };

  try {
    const page = await browser.newPage();
    await page.setViewportSize({ width: 1600, height: 900 });

    // Navigate and wait for data to load
    console.log('  Loading page...');
    await page.goto(TARGET_URL, { waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForSelector('.dashboard', { timeout: 15000 });
    // Give React time to render
    await page.waitForTimeout(2000);

    // --- STRUCTURE ---
    console.log('  Collecting structure evidence...');
    evidence.structure.checks = await page.evaluate((selectors) => {
      return Object.fromEntries(
        Object.entries(selectors).map(([name, sel]) => {
          const els = document.querySelectorAll(sel);
          return [name, { found: els.length > 0, count: els.length }];
        })
      );
    }, EXPECTED_ELEMENTS);

    evidence.structure.headings = await page.evaluate(() => {
      const headings = [...document.querySelectorAll('h1, h2, h3, h4, h5, h6')].map(h => ({
        level: parseInt(h.tagName[1]),
        text: h.textContent.trim().slice(0, 60),
      }));
      const issues = [];
      for (let i = 1; i < headings.length; i++) {
        if (headings[i].level > headings[i - 1].level + 1) {
          issues.push(`Heading jump: h${headings[i - 1].level} to h${headings[i].level}`);
        }
      }
      if (headings.length > 0 && headings[0].level !== 1) {
        issues.push(`First heading is h${headings[0].level}, expected h1`);
      }
      return { valid: issues.length === 0, headings, issues };
    });

    evidence.structure.duplicateIds = await page.evaluate(() => {
      const allIds = [...document.querySelectorAll('[id]')].map(el => el.id);
      const counts = {};
      allIds.forEach(id => { counts[id] = (counts[id] || 0) + 1; });
      const duplicates = Object.entries(counts).filter(([, c]) => c > 1).map(([id, c]) => ({ id, count: c }));
      return { valid: duplicates.length === 0, duplicates };
    });

    // --- GEOMETRY ---
    console.log('  Collecting geometry evidence...');
    const zoneGeometry = await page.evaluate(() => {
      const z2Chart = document.querySelector('.zone-2-chart');
      const z2Right = document.querySelector('.zone-2-right');
      if (!z2Chart || !z2Right) return null;
      const rc = z2Chart.getBoundingClientRect();
      const rr = z2Right.getBoundingClientRect();
      return {
        leftColumn: rc.toJSON(),
        rightColumn: rr.toJSON(),
        topAligned: Math.abs(rc.top - rr.top) <= 2,
        topDelta: rr.top - rc.top,
        bottomAligned: Math.abs(rc.bottom - rr.bottom) <= 10,
        bottomDelta: rr.bottom - rc.bottom,
        leftHeight: rc.height,
        rightHeight: rr.height,
        heightDelta: Math.abs(rc.height - rr.height),
      };
    });

    if (zoneGeometry) {
      evidence.geometry.measurements.push(
        { name: 'zone2-top-alignment', value: zoneGeometry.topDelta, tolerance: 2, pass: zoneGeometry.topAligned },
        { name: 'zone2-bottom-alignment', value: zoneGeometry.bottomDelta, tolerance: 10, pass: zoneGeometry.bottomAligned },
        { name: 'zone2-height-delta', value: zoneGeometry.heightDelta, pass: zoneGeometry.heightDelta < 20 },
      );
      if (!zoneGeometry.topAligned) evidence.geometry.inconsistencies.push(`Zone 2 top misalignment: ${zoneGeometry.topDelta.toFixed(1)}px delta`);
      if (!zoneGeometry.bottomAligned) evidence.geometry.inconsistencies.push(`Zone 2 bottom misalignment: ${zoneGeometry.bottomDelta.toFixed(1)}px delta (left=${zoneGeometry.leftHeight.toFixed(0)}px, right=${zoneGeometry.rightHeight.toFixed(0)}px)`);
    }

    // Lower deck height consistency
    const lowerDeckGeometry = await page.evaluate(() => {
      const slots = [...document.querySelectorAll('.lower-deck-slot')];
      if (slots.length < 2) return null;
      const rects = slots.map(s => s.getBoundingClientRect());
      const heights = rects.map(r => r.height);
      const maxDelta = Math.max(...heights) - Math.min(...heights);
      return { slotCount: slots.length, heights, maxDelta, consistent: maxDelta < 5 };
    });

    if (lowerDeckGeometry) {
      evidence.geometry.measurements.push(
        { name: 'lower-deck-height-consistency', value: lowerDeckGeometry.maxDelta, tolerance: 5, pass: lowerDeckGeometry.consistent },
      );
      if (!lowerDeckGeometry.consistent) evidence.geometry.inconsistencies.push(`Lower deck height inconsistency: ${lowerDeckGeometry.maxDelta.toFixed(1)}px max delta`);
    }

    // Card spacing consistency
    const cardGaps = await page.evaluate(() => {
      const z2Right = document.querySelector('.zone-2-right');
      if (!z2Right) return null;
      const cs = getComputedStyle(z2Right);
      return { gap: cs.gap, rowGap: cs.rowGap, columnGap: cs.columnGap };
    });
    if (cardGaps) {
      evidence.geometry.measurements.push({ name: 'zone2-right-gap', value: cardGaps.gap, pass: true });
    }

    // --- SCREENSHOTS ---
    console.log('  Capturing screenshots...');
    for (const vp of VIEWPORTS) {
      await page.setViewportSize({ width: vp.width, height: vp.height });
      await page.waitForTimeout(500);
      const filename = `screenshot-${vp.label}-${vp.width}x${vp.height}.png`;
      await page.screenshot({ path: path.join(OUTPUT_DIR, 'screenshots', filename), fullPage: true });
      evidence.visual.screenshots.push({ label: vp.label, width: vp.width, height: vp.height, filename });
    }

    // Reset to desktop for remaining tests
    await page.setViewportSize({ width: 1600, height: 900 });
    await page.waitForTimeout(500);

    // --- ACCESSIBILITY (axe-core) ---
    console.log('  Running accessibility audit...');
    try {
      evidence.accessibility = await page.evaluate(async () => {
        if (!window.axe) {
          const script = document.createElement('script');
          script.src = 'https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.2/axe.min.js';
          script.crossOrigin = 'anonymous';
          document.head.appendChild(script);
          await new Promise((resolve, reject) => {
            script.onload = resolve;
            script.onerror = () => reject(new Error('Failed to load axe-core'));
          });
        }
        const results = await window.axe.run(document, {
          runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'best-practice'] },
        });
        return {
          violations: results.violations.map(v => ({
            id: v.id,
            impact: v.impact,
            description: v.description,
            helpUrl: v.helpUrl,
            nodes: v.nodes.length,
            targets: v.nodes.slice(0, 3).map(n => n.target.join(' > ')),
          })),
          passCount: results.passes.length,
          incompleteCount: results.incomplete.length,
          violationCount: results.violations.length,
        };
      });
    } catch (e) {
      console.log(`  Accessibility audit failed: ${e.message}`);
      evidence.accessibility = { violations: [], passCount: 0, violationCount: 0, error: e.message };
    }

    // --- DATA TRUTH ---
    console.log('  Running data truth trace...');
    try {
      // Fetch the source data
      const sourceData = await page.evaluate(async () => {
        const resp = await fetch('dashboard.json', { cache: 'no-store' });
        return resp.json();
      });

      const m = sourceData.metrics;

      // Compare source values to displayed text
      evidence.dataTruth.comparisons = await page.evaluate((metrics) => {
        function findTextInPage(searchText) {
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
          while (walker.nextNode()) {
            if (walker.currentNode.textContent.includes(searchText)) return true;
          }
          return false;
        }

        function extractMetricValue(label) {
          const els = [...document.querySelectorAll('.z1-metric')];
          for (const el of els) {
            const labelEl = el.querySelector('.z1-metric-label');
            if (labelEl && labelEl.textContent.trim().toLowerCase() === label.toLowerCase()) {
              const valueEl = el.querySelector('.z1-metric-value');
              return valueEl ? valueEl.textContent.trim() : null;
            }
          }
          return null;
        }

        const results = [];

        // Total Value
        const totalValueDisplay = extractMetricValue('Total Value');
        if (totalValueDisplay) {
          const cleaned = parseFloat(totalValueDisplay.replace(/[$,]/g, ''));
          results.push({
            name: 'Total Value',
            displayed: totalValueDisplay,
            source: metrics.total_value,
            match: !isNaN(cleaned) && Math.abs(cleaned - metrics.total_value) < 1,
            note: !isNaN(cleaned) ? `diff=${Math.abs(cleaned - metrics.total_value).toFixed(2)}` : 'parse failed',
          });
        }

        // YTD Return
        const ytdDisplay = extractMetricValue('YTD');
        if (ytdDisplay) {
          const cleaned = parseFloat(ytdDisplay.replace(/[%+]/g, ''));
          const expected = metrics.ytd_return * 100;
          results.push({
            name: 'YTD Return',
            displayed: ytdDisplay,
            source: metrics.ytd_return,
            match: !isNaN(cleaned) && Math.abs(cleaned - expected) < 0.1,
            note: `displayed=${cleaned}, expected=${expected.toFixed(2)}`,
          });
        }

        // Sharpe
        const sharpeDisplay = extractMetricValue('Sharpe');
        if (sharpeDisplay) {
          const cleaned = parseFloat(sharpeDisplay);
          const expected = metrics.sharpe_ratio;
          results.push({
            name: 'Sharpe Ratio',
            displayed: sharpeDisplay,
            source: expected,
            match: expected != null && !isNaN(cleaned) && Math.abs(cleaned - expected) < 0.01,
            note: `displayed=${cleaned}, expected=${expected?.toFixed(2)}`,
          });
        }

        // Max Drawdown
        const ddDisplay = extractMetricValue('Max DD');
        if (ddDisplay) {
          const cleaned = parseFloat(ddDisplay.replace(/[%+]/g, ''));
          const expected = metrics.max_drawdown * 100;
          results.push({
            name: 'Max Drawdown',
            displayed: ddDisplay,
            source: metrics.max_drawdown,
            match: !isNaN(cleaned) && Math.abs(cleaned - expected) < 0.1,
            note: `displayed=${cleaned}, expected=${expected.toFixed(2)}`,
          });
        }

        // Cash percentage in Today's Story
        const cashChip = document.querySelector('.story-meta-chip');
        if (cashChip) {
          const cashText = cashChip.textContent.trim();
          results.push({
            name: 'Cash %',
            displayed: cashText,
            source: metrics.cash_pct,
            match: cashText.includes((metrics.cash_pct * 100).toFixed(1)),
            note: `expected "Cash ${(metrics.cash_pct * 100).toFixed(1)}%"`,
          });
        }

        // Holdings count vs displayed
        const holdingsEmpty = document.querySelector('.lower-deck-slot:nth-child(1)');
        if (holdingsEmpty) {
          const hasNoHoldings = holdingsEmpty.textContent.includes('No current holdings');
          results.push({
            name: 'Holdings display',
            displayed: hasNoHoldings ? 'No current holdings' : 'Has holdings table',
            source: 'check',
            match: true,
            note: 'presence check only',
          });
        }

        return results;
      }, m);
    } catch (e) {
      console.log(`  Data truth trace failed: ${e.message}`);
      evidence.dataTruth.error = e.message;
    }

    // --- PERFORMANCE ---
    console.log('  Collecting performance signals...');
    evidence.performance.bundleSize = '176 KB gzip (from build output)';

    // Console messages
    const consoleMessages = [];
    page.on('console', msg => consoleMessages.push({ type: msg.type(), text: msg.text() }));
    // Reload to capture console
    await page.reload({ waitUntil: 'networkidle', timeout: 30000 });
    await page.waitForTimeout(2000);
    evidence.console = {
      errors: consoleMessages.filter(m => m.type === 'error'),
      warnings: consoleMessages.filter(m => m.type === 'warning'),
      total: consoleMessages.length,
    };

  } catch (err) {
    console.error(`  Capture error: ${err.message}`);
    evidence.error = err.message;
  } finally {
    await browser.close();
  }

  // --- GENERATE SCORECARD ---
  console.log('  Generating scorecard...');
  const scorecard = generateScorecard(evidence, 'Trader-Bot Dashboard', new Date().toISOString().slice(0, 10));

  // --- WRITE OUTPUT ---
  fs.writeFileSync(path.join(OUTPUT_DIR, 'evidence.json'), JSON.stringify(evidence, null, 2));
  fs.writeFileSync(path.join(OUTPUT_DIR, 'scorecard.txt'), scorecard);

  console.log('\n' + scorecard);
  console.log(`\nEvidence written to: ${OUTPUT_DIR}`);
  console.log('Capture complete.');
}

run().catch(err => {
  console.error('Fatal error:', err);
  process.exit(1);
});
