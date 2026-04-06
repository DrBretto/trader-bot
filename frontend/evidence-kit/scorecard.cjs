/**
 * Scorecard generator — assembles evidence into the 7-dimension format.
 * Portable: takes collected evidence as input, produces formatted output.
 */

/**
 * Generate a formatted scorecard from collected evidence.
 *
 * @param {Object} evidence - Collected evidence from a capture run:
 *   - structure: { checks: {name: {found, count}}, headings: {...}, duplicateIds: {...} }
 *   - geometry: { measurements: [...], inconsistencies: [...] }
 *   - visual: { screenshots: [...], cls: number|null }
 *   - interaction: { flows: [{name, passed}] }  (optional — may be empty for static pages)
 *   - accessibility: { violations: [...], passCount, violationCount }
 *   - performance: { lcp: number|null, tbt: number|null, bundleSize: string|null }
 *   - dataTruth: { comparisons: [{name, match}] }
 * @param {string} pageName - Name of the page being scored
 * @param {string} date - Date string
 *
 * @returns {string} Formatted scorecard text
 */
function generateScorecard(evidence, pageName, date) {
  const lines = [];
  lines.push(`FRONTEND QUALITY SCORECARD — ${pageName} — ${date}`);
  lines.push('');

  // Dimension 1: Structural Correctness
  const structChecks = evidence.structure?.checks || {};
  const structTotal = Object.keys(structChecks).length;
  const structPassing = Object.values(structChecks).filter(c => c.found).length;
  const headingValid = evidence.structure?.headings?.valid ?? true;
  const dupsValid = evidence.structure?.duplicateIds?.valid ?? true;
  const structScore = structPassing + (headingValid ? 1 : 0) + (dupsValid ? 1 : 0);
  const structMax = structTotal + 2;
  lines.push(`Structural Correctness    ${structScore}/${structMax} checks passing`);

  // Dimension 2: Layout Geometry
  const geoMeasurements = evidence.geometry?.measurements?.length || 0;
  const geoInconsistencies = evidence.geometry?.inconsistencies?.length || 0;
  lines.push(`Layout Geometry           ${geoInconsistencies} inconsistencies in ${geoMeasurements} measured relationships`);

  // Dimension 3: Visual Stability
  const cls = evidence.visual?.cls;
  const viewports = evidence.visual?.screenshots?.length || 0;
  const clsStr = cls != null ? cls.toFixed(3) : 'N/A';
  lines.push(`Visual Stability          CLS: ${clsStr} | ${viewports} viewport(s) captured`);

  // Dimension 4: Interaction Success
  const flows = evidence.interaction?.flows || [];
  const flowsPassed = flows.filter(f => f.passed).length;
  if (flows.length > 0) {
    lines.push(`Interaction Success       ${flowsPassed}/${flows.length} flows completed`);
  } else {
    lines.push(`Interaction Success       (no interaction flows tested — static page)`);
  }

  // Dimension 5: Accessibility
  const a11y = evidence.accessibility || {};
  const violations = a11y.violations || [];
  const critical = violations.filter(v => v.impact === 'critical').length;
  const serious = violations.filter(v => v.impact === 'serious').length;
  const moderate = violations.filter(v => v.impact === 'moderate').length;
  const minor = violations.filter(v => v.impact === 'minor').length;
  lines.push(`Accessibility (automated) ${critical} critical, ${serious} serious, ${moderate} moderate, ${minor} minor violations`);
  lines.push(`                          (covers ~30% of WCAG 2.1 AA)`);

  // Dimension 6: Performance
  const perf = evidence.performance || {};
  const lcpStr = perf.lcp != null ? `${(perf.lcp / 1000).toFixed(1)}s` : 'N/A';
  const tbtStr = perf.tbt != null ? `${Math.round(perf.tbt)}ms` : 'N/A';
  const bundleStr = perf.bundleSize || 'N/A';
  lines.push(`Performance               LCP: ${lcpStr} | TBT: ${tbtStr} | Bundle: ${bundleStr}`);

  // Dimension 7: Data Truth
  const dtComparisons = evidence.dataTruth?.comparisons || [];
  const dtMatching = dtComparisons.filter(c => c.match).length;
  const dtTotal = dtComparisons.length;
  lines.push(`Data Truth                ${dtMatching}/${dtTotal} displayed values match source`);

  // Human review section
  lines.push('');
  lines.push('HUMAN REVIEW NEEDED:');
  lines.push('- Visual coherence and aesthetic quality');
  lines.push('- Information hierarchy and emphasis');
  lines.push('- Accessibility: focus order logic, alt text quality, motion sensitivity');
  lines.push('- Brand alignment and emotional tone');

  // Issues detail
  if (violations.length > 0 || geoInconsistencies > 0 || dtTotal - dtMatching > 0) {
    lines.push('');
    lines.push('ISSUES DETAIL:');

    if (geoInconsistencies > 0) {
      evidence.geometry.inconsistencies.forEach(inc => {
        lines.push(`  [geometry] ${inc}`);
      });
    }

    violations.forEach(v => {
      lines.push(`  [a11y/${v.impact}] ${v.id}: ${v.description} (${v.nodes} node(s))`);
    });

    dtComparisons.filter(c => !c.match).forEach(c => {
      lines.push(`  [data-truth] ${c.name}: displayed="${c.displayed}" source=${c.source} — ${c.note}`);
    });
  }

  return lines.join('\n');
}

module.exports = { generateScorecard };
