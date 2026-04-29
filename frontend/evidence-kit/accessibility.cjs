/**
 * Accessibility evidence collection — inject and run axe-core.
 * Portable: no project-specific configuration.
 */

/**
 * Inject axe-core from CDN and run accessibility audit.
 * Call via page.evaluate() — returns axe results object.
 *
 * Returns { violations: [...], passes: number, incomplete: number, inapplicable: number }
 */
async function runAxeAudit() {
  // Inject axe-core if not already present
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
    runOnly: {
      type: 'tag',
      values: ['wcag2a', 'wcag2aa', 'best-practice'],
    },
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
    inapplicableCount: results.inapplicable.length,
    violationCount: results.violations.length,
  };
}

module.exports = { runAxeAudit };
