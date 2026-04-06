/**
 * Data truth evidence collection — compare source API values to rendered display.
 * Portable: the mapping between source fields and display selectors is passed in.
 */

/**
 * Extract displayed values from DOM selectors and compare to source values.
 *
 * @param {Object} mappings - Array of { name, selector, sourceValue, tolerance?, format? }
 *   - name: human-readable label
 *   - selector: CSS selector to find the display element
 *   - sourceValue: the raw value from the data source
 *   - tolerance: (optional) numeric tolerance for comparison (default 0.01)
 *   - format: (optional) 'currency' | 'percent' | 'number' | 'text'
 *
 * Returns array of { name, displayed, source, match, note }
 */
function compareSourceToDisplay(mappings) {
  return mappings.map(m => {
    const el = document.querySelector(m.selector);
    const displayed = el ? el.textContent.trim() : null;

    if (displayed === null) {
      return { name: m.name, displayed: null, source: m.sourceValue, match: false, note: 'element not found' };
    }

    // Try numeric comparison if source is a number
    if (typeof m.sourceValue === 'number') {
      // Strip currency symbols, commas, percent signs for numeric extraction
      const cleaned = displayed.replace(/[$,%+]/g, '').replace(/,/g, '').trim();
      const displayedNum = parseFloat(cleaned);

      if (!isNaN(displayedNum)) {
        const tol = m.tolerance || 0.01;
        // Handle percent format — source might be decimal (0.02) while display is "2.19%"
        let sourceForComparison = m.sourceValue;
        if (m.format === 'percent') {
          sourceForComparison = m.sourceValue * 100;
        }
        const diff = Math.abs(displayedNum - sourceForComparison);
        const match = diff <= tol;
        return {
          name: m.name,
          displayed,
          source: m.sourceValue,
          sourceForComparison,
          match,
          diff: diff.toFixed(6),
          note: match ? 'numeric match within tolerance' : `numeric mismatch: diff=${diff.toFixed(6)}`,
        };
      }
    }

    // Text comparison
    const sourceStr = String(m.sourceValue);
    const match = displayed.includes(sourceStr) || sourceStr.includes(displayed);
    return {
      name: m.name,
      displayed,
      source: m.sourceValue,
      match,
      note: match ? 'text match' : 'text mismatch',
    };
  });
}

module.exports = { compareSourceToDisplay };
