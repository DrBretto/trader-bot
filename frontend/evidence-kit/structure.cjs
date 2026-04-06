/**
 * Structure evidence collection — DOM structure and content validation.
 * Portable: selectors and expectations are passed in, not hardcoded.
 */

/**
 * Check that expected elements exist on the page.
 * Pass an object of { name: selector } pairs.
 * Returns { name: { found: boolean, count: number, tagName: string|null } }
 */
function checkElementsExist(selectorMap) {
  return Object.fromEntries(
    Object.entries(selectorMap).map(([name, selector]) => {
      const els = document.querySelectorAll(selector);
      const first = els[0];
      return [name, {
        found: els.length > 0,
        count: els.length,
        tagName: first ? first.tagName.toLowerCase() : null,
      }];
    })
  );
}

/**
 * Check heading hierarchy — h1 before h2 before h3 etc.
 * Returns { valid: boolean, headings: [{level, text}], issues: string[] }
 */
function checkHeadingHierarchy() {
  const headings = [...document.querySelectorAll('h1, h2, h3, h4, h5, h6')].map(h => ({
    level: parseInt(h.tagName[1]),
    text: h.textContent.trim().slice(0, 60),
  }));
  const issues = [];
  for (let i = 1; i < headings.length; i++) {
    if (headings[i].level > headings[i - 1].level + 1) {
      issues.push(`Heading jump: h${headings[i - 1].level} to h${headings[i].level} ("${headings[i].text}")`);
    }
  }
  if (headings.length > 0 && headings[0].level !== 1) {
    issues.push(`First heading is h${headings[0].level}, expected h1`);
  }
  return { valid: issues.length === 0, headings, issues };
}

/**
 * Check for duplicate IDs on the page.
 */
function checkDuplicateIds() {
  const allIds = [...document.querySelectorAll('[id]')].map(el => el.id);
  const counts = {};
  allIds.forEach(id => { counts[id] = (counts[id] || 0) + 1; });
  const duplicates = Object.entries(counts).filter(([, c]) => c > 1).map(([id, c]) => ({ id, count: c }));
  return { valid: duplicates.length === 0, duplicates };
}

/**
 * Extract all visible text content from a set of selectors.
 * Returns { name: textContent }
 */
function extractTextContent(selectorMap) {
  return Object.fromEntries(
    Object.entries(selectorMap).map(([name, selector]) => {
      const el = document.querySelector(selector);
      return [name, el ? el.textContent.trim() : null];
    })
  );
}

module.exports = { checkElementsExist, checkHeadingHierarchy, checkDuplicateIds, extractTextContent };
