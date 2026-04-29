/**
 * Geometry evidence collection — bounding boxes and computed styles.
 * Portable: selectors are passed in, not hardcoded.
 */

/**
 * Get bounding boxes for all elements matching a selector.
 * Inject via page.evaluate().
 */
function measureElements(selector) {
  return [...document.querySelectorAll(selector)].map(el => ({
    tag: el.tagName.toLowerCase(),
    id: el.id || null,
    className: typeof el.className === 'string' ? el.className.split(' ').filter(Boolean).slice(0, 3).join(' ') : '',
    rect: el.getBoundingClientRect().toJSON(),
    visible: el.offsetParent !== null || el.tagName === 'BODY',
  }));
}

/**
 * Get computed styles for a specific element.
 * Pass the CSS properties you want as an array.
 */
function getComputedStyles(selector, properties) {
  const el = document.querySelector(selector);
  if (!el) return null;
  const cs = getComputedStyle(el);
  return Object.fromEntries(properties.map(p => [p, cs.getPropertyValue(p)]));
}

/**
 * Measure the gap between two elements (vertical or horizontal).
 */
function measureGap(selectorA, selectorB) {
  const a = document.querySelector(selectorA);
  const b = document.querySelector(selectorB);
  if (!a || !b) return null;
  const ra = a.getBoundingClientRect();
  const rb = b.getBoundingClientRect();
  return {
    horizontal: rb.left - ra.right,
    vertical: rb.top - ra.bottom,
    aRect: ra.toJSON(),
    bRect: rb.toJSON(),
  };
}

/**
 * Check if two elements have matching top edges (column alignment).
 */
function checkTopAlignment(selectorA, selectorB, tolerancePx) {
  const tol = tolerancePx || 2;
  const a = document.querySelector(selectorA);
  const b = document.querySelector(selectorB);
  if (!a || !b) return { aligned: false, error: 'element not found' };
  const ra = a.getBoundingClientRect();
  const rb = b.getBoundingClientRect();
  return {
    aligned: Math.abs(ra.top - rb.top) <= tol,
    deltaTop: rb.top - ra.top,
    aTop: ra.top,
    bTop: rb.top,
  };
}

/**
 * Check if two elements have matching bottom edges (column height match).
 */
function checkBottomAlignment(selectorA, selectorB, tolerancePx) {
  const tol = tolerancePx || 2;
  const a = document.querySelector(selectorA);
  const b = document.querySelector(selectorB);
  if (!a || !b) return { aligned: false, error: 'element not found' };
  const ra = a.getBoundingClientRect();
  const rb = b.getBoundingClientRect();
  return {
    aligned: Math.abs(ra.bottom - rb.bottom) <= tol,
    deltaBottom: rb.bottom - ra.bottom,
    aBottom: ra.bottom,
    bBottom: rb.bottom,
    aHeight: ra.height,
    bHeight: rb.height,
  };
}

module.exports = { measureElements, getComputedStyles, measureGap, checkTopAlignment, checkBottomAlignment };
