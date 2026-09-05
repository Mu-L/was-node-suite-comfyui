/**
 * Where a pointer is, in the pixels an interface draws in.
 *
 * A canvas inside a node is scaled by the graph's zoom, so a position in client pixels is not a
 * position in the interface's own pixels.
 */

/**
 * Read a pointer position in an element's own pixels.
 *
 * @param {HTMLElement} element - The element the interface is drawn on.
 * @param {PointerEvent|MouseEvent} event - Event to read.
 * @returns {{x: number, y: number}} Position inside the element.
 */
export function elementPoint(element, event) {
  const rect = element.getBoundingClientRect();
  const scaleX = rect.width ? element.clientWidth / rect.width : 1;
  const scaleY = rect.height ? element.clientHeight / rect.height : 1;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

// The units a wheel gesture is measured in beyond pixels.
const DELTA_LINE = 1;
const DELTA_PAGE = 2;

// Pixels one line covers.
const LINE_PIXELS = 16;

/**
 * A wheel gesture's distance in pixels.
 *
 * @param {WheelEvent} event - The gesture.
 * @param {HTMLElement} [element] - The element being scrolled, for a gesture measured in pages.
 * @returns {{x: number, y: number}} The distance, in CSS pixels.
 */
export function wheelPixels(event, element) {
  if (event.deltaMode === DELTA_LINE) {
    return { x: event.deltaX * LINE_PIXELS, y: event.deltaY * LINE_PIXELS };
  }
  if (event.deltaMode === DELTA_PAGE) {
    return {
      x: event.deltaX * (element?.clientWidth || 0),
      y: event.deltaY * (element?.clientHeight || 0),
    };
  }
  return { x: event.deltaX, y: event.deltaY };
}

/**
 * Take every wheel gesture over a panel, so the graph's zoom is left to the canvas around it.
 *
 * @param {HTMLElement} element - The panel's own element.
 * @param {(event: WheelEvent) => void} [onWheel] - Called with each gesture, for a panel that
 *   scrolls or steps through something. Left out, the gesture is swallowed.
 * @returns {() => void} Releases the listener.
 */
export function captureWheel(element, onWheel) {
  const handler = (event) => {
    event.preventDefault();
    event.stopPropagation();
    onWheel?.(event);
  };
  element.addEventListener("wheel", handler, { passive: false });
  return () => element.removeEventListener("wheel", handler);
}
