/**
 * Folding a widget away on either node renderer.
 *
 * A canvas node reads `hidden` and a DOM node reads `options.hidden`. Both are set together.
 * Folding is presentation only: `widgets_values` carries every widget whether it is drawn or
 * not.
 */

/**
 * Draw one widget or fold it away.
 *
 * @param {object} widget - The widget to set.
 * @param {boolean} hidden - True to fold it away.
 * @returns {boolean} True where this changed what is drawn.
 */
export function setWidgetHidden(widget, hidden) {
  if (!widget) return false;
  const wanted = Boolean(hidden);
  const options = widget.options ?? (widget.options = {});
  if (Boolean(widget.hidden) === wanted && Boolean(options.hidden) === wanted) return false;
  widget.hidden = wanted;
  options.hidden = wanted;
  return true;
}

/**
 * Fold a widget away for the life of the node, as a widget carrying state rather than a control.
 *
 * @param {object} widget - The widget to fold away.
 * @returns {void}
 */
export function keepWidgetHidden(widget) {
  if (!widget) return;
  widget.type = "hidden";
  widget.computeSize = () => [0, -4];
  widget.draw = () => {};
  setWidgetHidden(widget, true);
}
