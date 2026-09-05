/**
 * The `dynamic_prompts` switch on a text box, honoured where the value is serialised.
 *
 * ComfyUI resolves a `{red|blue}` alternation on the way out of the canvas, before the
 * backend sees anything. Switched off, the text is sent exactly as typed.
 */

import { app } from "../../scripts/app.js";

const EXT_NAME = "WASNodeSuite.DynamicPrompts";

// Node id -> the text widget the switch beside it governs.
const GOVERNED = {
  "Text Multiline": { text: "text", toggle: "dynamic_prompts" },
};

/**
 * Read a widget's value off a node.
 *
 * @param {object} node - The node to read.
 * @param {string} name - The widget's name.
 * @returns {*} Its value, or undefined where there is no such widget.
 */
function widgetValue(node, name) {
  return (node.widgets ?? []).find((candidate) => candidate.name === name)?.value;
}

app.registerExtension({
  name: EXT_NAME,

  nodeCreated(node) {
    const governed = GOVERNED[node?.comfyClass ?? node?.type];
    if (!governed) return;
    const widget = (node.widgets ?? []).find((candidate) => candidate.name === governed.text);
    if (!widget || widget.__was_dynamic_prompts) return;
    widget.__was_dynamic_prompts = true;

    // Core's own extension has already wrapped this at `nodeCreated`, so the wrap below
    // sits outside it and can hand back the raw value instead of calling it.
    const resolved = widget.serializeValue?.bind(widget);
    widget.serializeValue = (workflowNode, widgetIndex) => {
      try {
        if (widgetValue(node, governed.toggle) === false) {
          if (workflowNode?.widgets_values) {
            workflowNode.widgets_values[widgetIndex] = widget.value;
          }
          return widget.value;
        }
      } catch (error) {
        console.error(`[${EXT_NAME}] Failed to read ${governed.toggle}:`, error);
      }
      return resolved ? resolved(workflowNode, widgetIndex) : widget.value;
    };
  },
});
