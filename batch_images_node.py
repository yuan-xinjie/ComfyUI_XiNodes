# -*- coding: utf-8 -*-
"""
Xi Image Batch  (V3 / comfy_api)

A fault-tolerant re-implementation of ComfyUI's built-in BatchImagesNode,
written against the V3 node API and using the very same io.Autogrow input, so
the "+" button behaves exactly like the official node.

Two deliberate differences from the official node:

1. Every autogrow slot is *optional*.
   comfy_api/latest/_io.py::_expand_schema_for_dynamic decides required-vs-
   optional per generated slot like this::

       template_required = _input_type == "required"
       if i < min and template_required:  -> required
       else:                              -> optional

   so marking the template input optional turns every generated slot optional
   and makes "min" irrelevant to validation. That function even has an explicit
   branch for "all inputs are optional and no values are received", which hands
   execute() an empty dict. Result: a slot whose source node is muted or ignored
   is skipped instead of aborting the queue with "Required input is missing".

2. No resizing and no channel padding - this is a pure torch.cat. RGBA stays
   RGBA, sizes stay as they are. A genuine mismatch raises an error that names
   every slot's shape so it is obvious which link is wrong.
"""

import torch
from comfy_api.latest import io

# Same ceiling as the official Batch Images node.
MAX_BATCH_SLOTS = 50
SLOT_PREFIX = "image"


# --------------------------------------------------------------------------- #
# workflow introspection (cosmetic only - must never break execution)
# --------------------------------------------------------------------------- #
_LITEGRAPH_MODES = {2: "muted", 4: "bypassed"}


def _graph_slot_status(extra_pnginfo, unique_id):
    """Map slot name -> why its value never arrived, using the saved workflow.

    The prompt reaching the backend has already lost muted/ignored inputs
    entirely, so the only place that still records *why* is the editor workflow
    embedded in EXTRA_PNGINFO. Returns {} on any problem.
    """
    status = {}
    try:
        workflow = (extra_pnginfo or {}).get("workflow") or {}
        wf_nodes = workflow.get("nodes") or []
        wf_links = workflow.get("links") or []
        if not wf_nodes or not wf_links:
            return status

        my_id = int(unique_id)
        me = next((n for n in wf_nodes if n.get("id") == my_id), None)
        if me is None:
            return status

        nodes_by_id = {n.get("id"): n for n in wf_nodes}
        # link := [link_id, origin_id, origin_slot, target_id, target_slot, type]
        links_by_id = {}
        for link in wf_links:
            try:
                links_by_id[link[0]] = link
            except Exception:
                continue

        for slot in me.get("inputs") or []:
            name = slot.get("name")
            if not name or not str(name).startswith(SLOT_PREFIX):
                continue
            link = links_by_id.get(slot.get("link"))
            if link is None:
                status[name] = "unconnected"
                continue
            source = nodes_by_id.get(link[1])
            if source is None:
                status[name] = "source node #{} is gone".format(link[1])
                continue
            mode = _LITEGRAPH_MODES.get(source.get("mode", 0))
            title = source.get("title") or source.get("type") or "?"
            if mode:
                status[name] = "'{}' #{} is {}".format(title, link[1], mode)
            else:
                status[name] = "'{}' #{} produced no output".format(title, link[1])
    except Exception:
        return {}
    return status


# --------------------------------------------------------------------------- #
# scheduler control (same entry point the native Start Loop node uses)
# --------------------------------------------------------------------------- #
def _is_link(value):
    return (isinstance(value, list) and len(value) == 2
            and isinstance(value[0], str) and isinstance(value[1], int))


def _downstream_closure(dynprompt, my_id):
    """Every node id transitively fed by *my_id* (excluding *my_id* itself)."""
    children = {}
    for node_id in dynprompt.all_node_ids():
        inputs = dynprompt.get_node(node_id).get("inputs", {}) or {}
        for value in inputs.values():
            if _is_link(value):
                children.setdefault(value[0], set()).add(node_id)

    my_id = str(my_id)
    reachable = set()
    stack = list(children.get(my_id, ()))
    while stack:
        node_id = stack.pop()
        if node_id == my_id or node_id in reachable:
            continue
        reachable.add(node_id)
        stack.extend(children.get(node_id, ()))
    reachable.discard(my_id)
    return reachable


def _skip_downstream(execution_list, dynprompt, my_id):
    """Drop this node's whole downstream sub-graph from the current queue.

    comfy_extras/nodes_loop.py:230 uses this exact call for the native Start
    Loop node. Returns the number of nodes removed, or None when the scheduler
    is unavailable.
    """
    try:
        if execution_list is None or dynprompt is None or my_id is None:
            return None
        if getattr(execution_list, "staged_node_id", None) is None:
            return None  # not running inside a real node execution
        pending = getattr(execution_list, "pendingNodes", None)
        if pending is None:
            return None

        targets = _downstream_closure(dynprompt, my_id).intersection(pending.keys())
        targets.discard(str(my_id))
        if not targets:
            return 0
        execution_list.inhibit_nodes(targets)
        return len(targets)
    except Exception as exc:  # a convenience feature must never break a run
        print("[XiNodes] XiImageBatch: could not skip downstream ({})".format(exc))
        return None


def _fallback(mode, width, height, color):
    """Stand-in tensor for 'empty batch' / 'blank image'. float32 / CPU / 3ch."""
    batch = 0 if mode == "empty batch" else 1
    return torch.full((batch, height, width, 3), float(color), dtype=torch.float32), batch


# --------------------------------------------------------------------------- #
# node
# --------------------------------------------------------------------------- #
class XiImageBatch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.Autogrow.TemplatePrefix(
            io.Image.Input("image", optional=True),
            prefix=SLOT_PREFIX,
            min=1,
            max=MAX_BATCH_SLOTS,
        )
        return io.Schema(
            node_id="XiImageBatch",
            display_name="Image Batch (Tolerant)",
            category="XiNodes",
            description=(
                "Stack images into one IMAGE batch, like the official Batch Images node and with the "
                "same '+' input behaviour, but every slot is optional: an unconnected, muted or ignored "
                "slot is skipped instead of raising 'Required input is missing'. Pure torch.cat - no "
                "resizing, no channel padding, RGBA stays RGBA."
            ),
            search_aliases=["batch images", "combine images", "merge images", "stack images", "批量图像"],
            inputs=[
                io.Autogrow.Input(
                    "images",
                    template=template,
                    optional=True,
                    tooltip="Click '+' to add image slots (up to 50). A slot whose source node is muted or ignored is skipped automatically.",
                ),
                io.Combo.Input(
                    "on_empty",
                    options=["error", "no value", "skip downstream", "empty batch", "blank image"],
                    default="error",
                    tooltip="Only used when NO slot produced an image. Pick by what the consumer expects: 'no value' when the downstream image input is OPTIONAL (it then behaves exactly like an unconnected socket); 'skip downstream' when it is REQUIRED - that removes the whole downstream sub-graph from the queue so nothing runs and nothing is saved. 'empty batch' emits (0,H,W,3) - note Save Image crashes on that. 'blank image' emits one solid frame.",
                ),
                io.Int.Input("empty_width", default=512, min=1, max=16384, step=8,
                             advanced=True, tooltip="Width of the stand-in tensor for 'empty batch' / 'blank image'."),
                io.Int.Input("empty_height", default=512, min=1, max=16384, step=8,
                             advanced=True, tooltip="Height of the stand-in tensor for 'empty batch' / 'blank image'."),
                io.Float.Input("empty_color", default=0.0, min=0.0, max=1.0, step=0.01,
                               advanced=True, tooltip="Grey level of the 'blank image' stand-in. 0 = black, 1 = white."),
                io.Boolean.Input(
                    "print_info", default=True,
                    tooltip="Print a console summary and, using the saved workflow, name the exact source node that was muted or bypassed so nothing vanishes silently.",
                ),
            ],
            outputs=[
                io.Image.Output(display_name="IMAGE"),
                io.Int.Output(display_name="COUNT"),
            ],
            hidden=[
                io.Hidden.unique_id,
                io.Hidden.extra_pnginfo,
                io.Hidden.dynprompt,
                io.Hidden.execution_list,
            ],
        )

    @classmethod
    def execute(cls, images, on_empty, empty_width, empty_height, empty_color, print_info):
        slots = dict(images or {})
        status = _graph_slot_status(cls.hidden.extra_pnginfo, cls.hidden.unique_id) if print_info else {}

        valid = []
        problems = []
        for name, image in slots.items():
            if image is None:
                reason = status.get(name)
                if reason and reason != "unconnected":
                    problems.append("{}: {}".format(name, reason))
                continue
            if not isinstance(image, torch.Tensor) or image.ndim != 4 or 0 in image.shape:
                problems.append("{}: not a usable [B, H, W, C] image".format(name))
                continue
            valid.append((name, image))

        if not valid:
            detail = " [" + "; ".join(problems) + "]" if problems else ""
            if on_empty == "no value":
                # For an OPTIONAL downstream input a None is indistinguishable
                # from never being wired up: validate_prompt() only compares the
                # declared RETURN_TYPES, and the consumer falls back to its own
                # default.
                if print_info:
                    print("[XiNodes] XiImageBatch: no usable input{} - emitting no value; "
                          "optional downstream inputs treat it as unconnected.".format(detail))
                return io.NodeOutput(None, 0)
            if on_empty == "skip downstream":
                removed = _skip_downstream(cls.hidden.execution_list, cls.hidden.dynprompt,
                                           cls.hidden.unique_id)
                if print_info:
                    print("[XiNodes] XiImageBatch: no usable input{} - {}".format(
                        detail,
                        "the scheduler does not allow skipping here, emitting an empty batch instead."
                        if removed is None else
                        "removed {} downstream node(s) from the queue, nothing else will run.".format(removed)))
                return io.NodeOutput(*_fallback("empty batch", empty_width, empty_height, empty_color))
            if on_empty == "error":
                raise ValueError(
                    "Xi Image Batch: no usable image input - every connected slot is muted, ignored "
                    "or empty" + detail + ". Connect an active IMAGE source, or change 'on_empty'.")
            return io.NodeOutput(*_fallback(on_empty, empty_width, empty_height, empty_color))

        tensors = [image for _, image in valid]
        try:
            batched = torch.cat(tensors, dim=0)
        except RuntimeError as exc:
            # Pure concatenation on purpose - tell the user exactly what clashed
            # instead of silently resizing or padding behind their back.
            layout = ", ".join("{}: shape={} dtype={} device={}".format(
                name, tuple(image.shape), image.dtype, image.device) for name, image in valid)
            raise ValueError(
                "Xi Image Batch: torch.cat failed because the inputs are not compatible. This node "
                "deliberately does not resize or pad channels - make them match (or insert a Resize "
                "node). Inputs were -> " + layout) from exc

        count = int(batched.shape[0])
        if print_info:
            if problems:
                print("[XiNodes] XiImageBatch: batched {} image(s) from {} input(s); skipped {} "
                      "slot(s) -> {}".format(count, len(valid), len(problems), "; ".join(problems)))
            else:
                print("[XiNodes] XiImageBatch: batched {} image(s) from {} input(s).".format(
                    count, len(valid)))
        return io.NodeOutput(batched, count)


NODES = [XiImageBatch]
