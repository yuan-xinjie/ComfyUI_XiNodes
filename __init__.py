"""ComfyUI_XiNodes - self-use nodes, written against the V3 (comfy_api) node API.

Registration note: comfy_extras/../nodes.py::load_custom_node() checks for a
module-level NODE_CLASS_MAPPINGS FIRST and only falls back to comfy_entrypoint
when it is absent, so this package deliberately exposes neither mapping - the
V3 extension below is the single source of truth. Display names come from each
io.Schema(display_name=...), and WEB_DIRECTORY is read before that branch, so
./js keeps working unchanged.
"""

import logging
from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

try:
    from . import (
        audio_save_node,
        batch_images_node,
        gemini_api_node,
        image_crossfade_node,
        image_loader_node,
        json_node,
        save_text_node,
        transnetv2_shot_split_node,
    )
except ImportError:  # loaded as loose modules rather than as a package
    import audio_save_node  # type: ignore[no-redef]
    import batch_images_node  # type: ignore[no-redef]
    import gemini_api_node  # type: ignore[no-redef]
    import image_crossfade_node  # type: ignore[no-redef]
    import image_loader_node  # type: ignore[no-redef]
    import json_node  # type: ignore[no-redef]
    import save_text_node  # type: ignore[no-redef]
    import transnetv2_shot_split_node  # type: ignore[no-redef]

WEB_DIRECTORY = "./js"

_MODULES = (
    image_loader_node,
    save_text_node,
    image_crossfade_node,
    json_node,
    gemini_api_node,
    audio_save_node,
    batch_images_node,
    transnetv2_shot_split_node,
)

NODES: list[type[io.ComfyNode]] = []
for _module in _MODULES:
    NODES.extend(_module.NODES)

assert len({cls.__name__ for cls in NODES}) == len(NODES), "duplicate node class"


class XiNodesExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return NODES


async def comfy_entrypoint() -> XiNodesExtension:
    logging.info("ComfyUI_XiNodes: registering %d V3 nodes", len(NODES))
    return XiNodesExtension()


__all__ = ["NODES", "WEB_DIRECTORY", "XiNodesExtension", "comfy_entrypoint"]
