NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from . import image_loader_node, save_text_node, image_crossfade_node, json_node, gemini_api_node
except ImportError:
    import image_loader_node, save_text_node, image_crossfade_node, json_node, gemini_api_node

for module in (image_loader_node, save_text_node, image_crossfade_node, json_node, gemini_api_node):
    NODE_CLASS_MAPPINGS.update(module.NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(module.NODE_DISPLAY_NAME_MAPPINGS)

WEB_DIRECTORY = "./js"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
