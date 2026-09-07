import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Adds a playable audio preview widget to the XiSaveAudio node.
 *
 * The Python node already returns:
 *   "ui": { "audio": [{ filename, subfolder, type: "output" }] }
 *
 * The frontend needs an <audio> DOM widget on the node itself to display
 * that file after execution, exactly like ComfyUI's built-in SaveAudio.
 */
function buildAudioUrl(item) {
    const filename = item.filename ?? "";
    const subfolder = item.subfolder ?? "";
    const type = item.type ?? "output";
    const rand = Math.random().toString(36).slice(2);
    const path =
        `/view?filename=${encodeURIComponent(filename)}` +
        `&type=${encodeURIComponent(type)}` +
        `&subfolder=${encodeURIComponent(subfolder)}` +
        `&${rand}`;

    if (api && typeof api.apiURL === "function") {
        return api.apiURL(path);
    }
    return path;
}

app.registerExtension({
    name: "XiNodes.SaveAudioPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "XiSaveAudio" || !nodeData.input?.required) return;

        // Add a custom DOM widget (audio player) to the node.
        nodeData.input.required.audioUI = ["XIAUDIO_UI", {}];
    },
    getCustomWidgets() {
        return {
            XIAUDIO_UI(node, inputName) {
                const el = document.createElement("audio");
                el.controls = true;
                el.classList.add("comfy-audio");
                el.setAttribute("name", "media");

                const widget = node.addDOMWidget(inputName, "audioUI", el);
                widget.serialize = false;
                widget.options.serialize = false;

                const setAudio = (item) => {
                    if (!item || !item.filename) return;
                    el.src = buildAudioUrl(item);
                    widget.value = el.src;
                    widget.callback?.(el.src);
                    el.classList.remove("empty-audio-widget");
                };

                if (node.constructor?.nodeData?.output_node) {
                    el.classList.add("empty-audio-widget");
                    const onExecuted = node.onExecuted;
                    node.onExecuted = function (msg) {
                        onExecuted?.apply(this, arguments);
                        if (msg && msg.audio && msg.audio.length) {
                            setAudio(msg.audio[0]);
                        }
                    };
                }

                return widget;
            },
        };
    },
});
