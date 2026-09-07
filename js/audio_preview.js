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
    setup() {
        // ComfyUI's built-in file-drop workflow loader knows PNG/MP3/FLAC/etc.,
        // but in many frontend versions it does not try to parse .m4a files.
        // Our node writes prompt/workflow into M4A as ISOBMFF udta/meta/ilst,
        // so we teach the same drop handler to read that for .m4a files.
        const originalHandleFile = app.handleFile;
        if (typeof originalHandleFile !== "function") return;

        app.handleFile = async function (file, ...args) {
            const name = typeof file?.name === "string" ? file.name : "";
            const isM4a =
                name.toLowerCase().endsWith(".m4a") ||
                file.type === "audio/mp4" ||
                file.type === "audio/x-m4a";

            if (isM4a) {
                try {
                    const parseIsobmff = window.comfyAPI?.isobmff?.getFromIsobmffFile;
                    if (typeof parseIsobmff === "function") {
                        const meta = await parseIsobmff(file);
                        const payload =
                            meta?.workflow ??
                            meta?.Workflow ??
                            meta?.prompt ??
                            meta?.Prompt;

                        if (payload) {
                            const text = typeof payload === "string"
                                ? payload
                                : JSON.stringify(payload);
                            const jsonFile = new File(
                                [text],
                                name.replace(/\.m4a$/i, ".json"),
                                { type: "application/json" }
                            );
                            return originalHandleFile.call(this, jsonFile, ...args);
                        }
                    }
                } catch (e) {
                    console.warn("[XiNodes] Failed to extract workflow from m4a:", e);
                }
            }

            return originalHandleFile.call(this, file, ...args);
        };
    },
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
