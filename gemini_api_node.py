import os
import io
import re
import base64
import numpy as np
import torch
from PIL import Image

from google import genai
from google.genai import types

def tensor_to_pil(tensor_img, max_dim=1536):
    if len(tensor_img.shape) == 4:
        tensor_img = tensor_img[0]
    array = (tensor_img.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(array)

    w, h = img.size
    max_side = max(w, h)
    if max_side > max_dim:
        scale = max_dim / float(max_side)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    buf.seek(0)
    return Image.open(buf)

def pil_to_part(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=85)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

def detect_closest_aspect_ratio(pil_img):
    w, h = pil_img.size
    ratio = w / float(h)
    known_ratios = {
        "1:1": 1.0, "3:4": 3/4, "4:3": 4/3, "9:16": 9/16, "16:9": 16/9,
        "2:3": 2/3, "3:2": 3/2, "4:5": 4/5, "5:4": 5/4, "21:9": 21/9
    }
    return min(known_ratios.keys(), key=lambda k: abs(known_ratios[k] - ratio))

def clean_text_media_references(text):
    if not text:
        return ""
    text = re.sub(r'data:image/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+', '', text)
    text = re.sub(r'data:video/[a-zA-Z]+;base64,[A-Za-z0-9+/=]+', '', text)
    text = re.sub(r'!\[.*?\]\([^\)]*\)', '', text)
    text = re.sub(r'\[.*?\]\([^\)]*\.(?:mp4|webm|mov|avi|mkv|png|jpg|jpeg|webp)\)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<img[^>]*>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<video[^>]*>.*?</video>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'<video[^>]*>', '', text, flags=re.IGNORECASE)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


GLOBAL_CONTEXT_MEMORY = {}
MAX_CONTEXT_BYTES = 512 * 1024  # 512 KB

def get_context_history(session_id):
    if session_id not in GLOBAL_CONTEXT_MEMORY:
        GLOBAL_CONTEXT_MEMORY[session_id] = []
    return GLOBAL_CONTEXT_MEMORY[session_id]

def clear_context_history(session_id=None):
    global GLOBAL_CONTEXT_MEMORY
    if session_id:
        GLOBAL_CONTEXT_MEMORY.pop(session_id, None)
    else:
        GLOBAL_CONTEXT_MEMORY.clear()

def sanitize_and_trim_history(history_list):
    clean_history = []
    for item in history_list:
        if not isinstance(item, types.Content):
            continue
        
        clean_parts = []
        if item.parts:
            for p in item.parts:
                if hasattr(p, 'text') and p.text and str(p.text).strip():
                    clean_parts.append(types.Part.from_text(text=str(p.text).strip()))
        
        if not clean_parts:
            if item.role == "model":
                clean_parts.append(types.Part.from_text(text="[Generated image successfully]"))
            else:
                clean_parts.append(types.Part.from_text(text="[Provided reference image]"))

        clean_history.append(types.Content(role=item.role, parts=clean_parts))

    def calc_bytes(cnts):
        total = 0
        for c in cnts:
            for p in c.parts:
                if p.text:
                    total += len(p.text.encode('utf-8'))
        return total

    while clean_history and calc_bytes(clean_history) > MAX_CONTEXT_BYTES:
        if len(clean_history) >= 2:
            clean_history = clean_history[2:]
        else:
            clean_history.pop(0)

    return clean_history


# ==========================================
# Google GenAI 官方 SDK 驱动节点 (Gemini API Node)
# ==========================================
class XiGeminiAPI:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_url": ("STRING", {"default": "https://generativelanguage.googleapis.com", "multiline": False}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": ("STRING", {"default": "gemini-3.1-flash-image", "multiline": False}),
                "mode": (["image", "chat", "auto"], {"default": "image"}),
                "system_prompt": ("STRING", {"default": "", "multiline": True}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "aspect_ratio": (["Auto", "1:1", "3:4", "4:3", "9:16", "16:9", "2:3", "3:2", "4:5", "5:4", "21:9"], {"default": "Auto"}),
                "resolution": (["Auto", "512", "1K", "2K", "4K"], {"default": "1K"}),
                "context_memory": (["enable", "disable"], {"default": "enable"}),
                "clear_history": ("BOOLEAN", {"default": False}),
                "multi_image_mode": (["process_separately", "combine_in_one_prompt"], {"default": "process_separately"}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
                "timeout": ("INT", {"default": 180, "min": 1, "max": 3600, "step": 1}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "text")
    FUNCTION = "request_gemini"
    OUTPUT_NODE = True
    CATEGORY = "XiNodes"

    def request_gemini(self, base_url, api_key, model, mode, system_prompt, user_prompt, aspect_ratio, resolution, context_memory, clear_history, multi_image_mode, temperature, seed, timeout, **kwargs):
        # 1. 历史记忆控制
        session_id = f"{model.strip()}_{base_url.strip()}"
        if clear_history:
            clear_context_history(session_id)
            print(f"[XiNodes Gemini Memory] Context history cleared for session '{session_id}'.")

        # 2. 配置 client
        client_options = {}
        clean_base_url = base_url.strip().rstrip("/") if base_url else ""
        if clean_base_url:
            if not clean_base_url.endswith("/v1beta") and not clean_base_url.endswith("/v1"):
                http_options = types.HttpOptions(api_version="v1beta", base_url=clean_base_url)
            else:
                http_options = types.HttpOptions(base_url=clean_base_url)
            client_options["http_options"] = http_options

        if api_key and api_key.strip():
            client_options["api_key"] = api_key.strip()

        client = genai.Client(**client_options)
        model_name = model.strip() if model and model.strip() else "gemini-3.1-flash-image"

        # 3. 收集输入图片
        input_pil_images = []
        for i in range(1, 4):
            img_input = kwargs.get(f"image_{i}")
            if img_input is not None and isinstance(img_input, torch.Tensor):
                if len(img_input.shape) == 4:
                    num_batch = img_input.shape[0]
                    for b in range(num_batch):
                        pil_img = tensor_to_pil(img_input[b:b+1])
                        input_pil_images.append(pil_img)
                else:
                    pil_img = tensor_to_pil(img_input)
                    input_pil_images.append(pil_img)

        # 4. 单请求执行核心帮助函数
        def _call_single_generate(input_images_group, effective_aspect_ratio):
            current_user_parts = []
            for pil_img in input_images_group:
                current_user_parts.append(pil_to_part(pil_img))

            full_prompt_text = user_prompt.strip() if user_prompt else ""
            
            # 根据 mode 定向注入生图/Chat 指导说明
            if mode == "image":
                if input_images_group and full_prompt_text:
                    guidance = " [System Directive: You MUST generate and output a newly modified image based on the input image and instructions. Do not answer with text alone.]"
                    full_prompt_text = f"{full_prompt_text}{guidance}".strip()
                elif full_prompt_text:
                    guidance = " [System Directive: You MUST generate and output an image corresponding to the prompt. Do not answer with text alone.]"
                    full_prompt_text = f"{full_prompt_text}{guidance}".strip()
            elif mode == "chat":
                if full_prompt_text:
                    guidance = " [System Directive: Answer with text response only. Do not generate an image.]"
                    full_prompt_text = f"{full_prompt_text}{guidance}".strip()

            if full_prompt_text:
                current_user_parts.append(types.Part.from_text(text=full_prompt_text))

            contents = []
            if context_memory == "enable":
                raw_history = get_context_history(session_id)
                clean_history = sanitize_and_trim_history(raw_history)
                contents.extend(clean_history)

            user_content_obj = types.Content(role="user", parts=current_user_parts)
            contents.append(user_content_obj)

            config_kwargs = {
                "temperature": float(temperature),
            }

            # 配置 response_modalities 强制模式控制
            if mode == "image":
                config_kwargs["response_modalities"] = ["IMAGE"]
            elif mode == "chat":
                config_kwargs["response_modalities"] = ["TEXT"]

            if system_prompt and system_prompt.strip():
                config_kwargs["system_instruction"] = system_prompt.strip()

            if seed != -1 and seed is not None:
                config_kwargs["seed"] = int(seed) % 2147483647

            image_config_kwargs = {}
            if effective_aspect_ratio and effective_aspect_ratio != "Auto":
                image_config_kwargs["aspect_ratio"] = effective_aspect_ratio
            if resolution != "Auto":
                image_config_kwargs["image_size"] = resolution

            if image_config_kwargs:
                config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)

            config = types.GenerateContentConfig(**config_kwargs)

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
            except Exception as e:
                err_msg = str(e)
                if ("response_modalities" in err_msg or "responseModalities" in err_msg) and "response_modalities" in config_kwargs:
                    print(f"[XiNodes Gemini SDK Warning] Gateway does not support response_modalities ({err_msg}). Retrying without it...")
                    config_kwargs.pop("response_modalities", None)
                    config_retry = types.GenerateContentConfig(**config_kwargs)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config_retry,
                    )
                elif ("Image size" in err_msg or "not supported" in err_msg) and "image_config" in config_kwargs:
                    print(f"[XiNodes Gemini SDK Warning] Resolution limit error ({err_msg}). Retrying without image_size...")
                    image_config_kwargs.pop("image_size", None)
                    if image_config_kwargs:
                        config_kwargs["image_config"] = types.ImageConfig(**image_config_kwargs)
                    else:
                        config_kwargs.pop("image_config", None)
                    config_retry = types.GenerateContentConfig(**config_kwargs)
                    response = client.models.generate_content(
                        model=model_name,
                        contents=contents,
                        config=config_retry,
                    )
                else:
                    raise RuntimeError(f"[XiNodes Gemini SDK Error] Request failed: {e}")

            sub_parsed_imgs = []
            sub_raw_texts = []
            response_model_parts = []

            if response and response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            response_model_parts.append(part)
                            if part.inline_data and part.inline_data.data:
                                try:
                                    img_bytes = part.inline_data.data
                                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                                    sub_parsed_imgs.append(img)
                                except Exception as ex:
                                    print(f"[XiNodes Gemini SDK] Error loading inline_data: {ex}")
                            elif part.text:
                                sub_raw_texts.append(part.text)

            # 更新会话历史记忆
            if context_memory == "enable":
                history = get_context_history(session_id)
                txt_user_parts = [p for p in current_user_parts if hasattr(p, 'text') and p.text]
                if not txt_user_parts:
                    txt_user_parts = [types.Part.from_text(text="[Provided reference image]")]
                history.append(types.Content(role="user", parts=txt_user_parts))

                txt_model_parts = [p for p in response_model_parts if hasattr(p, 'text') and p.text]
                if not txt_model_parts:
                    txt_model_parts = [types.Part.from_text(text="[Generated image successfully]")]
                history.append(types.Content(role="model", parts=txt_model_parts))

                GLOBAL_CONTEXT_MEMORY[session_id] = sanitize_and_trim_history(history)
                print(f"[XiNodes Gemini Memory] Updated session '{session_id}' history (Lightweight text-only messages: {len(GLOBAL_CONTEXT_MEMORY[session_id])}).")

            return sub_parsed_imgs, sub_raw_texts

        # 5. 执行模式分支
        all_output_imgs = []
        all_output_texts = []

        if multi_image_mode == "process_separately" and len(input_pil_images) > 1:
            print(f"[XiNodes Gemini SDK] Multi-image mode: 'process_separately'. Loop processing {len(input_pil_images)} images independently...")
            for idx, single_pil in enumerate(input_pil_images):
                eff_ar = aspect_ratio
                if eff_ar == "Auto":
                    eff_ar = detect_closest_aspect_ratio(single_pil)

                imgs_res, txts_res = _call_single_generate([single_pil], eff_ar)
                all_output_imgs.extend(imgs_res)
                all_output_texts.extend(txts_res)
        else:
            eff_ar = aspect_ratio
            if eff_ar == "Auto" and len(input_pil_images) > 0:
                eff_ar = detect_closest_aspect_ratio(input_pil_images[0])

            imgs_res, txts_res = _call_single_generate(input_pil_images, eff_ar)
            all_output_imgs.extend(imgs_res)
            all_output_texts.extend(txts_res)

        # 6. 输出解析与纯粹 None 导出
        if all_output_imgs:
            tensors = []
            target_size = all_output_imgs[0].size
            for img in all_output_imgs:
                if img.size != target_size:
                    img = img.resize(target_size, Image.Resampling.BILINEAR)
                np_img = np.array(img).astype(np.float32) / 255.0
                tensors.append(torch.from_numpy(np_img))
            
            out_image_tensor = torch.stack(tensors, dim=0)
            out_text = None
        else:
            out_image_tensor = None
            raw_text_str = "\n".join(all_output_texts) if all_output_texts else ""
            out_text = clean_text_media_references(raw_text_str)

        return {
            "ui": {
                "text": [out_text if out_text else ""]
            },
            "result": (out_image_tensor, out_text)
        }

NODE_CLASS_MAPPINGS = {
    "XiGeminiAPI": XiGeminiAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XiGeminiAPI": "Gemini API",
}
