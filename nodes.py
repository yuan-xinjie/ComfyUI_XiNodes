import os
import torch
import numpy as np
from PIL import Image, ImageOps

try:
    from .gemini_api_node import XiGeminiAPI
except ImportError:
    from gemini_api_node import XiGeminiAPI

class XiMultiFolderImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        req = {
            "folder_count": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            "index": ("INT", {"default": 0, "min": 0, "max": 10000000}),
            "include_extension": ("BOOLEAN", {"default": False}),
        }
        for i in range(1, 11):
            req[f"folder_{i}"] = ("STRING", {"default": "", "multiline": False})
        return {"required": req}

    _ret_types = []
    _ret_names = []
    for i in range(1, 11):
        _ret_types.extend(["IMAGE", "STRING"])
        _ret_names.extend([f"IMAGE_{i}", f"FILE_PATH_{i}"])
    _ret_types.append("STRING")
    _ret_names.append("FILENAME")
    
    RETURN_TYPES = tuple(_ret_types)
    RETURN_NAMES = tuple(_ret_names)
    FUNCTION = "load_images"
    CATEGORY = "XiNodes"

    def get_supported_images(self, folder):
        if not folder or not os.path.exists(folder):
            return []
        supported_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
        files = []
        try:
            for f in os.listdir(folder):
                if os.path.isfile(os.path.join(folder, f)):
                    base, ext = os.path.splitext(f)
                    if ext.lower() in supported_exts:
                        files.append({'base': base, 'ext': ext, 'full': f})
        except Exception as e:
            print(f"Error reading folder {folder}: {e}")
        return files

    def load_specific_base(self, folder, files, target_base):
        if not folder or not os.path.exists(folder):
            return None, "", ""
        matched_file = next((f for f in files if f['base'] == target_base), None)
        if matched_file is None:
            raise ValueError(f"Sync Match Error: Target Folder '{folder}' does NOT contain any supported image named '{target_base}'.")
        image_path = os.path.join(folder, matched_file['full'])
        ext = matched_file['ext']
        try:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            return image_tensor, image_path, ext
        except Exception as e:
            print(f"XiNodes Loader Error: Failed to load '{image_path}': {e}")
            raise RuntimeError(f"XiNodes Loader Error: Failed to load image '{image_path}': {e}")

    def load_images(self, folder_count, index, include_extension=False, **kwargs):
        folders = [kwargs.get(f"folder_{i}", "") for i in range(1, 11)]
        files_array = []
        for i in range(10):
            if i < folder_count and folders[i] and os.path.exists(folders[i]):
                files_array.append(self.get_supported_images(folders[i]))
            else:
                files_array.append([])

        all_bases = set()
        for i in range(folder_count):
            if files_array[i]: 
                all_bases.update([f['base'] for f in files_array[i]])
                
        sorted_bases = sorted(list(all_bases))
        if len(sorted_bases) == 0:
            print("[XiNodes Info]: No valid images found crossing all input folders limits.")
            empty_returns = []
            for i in range(10):
                empty_returns.extend([None, ""])
            empty_returns.append("")
            return tuple(empty_returns)
            
        if index >= len(sorted_bases):
            raise ValueError(f"XiNodes -> Index ({index}) reaches the end! Total processed: {len(sorted_bases)}.")

        current_base = sorted_bases[index]
        images_result = []
        paths_result = []
        first_valid_ext = ""

        for i in range(10):
            folder = folders[i]
            files = files_array[i]
            if i < folder_count and folder and os.path.exists(folder):
                img, path, ext = self.load_specific_base(folder, files, current_base)
                images_result.append(img)
                paths_result.append(path)
                if ext and not first_valid_ext:
                    first_valid_ext = ext
            else:
                images_result.append(None)
                paths_result.append("")

        final_filename = current_base
        if include_extension and first_valid_ext:
             final_filename += first_valid_ext

        final_returns = []
        for i in range(10):
            final_returns.append(images_result[i])
            final_returns.append(paths_result[i])
            
        final_returns.append(final_filename)
        return tuple(final_returns)

class XiSaveTextFile:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
                "path": ("STRING", {"default": "./output"}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "filename_number_padding": ("INT", {"default": 5, "min": 1, "max": 10, "step": 1}),
                "file_extension": ("STRING", {"default": "txt"}),
                "encoding": (["utf-8", "utf-16", "gbk", "gb2312", "ascii"], {"default": "utf-8"}),
            },
            "optional": {
                "filename_suffix": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "save_text_file"
    OUTPUT_NODE = True
    CATEGORY = "XiNodes"

    def save_text_file(self, text, path="./output", filename_prefix="ComfyUI", filename_delimiter="_", filename_number_padding=5, file_extension="txt", encoding="utf-8", filename_suffix=""):
        import folder_paths
        try:
            base_output_dir = folder_paths.get_output_directory()
        except Exception:
            base_output_dir = os.path.abspath("./output")
            
        path = path.strip()
        filename_prefix = filename_prefix.strip()
        filename_delimiter = filename_delimiter.strip()
        filename_suffix = filename_suffix.strip()
        file_extension = file_extension.strip()
        
        if file_extension.startswith("."):
            file_extension = file_extension[1:]
            
        if os.path.isabs(path):
            target_dir = path
        else:
            clean_path = path
            if clean_path.startswith("./") or clean_path.startswith(".\\"):
                clean_path = clean_path[2:]
            if not clean_path or clean_path.lower() == "output":
                target_dir = base_output_dir
            else:
                target_dir = os.path.abspath(os.path.join(base_output_dir, clean_path))
                
        counter = 1
        while True:
            padded_counter = str(counter).zfill(filename_number_padding)
            filename = f"{filename_prefix}{filename_delimiter}{padded_counter}{filename_suffix}.{file_extension}"
            full_file_path = os.path.join(target_dir, filename)
            if not os.path.exists(full_file_path):
                break
            counter += 1
            
        os.makedirs(target_dir, exist_ok=True)
        try:
            with open(full_file_path, "w", encoding=encoding) as f:
                f.write(text)
        except Exception as e:
            print(f"[XiNodes Error] Failed to write text file to {full_file_path}: {e}")
            raise e
            
        return {
            "ui": {
                "text": [text],
                "file_path": [full_file_path]
            },
            "result": (text,)
        }

class XiImageBatchCrossfade:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images_a": ("IMAGE",),
                "images_b": ("IMAGE",),
                "transition_frames": ("INT", {"default": 10, "min": 0, "max": 10000, "step": 1}),
                "transition_type": (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "linear"}),
                "resize_behavior": (["resize_to_match_a", "resize_to_match_b", "error"], {"default": "resize_to_match_a"}),
                "alpha_handling": (["keep_alpha_channels", "layer_on_black", "layer_on_white", "layer_on_green"], {"default": "keep_alpha_channels"}),
                "frame_duration_behavior": (["freeze_ends", "reduce_frames"], {"default": "freeze_ends"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "crossfade"
    CATEGORY = "XiNodes"

    def crossfade(self, images_a, images_b, transition_frames, transition_type, resize_behavior, alpha_handling, frame_duration_behavior="freeze_ends"):
        if len(images_a.shape) == 3:
            images_a = images_a.unsqueeze(0)
        if len(images_b.shape) == 3:
            images_b = images_b.unsqueeze(0)
            
        def apply_alpha_handling(img, handling):
            if img.shape[-1] != 4:
                return img
            rgb = img[..., :3]
            alpha = img[..., 3:]
            if handling == "layer_on_black":
                return rgb * alpha
            elif handling == "layer_on_white":
                return rgb * alpha + (1.0 - alpha)
            elif handling == "layer_on_green":
                green_color = torch.tensor([0.0, 1.0, 0.0], dtype=rgb.dtype, device=rgb.device).view(1, 1, 1, 3)
                return rgb * alpha + (1.0 - alpha) * green_color
            else:
                return img

        images_a = apply_alpha_handling(images_a, alpha_handling)
        images_b = apply_alpha_handling(images_b, alpha_handling)
        
        N_A, H_A, W_A, C_A = images_a.shape
        N_B, H_B, W_B, C_B = images_b.shape

        if C_A != C_B:
            if C_A > 3:
                images_a = images_a[..., :3]
                C_A = 3
            if C_B > 3:
                images_b = images_b[..., :3]
                C_B = 3
            if C_A == 1 and C_B == 3:
                images_a = images_a.repeat(1, 1, 1, 3)
                C_A = 3
            elif C_A == 3 and C_B == 1:
                images_b = images_b.repeat(1, 1, 1, 3)
                C_A = 3

        if H_A != H_B or W_A != W_B:
            if resize_behavior == "resize_to_match_a":
                images_b_permuted = images_b.permute(0, 3, 1, 2)
                images_b_resized = torch.nn.functional.interpolate(
                    images_b_permuted, size=(H_A, W_A), mode="bilinear", align_corners=False
                )
                images_b = images_b_resized.permute(0, 2, 3, 1)
                H_B, W_B = H_A, W_A
            elif resize_behavior == "resize_to_match_b":
                images_a_permuted = images_a.permute(0, 3, 1, 2)
                images_a_resized = torch.nn.functional.interpolate(
                    images_a_permuted, size=(H_B, W_B), mode="bilinear", align_corners=False
                )
                images_a = images_a_resized.permute(0, 2, 3, 1)
                H_A, W_A = H_B, W_B
            elif resize_behavior == "error":
                raise ValueError(f"Resolution mismatch between images_a ({W_A}x{H_A}) and images_b ({W_B}x{H_B}) and resize_behavior is 'error'.")

        actual_transition = min(transition_frames, N_A, N_B)
        if actual_transition < transition_frames:
            print(f"[XiNodes Warning] Requested transition_frames ({transition_frames}) is larger than one of the batch sizes (A: {N_A}, B: {N_B}). Capping transition to {actual_transition} frames.")

        if actual_transition == 0:
            out_images = torch.cat([images_a, images_b], dim=0)
            return (out_images,)

        if actual_transition == 1:
            alphas = torch.tensor([0.5], dtype=images_a.dtype, device=images_a.device)
        else:
            alphas = torch.linspace(0.0, 1.0, steps=actual_transition, dtype=images_a.dtype, device=images_a.device)

        if transition_type == "ease_in":
            alphas = alphas ** 2
        elif transition_type == "ease_out":
            alphas = 1.0 - (1.0 - alphas) ** 2
        elif transition_type == "ease_in_out":
            alphas = alphas ** 2 * (3.0 - 2.0 * alphas)

        alphas = alphas.view(actual_transition, 1, 1, 1)

        parts = []
        if frame_duration_behavior == "reduce_frames":
            images_a_part = images_a[:N_A - actual_transition]
            images_b_part = images_b[actual_transition:]
            fade_out = images_a[-actual_transition:]
            fade_in = images_b[:actual_transition]
            transition_part = (1.0 - alphas) * fade_out + alphas * fade_in
            if images_a_part.shape[0] > 0:
                parts.append(images_a_part)
            parts.append(transition_part)
            if images_b_part.shape[0] > 0:
                parts.append(images_b_part)
        else:
            n_freeze_a = (actual_transition + 1) // 2
            n_freeze_b = actual_transition // 2
            fade_out_list = [images_a[-n_freeze_a:]]
            if n_freeze_b > 0:
                fade_out_list.append(images_a[-1:].repeat(n_freeze_b, 1, 1, 1))
            fade_out = torch.cat(fade_out_list, dim=0)
            fade_in_list = []
            if n_freeze_a > 0:
                fade_in_list.append(images_b[0:1].repeat(n_freeze_a, 1, 1, 1))
            if n_freeze_b > 0:
                fade_in_list.append(images_b[:n_freeze_b])
            fade_in = torch.cat(fade_in_list, dim=0)

            transition_part = (1.0 - alphas) * fade_out + alphas * fade_in
            images_a_part = images_a[:-n_freeze_a]
            images_b_part = images_b[n_freeze_b:]

            if images_a_part.shape[0] > 0:
                parts.append(images_a_part)
            parts.append(transition_part)
            if images_b_part.shape[0] > 0:
                parts.append(images_b_part)

        out_images = torch.cat(parts, dim=0)
        return (out_images,)

NODE_CLASS_MAPPINGS = {
    "XiMultiFolderImageLoader": XiMultiFolderImageLoader,
    "XiSaveTextFile": XiSaveTextFile,
    "XiImageBatchCrossfade": XiImageBatchCrossfade,
    "XiGeminiAPI": XiGeminiAPI
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XiMultiFolderImageLoader": "Multi Folder Image Loader",
    "XiSaveTextFile": "Save Text File",
    "XiImageBatchCrossfade": "Image Batch Crossfade",
    "XiGeminiAPI": "Gemini API"
}
