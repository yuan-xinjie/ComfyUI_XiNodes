import torch


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
                C_B = 3

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
    "XiImageBatchCrossfade": XiImageBatchCrossfade,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XiImageBatchCrossfade": "Image Batch Crossfade",
}
