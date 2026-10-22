import os
import torch
import numpy as np
from PIL import Image, ImageOps

from comfy_api.latest import io


class XiMultiFolderImageLoader(io.ComfyNode):
    FOLDER_SLOTS = 10

    @classmethod
    def define_schema(cls):
        # 端口名与顺序必须和 js/dynamic_ports.js 里的 folder_count / folder_N /
        # IMAGE_N / FILE_PATH_N / FILENAME 保持一致，改动会破坏动态端口脚本。
        inputs = [
            io.Int.Input("folder_count", default=3, min=1, max=10, step=1,
                         tooltip="实际启用的文件夹数量，前端据此显示/隐藏 folder_N 端口。"),
            io.Int.Input("index", default=0, min=0, max=10000000,
                         tooltip="按文件名排序后处理第几个同名图片组（0 起）。"),
            io.Boolean.Input("include_extension", default=False, tooltip="FILENAME 是否带扩展名。"),
        ]
        for i in range(1, cls.FOLDER_SLOTS + 1):
            inputs.append(io.String.Input("folder_{}".format(i), default="",
                                          tooltip="第 {} 个图片文件夹的绝对路径。".format(i)))
        outputs = []
        for i in range(1, cls.FOLDER_SLOTS + 1):
            outputs.append(io.Image.Output(display_name="IMAGE_{}".format(i)))
            outputs.append(io.String.Output(display_name="FILE_PATH_{}".format(i)))
        outputs.append(io.String.Output(display_name="FILENAME"))
        return io.Schema(
            node_id="XiMultiFolderImageLoader",
            display_name="Multi Folder Image Loader",
            category="XiNodes",
            description="可同时加载多个文件夹内的图片，要求每个文件夹内的图片文件名一一对应，以此进行打标等批量操作。输出图像、图像路径、文件名。",
            inputs=inputs,
            outputs=outputs,
        )

    @staticmethod
    def get_supported_images(folder):
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

    def load_specific_base(folder, files, target_base):
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

    @classmethod
    def execute(cls, folder_count, index, include_extension=False, **kwargs):
        folders = [kwargs.get(f"folder_{i}", "") for i in range(1, 11)]
        files_array = []
        for i in range(10):
            if i < folder_count and folders[i] and os.path.exists(folders[i]):
                files_array.append(cls.get_supported_images(folders[i]))
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
            return io.NodeOutput(*empty_returns)

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
                img, path, ext = cls.load_specific_base(folder, files, current_base)
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
        return io.NodeOutput(*final_returns)


NODES = [XiMultiFolderImageLoader]
