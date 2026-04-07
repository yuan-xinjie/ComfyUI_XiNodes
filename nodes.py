import os
import torch
import numpy as np
from PIL import Image, ImageOps

class XiMultiFolderImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_1": ("STRING", {"default": "", "multiline": False}),
                "folder_2": ("STRING", {"default": "", "multiline": False}),
                "folder_3": ("STRING", {"default": "", "multiline": False}),
                "index": ("INT", {"default": 0, "min": 0, "max": 10000000}),
                "include_extension": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("IMAGE_1", "IMAGE_2", "IMAGE_3", "FILE_PATH_1", "FILE_PATH_2", "FILE_PATH_3", "FILENAME")
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

    def load_images(self, folder_1, folder_2, folder_3, index, include_extension=False):
        # 1. 扫描三个文件夹获取合法的图片信息
        files1 = self.get_supported_images(folder_1)
        files2 = self.get_supported_images(folder_2)
        files3 = self.get_supported_images(folder_3)

        # 2. 获取所有的 base 文件名集合（求并集），从而确定一共有多少组图
        all_bases = set([f['base'] for f in files1])
        all_bases.update([f['base'] for f in files2])
        all_bases.update([f['base'] for f in files3])

        # 排序保证每一次根据 index 获取文件的顺序一致且固定
        sorted_bases = sorted(list(all_bases))
        
        # 检查是否没有任何文件
        if len(sorted_bases) == 0:
            print("XiNodes Loader: No images found in any of the provided folders.")
            return (self.create_placeholder(), self.create_placeholder(), self.create_placeholder(), "", "", "", "")
        
        # 当 Auto Queue 或 Primitive node 的 index 超出总图片数的时候，通过抛出报错主动叫停 ComfyUI 以停止批处理
        if index >= len(sorted_bases):
            print(f"XiNodes Info: Index {index} is exactly out of bounds. Processing of {len(sorted_bases)} images has successfully finished.")
            raise ValueError(f"XiNodes -> Index {index} reaches the end. Stopping queue. Total images processed: {len(sorted_bases)}.")
            
        current_base = sorted_bases[index]

        # 3. 在三个文件夹中寻找匹配的文件（若没找到直接抛出错误停止后续操作）
        img1, path1, ext1 = self.load_specific_base(folder_1, files1, current_base)
        img2, path2, ext2 = self.load_specific_base(folder_2, files2, current_base)
        img3, path3, ext3 = self.load_specific_base(folder_3, files3, current_base)

        final_filename = current_base
        if include_extension:
            first_ext = ext1 if ext1 else (ext2 if ext2 else ext3)
            if first_ext:
                final_filename = current_base + first_ext

        return (img1, img2, img3, path1, path2, path3, final_filename)

    def load_specific_base(self, folder, files, target_base):
        # 如果根本没传文件夹，静默返回占位符而不参与此约束
        if not folder or not os.path.exists(folder):
            return self.create_placeholder(), "", ""

        # 寻找本文件夹内，去扩展名后是否具有与目标 base 一致的文件
        matched_file = next((f for f in files if f['base'] == target_base), None)
        
        if matched_file is None:
            # 根据用户要求，填了路径的文件夹必须存在同名基底图像文件，否则直接中止并红字报错给用户
            raise ValueError(f"Sync Match Error: Target Folder '{folder}' does NOT contain any supported image named '{target_base}'. All specified folders must contain an identical filename for this sequence.")
        
        # 如果存在，则加载真实图片作为 tensor 返回
        image_path = os.path.join(folder, matched_file['full'])
        ext = matched_file['ext']
        try:
            i = Image.open(image_path)
            i = ImageOps.exif_transpose(i)
            # 转为 RGB 格式防止单通道灰度图导致的维度报错
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            # 将 numpy image 转成 tensor [1, H, W, 3] 范围 0.0 - 1.0 的格式 (ComfyUI 标准)
            image_np = np.array(image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)[None,]
            return image_tensor, image_path, ext
        except Exception as e:
            print(f"XiNodes Loader Error: Failed to load '{image_path}': {e}")
            raise RuntimeError(f"XiNodes Loader Error: Failed to load image '{image_path}': {e}")

    def create_placeholder(self):
        # 返回一个 64x64 的全黑占位张量
        return torch.zeros((1, 64, 64, 3), dtype=torch.float32)

NODE_CLASS_MAPPINGS = {
    "XiMultiFolderImageLoader": XiMultiFolderImageLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XiMultiFolderImageLoader": "Multi Folder Image Loader"
}
