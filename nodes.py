import os
import torch
import numpy as np
from PIL import Image, ImageOps

class XiMultiFolderImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        # 初始化 10 个预置插槽的定义
        req = {
            "folder_count": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
            "index": ("INT", {"default": 0, "min": 0, "max": 10000000}),
            "include_extension": ("BOOLEAN", {"default": False}),
        }
        for i in range(1, 11):
            req[f"folder_{i}"] = ("STRING", {"default": "", "multiline": False})
        return {"required": req}

    # 一次性静态注册包含 10组交替 (IMAGE, PATH) + 1路 FILENAME 的巨型返回规格参数
    # 采用交织排列能完美地配合前端的数组动态修剪而不至于打乱索引或者中断之前接好的线路
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
        # 如果未提供或者此槽位并非目标考察区域，由外部直接拦断，此处只管取文件
        if not folder or not os.path.exists(folder):
            return None, "", ""

        matched_file = next((f for f in files if f['base'] == target_base), None)
        
        if matched_file is None:
            # 严格控制组：遇到找不到的就报错阻断执行！
            raise ValueError(f"Sync Match Error: Target Folder '{folder}' does NOT contain any supported image named '{target_base}'. All active folders chosen within folder_count limit must contain an identical filename sequence.")
        
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
        
        # 将在限制范围内的且存在合法路径的槽位数据纳入检索，丢弃空槽或者超出个数限制以外的所有影响变量
        for i in range(10):
            if i < folder_count and folders[i] and os.path.exists(folders[i]):
                files_array.append(self.get_supported_images(folders[i]))
            else:
                files_array.append([])

        # 抽取出全网所有合法集合文件夹中共有的，或者各异的基础图片名称合并排序为一个统一基底轮播清单
        all_bases = set()
        for i in range(folder_count):
            if files_array[i]: 
                all_bases.update([f['base'] for f in files_array[i]])
                
        sorted_bases = sorted(list(all_bases))
         
        # 若大盘整体搜空，返回无状态交织格式
        if len(sorted_bases) == 0:
            print("[XiNodes Info]: No valid images found crossing all input folders limits.")
            empty_returns = []
            for i in range(10):
                empty_returns.extend([None, ""])
            empty_returns.append("")
            return tuple(empty_returns)
            
        # 超界防无限死循环检测截断机制
        if index >= len(sorted_bases):
            raise ValueError(f"XiNodes -> Index ({index}) reaches the end! Stopping auto-queue loop. Total processed: {len(sorted_bases)}.")

        current_base = sorted_bases[index]

        # 从 1~10 对各渠道并行进行寻迹读取与数据合成返回
        images_result = []
        paths_result = []
        first_valid_ext = ""

        for i in range(10):
            folder = folders[i]
            files = files_array[i]
            # 只有位于设定参与数目，且具有合法填充才进行检索动作：
            if i < folder_count and folder and os.path.exists(folder):
                img, path, ext = self.load_specific_base(folder, files, current_base)
                images_result.append(img)
                paths_result.append(path)
                # 记录最先找到的某个真实有效被使用文件的扩展后缀（以备 filename 输出时补充完整）
                if ext and not first_valid_ext:
                    first_valid_ext = ext
            else:
                # 不符合要求的越界通道直接留空丢弃，向外输出 None 和 ""
                images_result.append(None)
                paths_result.append("")

        # 整理 FILENAME 的尾端处理
        final_filename = current_base
        if include_extension and first_valid_ext:
             final_filename += first_valid_ext

        # 交织装配输出，向外推送
        final_returns = []
        for i in range(10):
            final_returns.append(images_result[i])
            final_returns.append(paths_result[i])
            
        final_returns.append(final_filename)
        return tuple(final_returns)

NODE_CLASS_MAPPINGS = {
    "XiMultiFolderImageLoader": XiMultiFolderImageLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XiMultiFolderImageLoader": "Multi Folder Image Loader"
}
