import os


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


NODE_CLASS_MAPPINGS = {
    "XiSaveTextFile": XiSaveTextFile,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XiSaveTextFile": "Save Text File",
}
