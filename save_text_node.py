import os

from comfy_api.latest import io


class XiSaveTextFile(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="XiSaveTextFile",
            display_name="Save Text File",
            category="XiNodes",
            is_output_node=True,
            description="保存文本节点，可自定义：保存路径、文件前缀、分隔符、数字padding、后缀、文件格式、编码格式，可被 ComfyUI 的 historyData.outputs 捕捉。",
            inputs=[
                io.String.Input("text", multiline=True, force_input=True, tooltip="要写入的文本。"),
                io.String.Input("path", default="./output", tooltip="相对于 ComfyUI output 目录的子目录；填 output 或留空即写到 output 根目录，也可填绝对路径。"),
                io.String.Input("filename_prefix", default="ComfyUI", tooltip="文件名前缀。"),
                io.String.Input("filename_delimiter", default="_", tooltip="前缀与序号之间的分隔符。"),
                io.Int.Input("filename_number_padding", default=5, min=1, max=10, step=1, tooltip="序号位数（自动补零）。"),
                io.String.Input("file_extension", default="txt", tooltip="文件扩展名，前导点可省略。"),
                io.Combo.Input("encoding", options=["utf-8", "utf-16", "gbk", "gb2312", "ascii"],
                               default="utf-8", tooltip="文件编码。"),
                io.String.Input("filename_suffix", default="", optional=True, tooltip="序号之后、扩展名之前的后缀。"),
            ],
            outputs=[io.String.Output(display_name="text")],
        )

    @classmethod
    def execute(cls, text, path="./output", filename_prefix="ComfyUI", filename_delimiter="_", filename_number_padding=5, file_extension="txt", encoding="utf-8", filename_suffix=""):
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

        return io.NodeOutput(
            text,
            ui={"text": [text], "file_path": [full_file_path]},
        )


NODES = [XiSaveTextFile]
