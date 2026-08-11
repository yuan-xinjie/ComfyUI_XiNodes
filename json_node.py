import ast
import json


def _parse_json_object(text, source_name):
    if text is None or not str(text).strip():
        return {}
    try:
        data = json.loads(str(text).strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"[XiNodes] {source_name} 不是合法的 JSON：{e}")
    if not isinstance(data, dict):
        raise ValueError(f"[XiNodes] {source_name} 必须是 JSON 对象，而不是 {type(data).__name__}。请用双引号包裹键名。")
    return data


def _parse_value(text):
    s = str(text).strip()
    if not s:
        return ""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return str(text)


def _to_jsonable(v):
    if isinstance(v, str):
        return _parse_value(v)
    if v is None or isinstance(v, (int, float, bool)):
        return v
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(item) for item in v]
    if isinstance(v, dict):
        return {str(k): _to_jsonable(item) for k, item in v.items()}
    return str(v)


def _first(value):
    if isinstance(value, list):
        return value[0] if value else None
    return value


class XiJsonCreate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "key": ("STRING", {"default": "", "multiline": False}),
                "value": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "json_in": ("STRING", {"forceInput": True}),
                "value_in": ("*", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "create_json"
    CATEGORY = "XiNodes"
    INPUT_IS_LIST = True
    DESCRIPTION = "创建（或向已有 JSON 对象追加）一个键值对，输出双引号格式的 JSON 字符串。值优先取 value_in（可连任意类型），未连接时用 value 文本框（自动尝试解析为数字/布尔/列表/对象，失败则按字符串）。连接多元素列表时整个列表作为值。key 和 json_in 有多个输入时只取第一个。"

    def create_json(self, key, value, json_in=None, value_in=None):
        key_str = str(_first(key) or "").strip()
        if not key_str:
            raise ValueError("[XiNodes] 键不能为空。")

        try:
            parsed_key = json.loads(key_str)
        except json.JSONDecodeError:
            parsed_key = key_str
        if isinstance(parsed_key, (dict, list, set)):
            raise ValueError(f"[XiNodes] 键不能是可变类型（如列表/对象），收到：{key_str}")

        data = _parse_json_object(_first(json_in), "输入 json_in")

        if key_str in data:
            raise ValueError(f"[XiNodes] 键 '{key_str}' 已存在，键必须独一无二，请换一个键名。")

        source = value_in if _first(value_in) is not None else value
        if isinstance(source, list) and len(source) > 1:
            data[key_str] = [_to_jsonable(v) for v in source]
        else:
            data[key_str] = _to_jsonable(_first(source) if isinstance(source, list) else source)
        return (json.dumps(data, ensure_ascii=False, indent=4),)


def _convert_value(value, output_type):
    if output_type == "auto":
        return value
    if output_type == "string":
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False)
        if isinstance(value, bool):
            return "true" if value else "false"
        if value is None:
            return "null"
        return str(value)
    if output_type == "int":
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            if value.is_integer():
                return int(value)
            raise ValueError(f"[XiNodes] 无法将 {value} 转为 int：不是整数，已阻止静默截断。")
        if isinstance(value, str):
            s = value.strip()
            try:
                return int(s)
            except ValueError:
                try:
                    f = float(s)
                except ValueError:
                    raise ValueError(f"[XiNodes] 无法将字符串 '{value}' 转为 int。")
                if f.is_integer():
                    return int(f)
                raise ValueError(f"[XiNodes] 无法将字符串 '{value}' 转为 int：不是整数，已阻止静默截断。")
        raise ValueError(f"[XiNodes] 类型 {type(value).__name__} 不支持转为 int。")
    if output_type == "float":
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                raise ValueError(f"[XiNodes] 无法将字符串 '{value}' 转为 float。")
        raise ValueError(f"[XiNodes] 类型 {type(value).__name__} 不支持转为 float。")
    if output_type == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            s = value.strip().lower()
            if s == "true":
                return True
            if s == "false":
                return False
            raise ValueError(f"[XiNodes] 无法将字符串 '{value}' 转为 boolean，仅认 'true'/'false'（大小写不敏感）。")
        raise ValueError(f"[XiNodes] 类型 {type(value).__name__} 不支持转为 boolean。")
    if output_type == "json":
        return json.dumps(value, ensure_ascii=False, indent=4)
    return value


class XiJsonGetValue:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_text": ("STRING", {"default": "", "multiline": True}),
                "key_path": ("STRING", {"default": "", "multiline": False}),
                "output_type": (["auto", "string", "int", "float", "boolean", "json"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("value",)
    FUNCTION = "get_value"
    CATEGORY = "XiNodes"
    DESCRIPTION = "按 '-' 分割的键路径提取 JSON 对象中的字段值，例如：分镜-0-分镜01。output_type 控制输出值的类型转换（auto 保持原样），转换失败会报错。"

    def get_value(self, json_text, key_path, output_type="auto"):
        data = _parse_json_object(json_text, "输入 json_text")

        path_str = str(key_path).strip()
        if not path_str:
            raise ValueError("[XiNodes] 键路径 key_path 不能为空，例如：分镜-0-分镜01")

        parts = path_str.split("-")
        current = data
        walked = []
        for part in parts:
            walked.append(part)
            location = "-".join(walked)
            if isinstance(current, dict):
                if part not in current:
                    raise KeyError(f"[XiNodes] 在 '{location}' 处找不到键 '{part}'。可用键：{list(current.keys())}")
                current = current[part]
            elif isinstance(current, list):
                if not part.lstrip("-").isdigit():
                    raise KeyError(f"[XiNodes] '{location}' 处是列表，需要数字索引，但收到 '{part}'。列表长度：{len(current)}")
                idx = int(part)
                if idx < 0 or idx >= len(current):
                    raise IndexError(f"[XiNodes] '{location}' 处索引 {idx} 超出范围，列表长度：{len(current)}")
                current = current[idx]
            else:
                raise TypeError(f"[XiNodes] '{location}' 处的值是 {type(current).__name__}，无法继续向下提取 '{part}'。当前值：{current}")

        return (_convert_value(current, output_type),)


class XiJsonFormat:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "json_text": ("STRING", {"default": "", "multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json",)
    FUNCTION = "format_json"
    CATEGORY = "XiNodes"
    DESCRIPTION = "将符合 JSON 格式的输入校验并标准化为双引号、缩进格式的 JSON 字符串。也兼容 Python 风格（单引号、True/False/None）的文本。需要连线时可将文本框右键转为输入口。"

    def format_json(self, json_text):
        text = str(json_text).strip()
        if not text:
            raise ValueError("[XiNodes] 输入不能为空，请提供 JSON 格式的文本。")

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(text)
            except (ValueError, SyntaxError) as e:
                raise ValueError(f"[XiNodes] 输入不是合法的 JSON（也不是可转换的 Python 字面量）：{e}")

        try:
            return (json.dumps(data, ensure_ascii=False, indent=4),)
        except (TypeError, ValueError) as e:
            raise ValueError(f"[XiNodes] 内容无法序列化为 JSON：{e}")


NODE_CLASS_MAPPINGS = {
    "XiJsonCreate": XiJsonCreate,
    "XiJsonGetValue": XiJsonGetValue,
    "XiJsonFormat": XiJsonFormat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XiJsonCreate": "Create JSON Object",
    "XiJsonGetValue": "Get JSON Value",
    "XiJsonFormat": "Format JSON",
}
