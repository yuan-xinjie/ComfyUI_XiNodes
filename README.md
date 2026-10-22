# ComfyUI\_XiNodes

自用节点

> **全部节点已迁移到 V3 规范**（`comfy_api.latest` 的 `io.ComfyNode` + `io.Schema`）。
> 注册方式改为包级 `comfy_entrypoint()` + `ComfyExtension`，因此 `__init__.py` **不再导出 `NODE_CLASS_MAPPINGS`**——`nodes.py::load_custom_node()` 会优先取 `NODE_CLASS_MAPPINGS`，两者同时存在时 V3 入口会被忽略。
> 所有节点的 `node_id`、输入名、输出名、默认值均与 V1 版本**逐字保持一致**，旧工作流可直接加载、连线不丢；`WEB_DIRECTORY = "./js"` 保持不变，`dynamic_ports.js` / `audio_preview.js` 照常生效。
> V3 的两个硬性约束：执行方法必须命名为 `execute`（不再有 `FUNCTION`），类必须继承 `io.ComfyNode`（`VALIDATE_CLASS()` 会检查 `define_schema` 与 `execute` 两个覆写）。

## 节点清单

| node_id | 显示名 | 输出节点 |
| --- | --- | --- |
| `XiMultiFolderImageLoader` | Multi Folder Image Loader | |
| `XiSaveTextFile` | Save Text File | ✅ |
| `XiImageBatchCrossfade` | Image Batch Crossfade | |
| `XiJsonCreate` | Create JSON Object | |
| `XiJsonGetValue` | Get JSON Value | |
| `XiJsonFormat` | Format JSON | |
| `XiGeminiAPI` | Gemini API | ✅ |
| `XiSaveAudio` | Save Audio | ✅ |
| `XiImageBatch` | Image Batch (Tolerant) | |

## Multi Folder Image Loader

可同时加载最多三个文件夹内的图片，要求每个文件夹内的图片文件名一一对应，以此来进行打标等批量操作。
可输出图像、图像路径、文件名

## Save Text File

保存文本节点，可自定义：保存路径、文件前缀、分隔符、数字padding、后缀、文件格式、编码格式，可被ComfyUI的historyData.outputs捕捉

## Save Audio

保存音频节点，输入/输出均为 ComfyUI AUDIO 类型（透传）。可自定义：文件名前缀、格式（m4a/wav/flac/mp3/ogg/opus，默认 m4a）、采样率（默认 44100Hz）、码率（默认 128kbps）。保存到 ComfyUI output 目录，前端可播放，并把 ComfyUI 工作流数据写入音频文件（m4a/mp3/flac/ogg/opus），拖回 ComfyUI 界面可直接恢复工作流。

> 依赖 `av`（PyAV）；重采样优先使用 ComfyUI 自带的 `torchaudio`，缺失时自动降级为 numpy 线性插值。

## Image Batch Crossfade

给两段视频添加叠化过渡效果，可自定义：过渡时长(帧数)，根据批次resize(a/b)，透明通道处理

## Gemini API Node

google-genai协议的api节点，可以用于chat和image，可自定义url,key,model,生图比例、分辨率，最多输入三张图，可输出图像或文本

## Create JSON + Get JSON Value + Format JSON

* Create JSON创建json格式的数据，可以多json串联相加，值可以输入任意格式(string, int, float, boolean, list, json)，int, float, boolean类型用value\_in连接，优先使用连接的数据
* Get JSON Value用于提取json数据的键值，在key\_path输入要提取的键，多层键值用"键值1-键值2-键值3"表示，类似python的\["键值1"]\["键值2"]\[0]，output\_type选择输出的类型
* Format JSON输入标准的json格式的字符串，转换输出为json类型


## Image Batch (Tolerant)

官方 **批量图像 (`BatchImagesNode`)** 的容错重制版，V3 规范，节点 ID `XiImageBatch`，分类 `XiNodes`。
输出 `IMAGE` + `COUNT`（实际合并张数，方便接条件分支）。

### 与官方节点的两点差异

**1. 每个 Autogrow 槽位都是 optional**

用的就是官方那套 `io.Autogrow`，所以 **`+` 号逐个加输入口的交互与官方完全一致**（最多 50 个），而不是把 50 个端口一次性摊出来。

关键在 `comfy_api/latest/_io.py::_expand_schema_for_dynamic`：

```python
template_required = _input_type == "required"
if i < min and template_required:  -> required
else:                              -> optional
```

把模板输入写成 `io.Image.Input("image", optional=True)`，**所有生成的槽位都会落进 optional，`min` 对校验不再起作用**。该函数还专门有一条注释为 *"account for the edge case that all inputs are optional and no values are received"* 的分支，让全空时 `execute()` 收到空字典。

对比官方节点：它的槽位是 required，而前端 `ExecutableNodeDTO.resolveOutput()` 对 `mode===NEVER`（静音）直接 `return undefined`，序列化时**整个 input 键被丢弃**，后端 `validate_prompt()` 就抛 `Required input is missing`。本节点不会。

**2. 纯 `torch.cat`，不做任何对齐**

不缩放、不补通道、不改尺寸——**RGBA 就输出 RGBA**。尺寸或通道数不一致时直接报错，但错误信息会把每个槽位的 `shape / dtype / device` 全列出来，一眼看出是哪根线接错了：

```text
Xi Image Batch: torch.cat failed because the inputs are not compatible. This node
deliberately does not resize or pad channels - make them match (or insert a Resize
node). Inputs were -> image0: shape=(1, 64, 64, 3) dtype=torch.float32 device=cpu,
image1: shape=(1, 64, 64, 4) dtype=torch.float32 device=cpu
```

### 容错范围

| 情况 | 行为 |
| --- | --- |
| 槽位未连接 | 跳过 |
| 上游节点被静音 (mode=2) | 跳过，并在控制台点名是哪个节点 |
| 上游节点被旁路 (mode=4) 且类型不匹配 | 跳过，并点名 |
| 上游节点已被删除 | 跳过，并点名 |
| 上游正常但没产出值 | 跳过，并点名 |
| 传入的不是 4 维图像张量 | 跳过，并点名 |
| 全部槽位都不可用 | 由 `on_empty` 决定（见下） |

诊断信息通过 `io.Hidden.extra_pnginfo` 里的工作流 + `io.Hidden.unique_id` 反查得到，整段包在 `try/except` 里，**任何情况下都不会因为诊断逻辑把执行搞崩**。

### 参数

* **images** — Autogrow 槽位，点 `+` 添加，最多 50 个，全部可选
* **on_empty** — 仅当**所有**槽位都没有可用图像时生效，默认 `error`。**选哪个取决于下游那个图像输入是 optional 还是 required**：
  * `error` — 抛出一条**说明原因**的清晰错误（例如 `image1: 'Quiet' #1 is muted`）
  * **`no value`** — 什么都不输出（输出 `None`）。下游**可选**输入会把它当成"没接线"，带着其余输入继续正常运行
  * **`skip downstream`** — 把本节点**下游整条链**从执行队列里移除：不运行、不产出、不报错
  * `empty batch` — 输出 `(0, H, W, 3)`
  * `blank image` — 输出 1 张纯色图 `(1, H, W, 3)`
* **print_info** — 控制台摘要，并点名被静音/旁路的源节点
* 高级（默认折叠）：**empty_width** / **empty_height** / **empty_color** — 兜底张量的尺寸与灰度

### 关于"让下游以为没有输入"

ComfyUI 的图在**执行前**就由前端序列化定死了，节点无法让自己的连线消失。该选哪种模式，取决于下游那个输入是 optional 还是 required。

#### 下游是 optional 输入 → 用 `no value`

输出 `None` 在可选输入上**与"从未接线"完全等价**，已用真实执行管线逐环节验证（`get_output_from_returns` → `get_input_data` → `_async_map_node_over_list`）：

```text
下游可选输入，连到本节点的 None 输出 -> consumer 收到 ('none',)
下游可选输入，从未接线（参考组）      -> consumer 收到 ('none',)
两者完全一致
```

原因：`validate_prompt()` 只比对**声明的** `RETURN_TYPES`，从不检查运行时的值，所以 `None` 照样通过 IMAGE 类型校验；到 `get_input_data()` 里 `None` 被原样交给下游函数，而下游那个可选参数的默认值本来就是 `None`。

#### 下游是 required 输入 → 用 `skip downstream`

此时 `None` 是致命的。实测（本机 v0.41.3 + 前端 1.55.17）：

| 本节点输出 | 官方图像处理节点 | Save Image / Preview Image |
| --- | --- | --- |
| `None` | ❌ 全部 `AttributeError` | ❌ `TypeError` |
| `empty batch` `(0,H,W,3)` | ✅ Blur / Blend / Resize / Invert / `torch.cat` 正常透传 | ❌ **`IndexError`** |
| `blank image` `(1,H,W,3)` | ✅ 全部正常 | ✅ 正常存一张占位图 |

> `empty batch` 那个坑是官方 bug：`SaveImage.save_images` 在 `for` 循环**之前**就访问了 `images[0]`，所以 0 长度 batch 直接 `IndexError`。它能骗过所有处理节点，唯独骗不过输出节点。

`skip downstream` 用的是 ComfyUI **原生**调度器接口，不是 hack：

```python
# comfy_extras/nodes_loop.py:230 —— 官方 Start Loop 节点同一行代码
execution_list.inhibit_nodes(body)
```

节点通过 `io.Hidden.execution_list` + `io.Hidden.dynprompt` 拿到调度器和图，沿 `inputs` 反向邻接表算出**传递闭包下游**，调用 `ExecutionList.inhibit_nodes()` 把它们从 `pendingNodes` 里摘掉，并把 `blockCount` 里残留的等待数减回去。实测：

```text
pending before: ['2','3','4','5','6','7']
pending after : ['2','6','7']        # 3→4→5 被移除；无关分支 6→7 完好
```

#### 五种模式实测对照（全部槽位都无可用图像时）

| on_empty | 输出 | 下游是否仍执行 | 适用场景 |
| --- | --- | --- | --- |
| `error` | 抛错 | — | 默认，最安全 |
| `no value` | `None` | ✅ 是 | **下游图像输入是 optional** |
| `skip downstream` | 兜底空张量 | ❌ 整条链移出队列 | 下游是 required，且希望"什么都没发生过" |
| `empty batch` | `(0,H,W,3)` | ✅ 是 | 下游全是处理类节点，且不含 Save/Preview |
| `blank image` | `(1,H,W,3)` | ✅ 是 | 下游必须吃到图，且能接受占位图 |

> 用法：双击画布搜索 `XiImageBatch` 或 `Image Batch (Tolerant)`。
