# ComfyUI_XiNodes

自用 ComfyUI 节点，全部为 V3 规范，分类 `XiNodes`。在画布中搜索节点 ID 或显示名即可添加。

## 安装

把本目录放进 `ComfyUI/custom_nodes/`，然后：

```bash
pip install -r requirements.txt
```

TransNetV2 的权重已包含在 `transnetv2-pytorch` 里，默认不需要另外下载。

## 节点速查

| 节点 ID | 用途 |
| --- | --- |
| `XiMultiFolderImageLoader` | 同时读取最多三个文件夹的图片 |
| `XiSaveTextFile` | 保存文本文件 |
| `XiImageBatchCrossfade` | 两段图像批次叠化 |
| `XiJsonCreate` / `XiJsonGetValue` / `XiJsonFormat` | JSON 创建、取值、格式化 |
| `XiGeminiAPI` | Gemini 文本/图像 API |
| `XiSaveAudio` | 保存 AUDIO |
| `XiImageBatch` | 容错版 Batch Images |
| `XiTransNetV2ShotSplitVideo` | TransNetV2 分镜 / 固定时长分段，输出分段视频 |

## TransNetV2 Shot Split

基于 [transnetv2-pytorch](https://github.com/allenday/transnetv2_pytorch) 的镜头切割，只有一个节点：

| 节点 ID | 输入 | 输出 | 适用 |
| --- | --- | --- | --- |
| `XiTransNetV2ShotSplitVideo` | VIDEO 或本地路径 + 分段方式 | `videos`（VIDEO 列表）+ `shot_info` + `shot_count` + `video_name` | 长视频、只导出分段 |

### 两种分段方式

节点的 `分段方式`（`split_mode`）是动态输入，切换后下面的参数会跟着切换：

- `分镜`：用 TransNetV2 检测镜头边界，可调权重、阈值、设备、短段合并。
- `时长`：不加载模型，按固定秒数 `segment_sec` 切分，最后不足一段的余量也会保留。

### 怎么用

1. 接 `media`（`Load Video` 的 VIDEO），或者留空 `media`，在 `video_path` 填本地视频路径。
2. 选择 `分段方式`：
   - `分镜`：一般保持 `threshold=0.5`、短段合并默认 3/6 即可。
   - `时长`：填 `segment_sec`，例如 3 表示每 3 秒一段。
3. 后面接 `Save Video` 就能一次导出全部分段；列表输出会让下游对每段各执行一次。

只填路径的写法：

```text
XiTransNetV2ShotSplitVideo
  video_path: D:\videos\input.mp4
  split_mode: 分镜
    weights: auto (bundled)
    threshold: 0.5
  → Save Video
```

### 公共输入

| 输入/参数 | 默认 | 影响 |
| --- | --- | --- |
| `media` | 空 | 只接 ComfyUI 的 VIDEO；自动读取帧率。IMAGE 批次不支持 |
| `video_path` | 空 | 本地视频路径。填写后优先于 `media`，不经过 HTTP 上传，绕过 `--max-upload-size` |
| `split_mode` | `分镜` | `分镜` = 模型检测镜头；`时长` = 固定时长切分 |
| `output_csv` | 关 | 写入 `output/transnetv2/shots_XXXXX_.csv` |
| `seed` | 0 | 运行标识，写入 `shot_info`/CSV；不改变分段结果，但参与 ComfyUI 缓存键。固定会复用缓存，递增/随机会让同一视频重新执行 |

### 分镜模式参数

| 参数 | 默认 | 影响 |
| --- | --- | --- |
| `weights` | `auto (bundled)` | 默认用自带权重；也可把 `.pth/.pt/.safetensors` 放进 `models/transnetv2/` 后选择 |
| `threshold` | 0.5 | 边界阈值。越高切得越少，越低切得越多 |
| `device` | auto | `auto` 优先 CUDA，否则 CPU；MPS 需显式选择 |
| `drop_transition_frames` | 关 | 关：转场帧作为上一段的尾帧，所有帧保留；开：丢弃转场帧，可能每段少 1–N 帧 |
| `merge_short_sec` | 3.0 | 分段 ≤ 此秒数时尝试与相邻段合并；0 = 不合并 |
| `merge_max_sec` | 6.0 | 合并后时长上限；超过就放弃这次合并；0 = 不合并 |

### 时长模式参数

| 参数 | 默认 | 影响 |
| --- | --- | --- |
| `segment_sec` | 3.0 | 每段的目标秒数，必须大于 0 |

- 按时间轴在 `N`、`2N`、`3N`… 秒处切；每段结束于该时间点或之后的第一个实际帧。
- 最后一段不足 `segment_sec` 也会输出，不会丢帧。
- 例：10 秒视频、`segment_sec=3`，得到 3 + 3 + 3 + 1 秒四段。
- 不加载 TransNetV2，也不做短段合并；文件输入是 VFR 时按真实时间戳切，不按平均帧率估算。

从旧版升级时，`分段方式` 是新增的动态输入，旧工作流里平铺的 `weights`/`threshold` 等参数不会自动迁移；如果保存过旧的 `transnetv2`/`duration` 值，重新添加节点并选一次 `分镜`/`时长` 即可。

### 重复处理同一个视频

ComfyUI 会用输入值做缓存键。`seed` 虽然是元数据、不会改变分段结果，但它也在缓存键里：

- `fixed`：同一个视频 + 同样的参数 + 同样的 seed → 命中缓存，节点不会重新执行。
- `increment` / `randomize`：每次 queue 后 seed 会变 → 缓存失效，同一个视频会重新处理一遍。
- `decrement`：和 increment 一样会改变 seed，只是方向相反。

所以如果想重复跑同一个视频，把 seed 的生成后行为设为递增或随机；想复用缓存结果就设为固定。

### 分段边界

- `分镜` 默认切点帧属于**上一段**，是上一段的最后一帧；下一段从切点之后开始，所以相邻段首尾相接、不丢帧。
- 开启 `drop_transition_frames` 才会丢切点帧，得到的是官方 `predictions_to_scenes` 那种有间隔的分段。
- `时长` 模式同样首尾相接：上一段的结束时间就是下一段的开始时间。

### 合并规则（仅分镜模式）

- 短段优先和后一段拼；会超过 `merge_max_sec` 时改试前一段；两边都超就保留原段，不删帧。
- 合并后的段如果仍然 ≤ `merge_short_sec`，会继续尝试拼接；多个短段可一直并到接近 `merge_max_sec`。
- 例：7 个 1 秒段、上限 6 秒，会合并成约 4 秒 + 3 秒两段；3 个 2 秒段会合并成 6 秒一段。

### 输出

| 输出 | 说明 |
| --- | --- |
| `videos` | VIDEO 列表，每个输出分段一个视频，带原视频音频 |
| `shot_info` | JSON：`split_mode`、`segment_sec`、帧率、总帧数、分段起止帧/时间、`video_name`、`seed`、`detected_shots`、`merged_shots`、CSV 路径等 |
| `shot_count` | 合并后的实际分段数（时长模式就是固定切分的段数） |
| `video_name` | 加载视频的文件名，不含扩展名；无文件路径的输入为空字符串 |

`videos` 是列表输出：接一个 `Save Video`，会生成 `shot_00001_.mp4`、`shot_00002_.mp4`……

`video_name` 可直接作为 `Save Video` 的 `filename_prefix`（前端支持把字符串控件转成输入时），这样输出文件会以原视频名开头。

### 路径与内存

- `video_path` 支持绝对路径；相对路径先找 `ComfyUI/input/`，再找 ComfyUI 当前目录。
- 路径是 **ComfyUI 服务器上的路径**；远程服务时不要填浏览器本机的路径。
- `分镜` 模式流式解码，只保留约 100 帧 48×27 小图和逐帧概率，不保存原始帧。
- `时长` 模式流式解码时只收集帧时间戳，不加载模型、不做 48×27 缩放，参数切换后资源开销也更低。
- `Create Video` 这类内存型 VIDEO 也能处理，但没有文件路径，`video_name` 为空。

### CSV

`output_csv` 开启后写入 `output/transnetv2/shots_XXXXX_.csv`（UTF-8 BOM）。

列：`shot_id, start_frame, end_frame, frame_count, start_time, end_time, duration_sec, boundary_score, fps, threshold, total_frames, source, weights, device, merged_shots, video_name, seed, split_mode, segment_sec`

`end_frame` 为闭区间；`merged_shots` 表示该段由几个原始镜头合并而成；`video_name` / `seed` 原样记录本次输入和运行标识。`split_mode` 记录内部值 `transnetv2` 或 `duration`（界面显示为 `分镜`/`时长`）；`时长` 模式下 `threshold`、`weights`、`device`、`boundary_score` 留空，`segment_sec` 记录实际使用的分段时长。

## 其他节点

### Multi Folder Image Loader

同时加载最多三个文件夹的图片；要求各文件夹文件名一一对应，适合批量打标。输出图像、图像路径、文件名。

### Save Text File

保存文本。可设置保存路径、文件前缀、分隔符、数字补零位数、后缀、文件格式、编码；结果可在 ComfyUI history outputs 中获取。

### Save Audio

输入/输出都是 `AUDIO`（透传）。可设置文件名前缀、格式（m4a/wav/flac/mp3/ogg/opus，默认 m4a）、采样率（默认 44100）、码率（默认 128k）。保存到 `output`，会把工作流数据写进音频文件，拖回 ComfyUI 可恢复工作流。

### Image Batch Crossfade

给两段图像批次添加叠化。参数：过渡帧数、A/B 的 resize 方式、透明通道处理。

### Gemini API

用于 Gemini 文本/图像调用。可设置 URL、API Key、模型、生图比例、分辨率；最多输入三张图；输出图像或文本。

### JSON 节点

- `XiJsonCreate`：创建 JSON，可串联多个 JSON，值支持 string/int/float/boolean/list/JSON。
- `XiJsonGetValue`：按 `键1-键2-键3` 的路径取值，并选择输出类型。
- `XiJsonFormat`：把标准 JSON 字符串转成 JSON 类型。

### Image Batch (Tolerant)

容错版 Batch Images，最多 50 个可选图像槽位。输出 `IMAGE` + `COUNT`（实际合并张数）。

参数：

- `on_empty`：所有槽位都为空时的行为。
  - `error`：报错（默认）
  - `no value`：输出 `None`，适合下游输入是 optional
  - `skip downstream`：把下游整条链移出执行队列
  - `empty batch`：输出 `(0,H,W,3)`；只适合处理类节点，不要直接接 Save/Preview
  - `blank image`：输出 1 张占位图
- `print_info`：在控制台打印被静音/旁路/删除的源节点。
- `empty_width` / `empty_height` / `empty_color`：兜底图尺寸和颜色。

不会自动缩放、补通道或对齐；尺寸/通道不一致会报错并列出每个输入的 shape/dtype/device。
