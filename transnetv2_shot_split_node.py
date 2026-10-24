# -*- coding: utf-8 -*-
"""Xi TransNetV2 Shot Split (V3 / comfy_api).

Shot boundary detection and splitting with the `transnetv2-pytorch` package.
The wheel ships the official converted weights, so the node works out of the
box; custom weights can be dropped into `models/transnetv2/`.

Design notes
------------
* `XiTransNetV2ShotSplitVideo` has a `split_mode` DynamicCombo with two
  choices: `分镜` (model-driven shot detection) and `时长` (fixed seconds per
  segment).  In `时长` mode the model is never loaded; the file is only scanned
  for frame timestamps.
* `分镜` mode streams the file through PyAV, keeps only a 100-frame
  48x27 ring buffer plus one probability per frame, and returns lazily trimmed
  VIDEO segments.  It intentionally has no IMAGE output, because producing
  frames necessarily materialises them.
* `videos` is a list output.  ComfyUI runs downstream nodes once per list item,
  so a single Save Video exports every segment in one queue run.
* Inference is windowed exactly like transnetv2-pytorch's own predict_frames
  (100-frame windows, 50-frame step, 25-frame padding), so long inputs do not
  allocate the whole sequence in one forward pass.
"""

import csv
import json
import os
import random
import threading
from bisect import bisect_left
from collections import OrderedDict
from contextlib import contextmanager
from fractions import Fraction

import numpy as np
import torch
import torch.nn.functional as F

from comfy_api.latest import Input, InputImpl, Types, io, ui

try:
    import folder_paths
except Exception:  # only when imported outside a full ComfyUI install
    folder_paths = None

MODEL_HEIGHT = 27
MODEL_WIDTH = 48
WINDOW_SIZE = 100
WINDOW_STEP = 50
WINDOW_PAD = 25

BUNDLED_WEIGHTS = "auto (bundled)"
WEIGHTS_FOLDER = "transnetv2"
WEIGHT_EXTENSIONS = (".pth", ".pt", ".pt2", ".bin", ".ckpt", ".safetensors")
CSV_SUBFOLDER = "transnetv2"

if folder_paths is not None:
    try:
        folder_paths.add_model_folder_path(
            WEIGHTS_FOLDER,
            os.path.join(folder_paths.models_dir, WEIGHTS_FOLDER),
            is_default=True,
        )
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# device helpers
# --------------------------------------------------------------------------- #
def _mps_available():
    try:
        return bool(torch.backends.mps.is_available())
    except (AttributeError, RuntimeError):
        return False


def _comfy_device():
    try:
        import comfy.model_management as model_management

        return model_management.get_torch_device()
    except Exception:
        return None


def _comfy_forced_cpu():
    try:
        from comfy.cli_args import args

        return bool(getattr(args, "cpu", False))
    except Exception:
        return False


def _resolve_device(choice):
    choice = str(choice or "auto").strip().lower()
    if choice not in ("auto", "cuda", "cpu", "mps"):
        raise ValueError(
            "[XiNodes TransNetV2] unknown device '{}'. Expected one of: "
            "auto, cuda, cpu, mps.".format(choice)
        )

    if choice == "cpu":
        return "cpu"

    forced_cpu = _comfy_forced_cpu()
    comfy_dev = _comfy_device()

    if choice == "cuda":
        if forced_cpu:
            raise ValueError(
                "[XiNodes TransNetV2] ComfyUI was started with --cpu, so device='cuda' "
                "is not allowed. Use device='auto' or 'cpu'."
            )
        if not torch.cuda.is_available():
            raise ValueError(
                "[XiNodes TransNetV2] device='cuda' requested, but CUDA is not available."
            )
        if comfy_dev is not None and comfy_dev.type == "cuda":
            return str(comfy_dev)
        return "cuda:0"

    if choice == "mps":
        if not _mps_available():
            raise ValueError(
                "[XiNodes TransNetV2] device='mps' requested, but MPS is not available."
            )
        return "mps"

    # auto: CUDA when usable, otherwise CPU. MPS is opt-in on purpose, matching
    # transnetv2-pytorch's recommendation (some 3D ops fall back to CPU there
    # and results differ from a pure CPU run).
    if forced_cpu:
        return "cpu"
    if comfy_dev is not None and comfy_dev.type == "cuda":
        return str(comfy_dev)
    if torch.cuda.is_available():
        return "cuda:0"
    if _mps_available():
        print(
            "[XiNodes TransNetV2] MPS detected; device='auto' uses CPU for consistent "
            "results. Pick device='mps' explicitly if you prefer speed over consistency."
        )
    return "cpu"


# --------------------------------------------------------------------------- #
# model loading / caching
# --------------------------------------------------------------------------- #
_MODEL_CACHE = OrderedDict()
_MODEL_LOCK = threading.Lock()


@contextmanager
def _preserve_global_state():
    """Undo the global RNG / deterministic side effects of TransNetV2.__init__."""
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = None
    if torch.cuda.is_available():
        try:
            cuda_state = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_state = None
    deterministic = bool(torch.are_deterministic_algorithms_enabled())
    warn_only = False
    if hasattr(torch, "is_deterministic_algorithms_warn_only_enabled"):
        try:
            warn_only = bool(torch.is_deterministic_algorithms_warn_only_enabled())
        except Exception:
            warn_only = False
    cudnn_deterministic = bool(torch.backends.cudnn.deterministic)
    cudnn_benchmark = bool(torch.backends.cudnn.benchmark)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            try:
                torch.cuda.set_rng_state_all(cuda_state)
            except Exception:
                pass
        try:
            torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        except Exception:
            pass
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


def _resolve_video_path(video_path):
    """Resolve a user supplied local video path.

    Absolute paths are used as-is.  Relative paths are tried against
    ``ComfyUI/input`` first (the same folder Load Video scans) and then against
    the current working directory.  Empty/whitespace means "not provided".
    """
    if video_path is None:
        return None
    raw = str(video_path).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
        raw = raw[1:-1].strip()
    if not raw:
        return None
    if raw.lower().startswith("file://"):
        raw = raw[7:]
        if raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
            raw = raw[1:]
    raw = os.path.expandvars(os.path.expanduser(raw))

    if os.path.isabs(raw):
        candidates = [raw]
    else:
        candidates = []
        relative = raw
        normalized = raw.replace("\\", "/")
        if normalized.lower().startswith("input/"):
            relative = raw[6:].lstrip("\\/")
        if folder_paths is not None:
            try:
                candidates.append(
                    os.path.join(folder_paths.get_input_directory(), relative)
                )
            except Exception:
                pass
        candidates.append(os.path.abspath(raw))

    seen = set()
    checked = []
    for candidate in candidates:
        candidate = os.path.abspath(candidate)
        if candidate in seen:
            continue
        seen.add(candidate)
        checked.append(candidate)
        if os.path.isfile(candidate):
            return candidate

    raise ValueError(
        "[XiNodes TransNetV2] video_path not found (or is not a file): {}. "
        "Tried: {}".format(raw, " | ".join(checked))
    )


def _video_basename(media, resolved_path=None):
    """Filename stem of the loaded video, or ``""`` for non-file inputs."""
    path = resolved_path
    if path is None:
        get_source = getattr(media, "get_stream_source", None)
        if callable(get_source):
            try:
                source = get_source()
            except Exception:
                source = None
            if isinstance(source, (str, os.PathLike)):
                path = os.fspath(source)
    if not path:
        return ""
    name = os.path.basename(str(path))
    return os.path.splitext(name)[0]


def _split_mode_kind(split_mode):
    """Return ``(kind, values)`` for the DynamicCombo value."""
    if split_mode is None:
        return "transnetv2", {}
    if isinstance(split_mode, str):
        values = {"split_mode": split_mode}
    elif isinstance(split_mode, dict):
        values = dict(split_mode)
    else:
        raise ValueError(
            "[XiNodes TransNetV2] split_mode must be a dict, got {}.".format(
                type(split_mode).__name__
            )
        )
    kind = values.get("split_mode") or values.get("mode") or "分镜"
    kind = str(kind).strip().lower()
    if kind in ("transnetv2", "auto", "ai", "分镜", "镜头", "镜头检测"):
        return "transnetv2", values
    if kind in ("duration", "fixed", "fixed_duration", "时长", "固定时长", "按时间"):
        return "duration", values
    raise ValueError(
        "[XiNodes TransNetV2] unknown split_mode '{}'; expected 分镜 or 时长.".format(
            kind
        )
    )


def _weights_options():
    options = [BUNDLED_WEIGHTS]
    if folder_paths is not None:
        try:
            files = folder_paths.get_filename_list(WEIGHTS_FOLDER)
        except Exception:
            files = []
        for name in sorted(files):
            if os.path.splitext(name)[1].lower() in WEIGHT_EXTENSIONS:
                options.append(name)
    return options


def _resolve_weights_path(weights_name):
    if not weights_name or weights_name == BUNDLED_WEIGHTS:
        return None
    if folder_paths is None:
        raise RuntimeError(
            "[XiNodes TransNetV2] folder_paths is unavailable; cannot resolve custom weights."
        )
    path = folder_paths.get_full_path(WEIGHTS_FOLDER, weights_name)
    if path is None or not os.path.isfile(path):
        raise ValueError(
            "[XiNodes TransNetV2] weights file '{}' was not found in models/{}.".format(
                weights_name, WEIGHTS_FOLDER
            )
        )
    return path


def _load_state_dict(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(path)
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)

    if isinstance(state, dict):
        for key in ("state_dict", "model_state_dict", "model", "module"):
            inner = state.get(key)
            if isinstance(inner, dict) and inner:
                state = inner
                break
    if not isinstance(state, dict) or not state:
        raise ValueError(
            "[XiNodes TransNetV2] '{}' does not contain a state_dict.".format(
                os.path.basename(path)
            )
        )
    if all(isinstance(k, str) and k.startswith("module.") for k in state):
        state = {k[len("module.") :]: v for k, v in state.items()}
    return state


def _load_model(device, weights_name):
    weights_path = _resolve_weights_path(weights_name)
    cache_key = (
        device,
        weights_name,
        os.path.getmtime(weights_path) if weights_path else None,
    )

    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            _MODEL_CACHE.move_to_end(cache_key)
            return cached, weights_path

        try:
            from transnetv2_pytorch import TransNetV2
        except Exception as e:
            raise RuntimeError(
                "[XiNodes TransNetV2] the 'transnetv2-pytorch' package is required. "
                "Install it into ComfyUI's Python environment with: "
                "pip install transnetv2-pytorch"
            ) from e

        # TransNetV2.__init__ seeds Python/NumPy/Torch and enables deterministic
        # algorithms globally. Restore everything so unrelated ComfyUI nodes are
        # unaffected.
        with _preserve_global_state():
            model = TransNetV2(device=device)
            if weights_path is not None:
                state = _load_state_dict(weights_path)
                try:
                    model.load_state_dict(state)
                except Exception as e:
                    raise RuntimeError(
                        "[XiNodes TransNetV2] failed to load custom weights '{}': {}".format(
                            os.path.basename(weights_path), e
                        )
                    ) from e
            model.to(torch.device(device))
            model.eval()

        _MODEL_CACHE[cache_key] = model
        while len(_MODEL_CACHE) > 4:
            _MODEL_CACHE.popitem(last=False)
        return model, weights_path


# --------------------------------------------------------------------------- #
# inference helpers
# --------------------------------------------------------------------------- #
def _images_to_model_frames(images):
    """Convert a ComfyUI IMAGE batch [N,H,W,C] to uint8 [N,27,48,3]."""
    if images.ndim == 3:
        images = images.unsqueeze(0)
    if images.ndim != 4:
        raise ValueError(
            "[XiNodes TransNetV2] IMAGE input must be [N,H,W,C], got {}.".format(
                tuple(images.shape)
            )
        )
    count, channels = images.shape[0], images.shape[-1]
    if count < 1:
        raise ValueError("[XiNodes TransNetV2] IMAGE input is empty.")
    if channels < 3:
        raise ValueError(
            "[XiNodes TransNetV2] IMAGE input needs at least 3 channels (RGB), "
            "got {}.".format(channels)
        )

    out = torch.empty((count, MODEL_HEIGHT, MODEL_WIDTH, 3), dtype=torch.uint8)
    # 32 frames per chunk keeps the resize temporary at a few hundred MB at
    # 1080p while being much faster than a per-frame loop.
    for start in range(0, count, 32):
        chunk = images[start : start + 32, ..., :3]
        chunk = chunk.detach().to(dtype=torch.float32, device="cpu")
        x = chunk.permute(0, 3, 1, 2)
        x = F.interpolate(
            x,
            size=(MODEL_HEIGHT, MODEL_WIDTH),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        x = x.clamp_(0.0, 1.0).mul_(255.0).round_().to(torch.uint8)
        out[start : start + chunk.shape[0]] = x.permute(0, 2, 3, 1)
    return out


def _predict_frames(model, frames):
    """Windowed inference matching transnetv2-pytorch's predict_frames()."""
    count = int(frames.shape[0])
    padded_len = count + WINDOW_PAD + (
        WINDOW_PAD + WINDOW_STEP - (count % WINDOW_STEP or WINDOW_STEP)
    )
    padded = torch.empty((padded_len, MODEL_HEIGHT, MODEL_WIDTH, 3), dtype=torch.uint8)
    padded[:WINDOW_PAD] = frames[0]
    padded[WINDOW_PAD : WINDOW_PAD + count] = frames
    padded[WINDOW_PAD + count :] = frames[-1]

    total_windows = (padded_len - WINDOW_SIZE) // WINDOW_STEP + 1
    progress = None
    try:
        import comfy.utils

        progress = comfy.utils.ProgressBar(total_windows)
    except Exception:
        progress = None

    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except Exception:
        throw_exception_if_processing_interrupted = None

    predictions = []
    with torch.inference_mode():
        for window in range(total_windows):
            if throw_exception_if_processing_interrupted is not None:
                throw_exception_if_processing_interrupted()
            start = window * WINDOW_STEP
            batch = padded[start : start + WINDOW_SIZE].unsqueeze(0).to(model.device)
            single_frame, _ = model.predict_raw(batch)
            predictions.append(
                single_frame[0, WINDOW_PAD : WINDOW_PAD + WINDOW_STEP, 0]
                .detach()
                .float()
                .cpu()
            )
            if progress is not None:
                progress.update(1)

    if not predictions:
        return np.zeros((0,), dtype=np.float32)
    return torch.cat(predictions, dim=0)[:count].numpy()


def _stream_predict_video(model, media):
    """Decode a VIDEO and optionally run TransNetV2 in one pass.

    When ``model`` is None only the frame timestamps/count are collected, which
    is all the fixed-duration mode needs.  Otherwise only a ~100-frame 48x27
    ring buffer, the per-frame probabilities and the frame timestamps are kept
    in memory, so peak RAM no longer scales with the video length.  Returns
    ``(predictions, frame_count, timestamps)`` where predictions is None when
    no model was supplied and timestamps are seconds relative to the active trim
    window.
    """
    try:
        import av
    except ImportError as e:
        raise RuntimeError(
            "[XiNodes TransNetV2] streaming VIDEO mode requires PyAV. "
            "Install it with: pip install av"
        ) from e

    source = media.get_stream_source()
    try:
        start_time, trim_duration = media.get_active_trim_window()
    except Exception:
        start_time, trim_duration = 0.0, 0.0
    end_time = start_time + trim_duration if trim_duration else None
    try:
        fps_est = float(media.get_frame_rate())
    except Exception:
        fps_est = 24.0
    if fps_est <= 0:
        fps_est = 24.0

    buffer = []
    predictions = []
    timestamps = []
    count = 0
    last_small = None

    def emit_window():
        batch = torch.from_numpy(np.stack(buffer[:WINDOW_SIZE], axis=0))
        with torch.inference_mode():
            single_frame, _ = model.predict_raw(batch.unsqueeze(0).to(model.device))
        predictions.append(
            single_frame[0, WINDOW_PAD : WINDOW_PAD + WINDOW_STEP, 0]
            .detach()
            .float()
            .cpu()
        )
        del buffer[:WINDOW_STEP]

    try:
        from comfy.model_management import throw_exception_if_processing_interrupted
    except Exception:
        throw_exception_if_processing_interrupted = None

    with av.open(source, mode="r") as container:
        if not container.streams.video:
            raise ValueError("[XiNodes TransNetV2] the VIDEO has no video stream.")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        if start_time > 0 and stream.time_base is not None:
            try:
                offset = int(start_time / float(stream.time_base))
                container.seek(offset, stream=stream, backward=True)
            except Exception:
                pass

        for frame in container.decode(stream):
            if throw_exception_if_processing_interrupted is not None:
                throw_exception_if_processing_interrupted()
            timestamp = frame.time
            if timestamp is not None:
                if timestamp < start_time - 1e-6:
                    continue
                if end_time is not None and timestamp >= end_time:
                    break
            if model is not None:
                # Decode directly into a 48x27 float RGB frame (same
                # gbrpf32le colour conversion as get_components).  Swscale's
                # downscale differs from torch's antialias by <0.003 in the
                # probability domain in testing, and keeps the pass fast.
                rgb = frame.reformat(
                    width=MODEL_WIDTH, height=MODEL_HEIGHT, format="gbrpf32le"
                ).to_ndarray()
                small = (
                    torch.from_numpy(rgb)
                    .clamp_(0.0, 1.0)
                    .mul_(255.0)
                    .round_()
                    .byte()
                    .numpy()
                )
                small = np.array(small, dtype=np.uint8, copy=True)
                if count == 0:
                    buffer.extend([small] * WINDOW_PAD)
                buffer.append(small)
                last_small = small
            timestamps.append(
                count / fps_est if timestamp is None else timestamp - start_time
            )
            count += 1
            if model is not None:
                while len(buffer) >= WINDOW_SIZE:
                    emit_window()

    if count == 0:
        raise ValueError(
            "[XiNodes TransNetV2] no frames could be decoded from this VIDEO."
        )

    if model is None:
        return None, count, timestamps

    # Same tail padding as transnetv2-pytorch's predict_frames.
    end_pad = WINDOW_PAD + WINDOW_STEP - (
        count % WINDOW_STEP or WINDOW_STEP
    )
    buffer.extend([last_small] * end_pad)
    while len(buffer) >= WINDOW_SIZE:
        emit_window()

    preds = torch.cat(predictions, dim=0)[:count].numpy()
    return preds, count, timestamps


def _video_time_base(media):
    """Return the source's time_base in seconds, or 0 if it cannot be read."""
    try:
        import av

        source = media.get_stream_source()
        with av.open(source, mode="r") as container:
            if container.streams.video:
                return float(container.streams.video[0].time_base)
    except Exception:
        pass
    return 0.0


def _detect_shots(predictions, threshold, drop_transition_frames=False):
    """Group frames into shots.

    By default the transition frames (a run of ``prediction > threshold``) are
    kept as the **last frames of the preceding shot**, so the cut frame belongs
    to the outgoing segment and the next segment starts after it.  The shots are
    contiguous and every source frame appears in exactly one segment.  With
    ``drop_transition_frames=True`` the runs are excluded from both neighbours,
    matching the official ``predictions_to_scenes`` behaviour (which leaves the
    transition frames out).  Per-shot ``boundary_score`` is the peak
    probability of the run that starts that shot.
    """
    count = int(predictions.shape[0])
    binary = predictions > threshold

    shots = []
    start = 0
    boundary_score = None
    i = 0

    if drop_transition_frames:
        while i < count:
            if not binary[i]:
                i += 1
                continue
            j = i
            while j < count and binary[j]:
                j += 1
            score = float(predictions[i:j].max())
            if i > start:
                shots.append(
                    {"start": start, "end": i, "boundary_score": boundary_score}
                )
            boundary_score = score
            start = j
            i = j
        if start < count:
            shots.append(
                {"start": start, "end": count, "boundary_score": boundary_score}
            )
        if not shots:
            shots = [{"start": 0, "end": count, "boundary_score": None}]
        return shots

    while i < count:
        if not binary[i]:
            i += 1
            continue
        j = i
        while j < count and binary[j]:
            j += 1
        score = float(predictions[i:j].max())
        if i > start:
            # The transition belongs to the outgoing shot: close the previous
            # shot at the end of the run and start the next one after it.
            shots.append(
                {"start": start, "end": j, "boundary_score": boundary_score}
            )
            start = j
            boundary_score = score
        i = j

    if start < count:
        shots.append({"start": start, "end": count, "boundary_score": boundary_score})
    if not shots:
        shots = [{"start": 0, "end": count, "boundary_score": None}]
    return shots


def _slice_audio(audio, start, end, fps):
    if not audio:
        return None
    waveform = audio.get("waveform")
    sample_rate = int(audio.get("sample_rate", 0))
    if waveform is None or sample_rate <= 0:
        return audio
    total_samples = int(waveform.shape[-1])
    first = max(0, min(total_samples - 1, int(round(start / fps * sample_rate))))
    last = max(first + 1, min(total_samples, int(round(end / fps * sample_rate))))
    return {"waveform": waveform[..., first:last], "sample_rate": sample_rate}


def _write_csv(rows, meta):
    if folder_paths is None:
        raise RuntimeError(
            "[XiNodes TransNetV2] folder_paths is unavailable; cannot write the CSV."
        )
    output_dir = folder_paths.get_output_directory()
    full_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
        "{}/shots".format(CSV_SUBFOLDER), output_dir
    )
    os.makedirs(full_folder, exist_ok=True)
    csv_path = os.path.join(full_folder, "{}_{:05}_.csv".format(filename, counter))

    fieldnames = [
        "shot_id",
        "start_frame",
        "end_frame",
        "frame_count",
        "start_time",
        "end_time",
        "duration_sec",
        "boundary_score",
        "fps",
        "threshold",
        "total_frames",
        "source",
        "weights",
        "device",
        "merged_shots",
        "video_name",
        "seed",
        "split_mode",
        "segment_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "shot_id": row["shot_id"],
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "frame_count": row["frame_count"],
                    "start_time": "{:.3f}".format(row["start_time"]),
                    "end_time": "{:.3f}".format(row["end_time"]),
                    "duration_sec": "{:.3f}".format(row["duration"]),
                    "boundary_score": (
                        ""
                        if row["boundary_score"] is None
                        else "{:.4f}".format(row["boundary_score"])
                    ),
                    "fps": "{:.6g}".format(meta["fps"]),
                    "threshold": (
                        ""
                        if meta.get("threshold") is None
                        else "{:g}".format(meta["threshold"])
                    ),
                    "total_frames": meta["total_frames"],
                    "source": meta["source"],
                    "weights": meta["weights"],
                    "device": meta["device"],
                    "merged_shots": int(row.get("merged_shots", 1)),
                    "video_name": meta.get("video_name", ""),
                    "seed": meta.get("seed", 0),
                    "split_mode": meta.get("split_mode", "transnetv2"),
                    "segment_sec": (
                        ""
                        if meta.get("segment_sec") is None
                        else "{:g}".format(meta["segment_sec"])
                    ),
                }
            )
    return csv_path


def _shot_rows(detected, fps_value):
    """Turn ``_detect_shots`` output into serialisable per-shot rows."""
    rows = []
    for index, shot in enumerate(detected, start=1):
        start = int(shot["start"])
        end = int(shot["end"])
        frame_count = end - start
        if frame_count <= 0:
            continue
        rows.append(
            {
                "shot_id": index,
                "start_frame": start,
                "end_frame": end - 1,
                "frame_count": frame_count,
                "start_time": start / fps_value,
                "end_time": end / fps_value,
                "duration": frame_count / fps_value,
                "boundary_score": shot["boundary_score"],
                "merged_shots": 1,
            }
        )
    return rows


def _fixed_duration_rows(
    total_frames, fps_value, segment_sec, timestamps=None, total_end_time=None
):
    """Build contiguous rows at fixed time boundaries.

    With real ``timestamps`` (VFR/streaming) each boundary is an absolute time
    on the timeline; the segment ends on the first frame at/after that boundary.
    Without timestamps the frame count is ``round(segment_sec * fps)``.  The
    final partial segment is kept.
    """
    fps_value = max(float(fps_value), 1e-6)
    segment_sec = float(segment_sec)
    rows = []

    if timestamps is not None:
        if len(timestamps) != total_frames:
            raise RuntimeError(
                "[XiNodes TransNetV2] timestamp count does not match frame count "
                "({} vs {}).".format(len(timestamps), total_frames)
            )
        start = 0
        boundary = segment_sec
        while start < total_frames:
            end = bisect_left(timestamps, boundary, lo=start + 1, hi=total_frames)
            if end <= start:
                end = start + 1
            start_ts = timestamps[start]
            if end < total_frames:
                end_ts = timestamps[end]
            else:
                end_ts = total_end_time
                if end_ts is None or end_ts <= timestamps[-1]:
                    if total_frames > 1:
                        tail = timestamps[-1] - timestamps[-2]
                    else:
                        tail = 1.0 / fps_value
                    end_ts = timestamps[-1] + max(tail, 1.0 / fps_value)
            if end_ts <= start_ts:
                end_ts = start_ts + 1.0 / fps_value
            frame_count = end - start
            rows.append(
                {
                    "shot_id": len(rows) + 1,
                    "start_frame": start,
                    "end_frame": end - 1,
                    "frame_count": frame_count,
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "duration": end_ts - start_ts,
                    "boundary_score": None,
                    "merged_shots": 1,
                }
            )
            start = end
            boundary += segment_sec
    else:
        frames_per_segment = max(1, int(round(segment_sec * fps_value)))
        for start in range(0, total_frames, frames_per_segment):
            end = min(total_frames, start + frames_per_segment)
            frame_count = end - start
            start_ts = start / fps_value
            end_ts = end / fps_value
            rows.append(
                {
                    "shot_id": len(rows) + 1,
                    "start_frame": start,
                    "end_frame": end - 1,
                    "frame_count": frame_count,
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "duration": end_ts - start_ts,
                    "boundary_score": None,
                    "merged_shots": 1,
                }
            )
    return rows


def _merge_short_shots(rows, short_sec=3.0, max_sec=6.0):
    """Merge short shots into neighbours without exceeding ``max_sec``.

    A shot (or a group already built from several shots) with
    ``duration <= short_sec`` is merged with the **next** contiguous group when
    the sum fits, otherwise with the **previous** one.  When neither neighbour
    fits, the shot is left as-is instead of dropping frames.  The scan repeats
    so several short shots can chain into one longer segment, and stops when no
    further merge is possible.

    ``short_sec <= 0`` or ``max_sec <= 0`` disables merging entirely.  Only
    frame-contiguous rows are merged, which also keeps this safe when
    ``drop_transition_frames`` left gaps between shots.

    Returns ``(merged_rows, merge_operations)``.
    """
    if short_sec is None or max_sec is None or short_sec <= 0 or max_sec <= 0:
        return rows, 0
    if len(rows) < 2:
        return rows, 0

    eps = 1e-9
    groups = [dict(row, merged_shots=int(row.get("merged_shots", 1))) for row in rows]
    merge_operations = 0

    def contiguous(left, right):
        return int(left["end_frame"]) + 1 == int(right["start_frame"])

    def merge_pair(left, right):
        merged = {
            "shot_id": left["shot_id"],
            "start_frame": int(left["start_frame"]),
            "end_frame": int(right["end_frame"]),
            "frame_count": int(left["frame_count"]) + int(right["frame_count"]),
            "start_time": float(left["start_time"]),
            "end_time": float(right["end_time"]),
            "duration": float(left["duration"]) + float(right["duration"]),
            "boundary_score": left["boundary_score"],
            "merged_shots": int(left["merged_shots"]) + int(right["merged_shots"]),
        }
        return merged

    while True:
        merged_any = False
        for index, group in enumerate(groups):
            if float(group["duration"]) > short_sec + eps:
                continue

            left_index = right_index = None
            if index + 1 < len(groups):
                nxt = groups[index + 1]
                if (
                    contiguous(group, nxt)
                    and float(group["duration"]) + float(nxt["duration"])
                    <= max_sec + eps
                ):
                    left_index, right_index = index, index + 1
            if left_index is None and index > 0:
                prev = groups[index - 1]
                if (
                    contiguous(prev, group)
                    and float(prev["duration"]) + float(group["duration"])
                    <= max_sec + eps
                ):
                    left_index, right_index = index - 1, index
            if left_index is None:
                continue

            merged = merge_pair(groups[left_index], groups[right_index])
            groups[left_index : right_index + 1] = [merged]
            merge_operations += 1
            merged_any = True
            break

        if not merged_any:
            break

    for index, row in enumerate(groups, start=1):
        row["shot_id"] = index
    return groups, merge_operations


# --------------------------------------------------------------------------- #
# node
# --------------------------------------------------------------------------- #
class XiTransNetV2ShotSplitVideo(io.ComfyNode):
    """Video-only splitter with ``分镜`` and ``时长`` modes.

    ``分镜`` decodes once at 48x27, keeps only a 100-frame ring buffer plus
    per-frame probabilities, and returns lazily trimmed VIDEO segments.
    ``时长`` never loads the model and only needs frame timestamps.  Both
    modes deliberately have no IMAGE output so the node can never accidentally
    materialise every frame.
    """

    @classmethod
    def define_schema(cls):
        weights = _weights_options()
        return io.Schema(
            node_id="XiTransNetV2ShotSplitVideo",
            display_name="TransNetV2 Shot Split (Video Only)",
            category="XiNodes",
            description=(
                "流式解码 VIDEO 并按镜头分段，只有 videos 输出。"
                "split_mode 可切换两种方式：分镜 = 模型检测镜头"
                "（阈值、权重、短段合并等参数）；时长 = 按固定秒数切分"
                "（segment_sec，最后不足一段的余量也会保留）。"
                "videos 使用懒加载裁剪，接 Save Video 可一次导出全部分段。"
            ),
            search_aliases=[
                "shot detection",
                "scene detection",
                "shot boundary",
                "split video",
                "fixed duration",
                "segment by time",
                "transnet",
                "long video",
                "分镜",
                "镜头切割",
                "固定时长",
                "按时间分段",
                "长视频",
            ],
            inputs=[
                io.Video.Input(
                    "media",
                    optional=True,
                    tooltip=(
                        "接 ComfyUI 原生 VIDEO（推荐 Load Video）。"
                        "文件型 VIDEO 会流式解码；Create Video 产生的内存型 VIDEO 仍可处理，"
                        "但它本来就已经在内存里了。也可以不接，直接填下面的 video_path。"
                    ),
                ),
                io.String.Input(
                    "video_path",
                    optional=True,
                    default="",
                    display_name="video path (local)",
                    multiline=False,
                    placeholder=r"例如 D:\videos\input.mp4 或 input/clip.mp4",
                    tooltip=(
                        "本地视频路径，填写后优先于 media，绕过 ComfyUI 的 HTTP 上传限制。"
                        "支持绝对路径；相对路径先找 ComfyUI/input，再找当前目录。"
                        "留空则使用 media 输入。"
                    ),
                ),
                io.DynamicCombo.Input(
                    "split_mode",
                    display_name="分段方式",
                    options=[
                        io.DynamicCombo.Option(
                            "分镜",
                            [
                                io.Combo.Input(
                                    "weights",
                                    options=weights,
                                    default=BUNDLED_WEIGHTS,
                                    tooltip=(
                                        "auto (bundled) 使用 transnetv2-pytorch 自带的官方权重，"
                                        "无需下载。也可把 .pth/.safetensors 放到 models/{} "
                                        "后在这里选择。".format(WEIGHTS_FOLDER)
                                    ),
                                ),
                                io.Float.Input(
                                    "threshold",
                                    default=0.5,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    tooltip=(
                                        "镜头边界概率阈值：越高越保守（切得少），"
                                        "越低越敏感（切得多）。"
                                    ),
                                ),
                                io.Combo.Input(
                                    "device",
                                    options=["auto", "cuda", "cpu", "mps"],
                                    default="auto",
                                    tooltip=(
                                        "auto = 有 CUDA 用 CUDA，否则 CPU。MPS 需显式选择："
                                        "TransNetV2 的部分 3D 卷积在 MPS 上会回退到 CPU，"
                                        "结果与纯 CPU 不一致。"
                                    ),
                                ),
                                io.Boolean.Input(
                                    "drop_transition_frames",
                                    optional=True,
                                    default=False,
                                    label_on="drop",
                                    label_off="keep",
                                    tooltip=(
                                        "默认关闭：转场帧作为上一段的尾帧，所有帧都保留。"
                                        "开启后丢弃转场帧（与官方 predictions_to_scenes 一致），"
                                        "非第一段可能少 1–N 帧。"
                                    ),
                                ),
                                io.Float.Input(
                                    "merge_short_sec",
                                    optional=True,
                                    default=3.0,
                                    min=0.0,
                                    max=3600.0,
                                    step=0.1,
                                    tooltip=(
                                        "短镜头阈值：时长 ≤ 此值的分段会尝试和相邻分段拼接。"
                                        "0 = 不拼接；默认 3 秒。"
                                    ),
                                ),
                                io.Float.Input(
                                    "merge_max_sec",
                                    optional=True,
                                    default=6.0,
                                    min=0.0,
                                    max=3600.0,
                                    step=0.1,
                                    tooltip=(
                                        "拼接后的时长上限：与相邻段相加超过此值就放弃这次拼接；"
                                        "多个短段可以在不超过此值的前提下继续合并。"
                                        "0 = 不拼接；默认 6 秒。"
                                    ),
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "时长",
                            [
                                io.Float.Input(
                                    "segment_sec",
                                    default=3.0,
                                    min=0.1,
                                    max=3600.0,
                                    step=0.1,
                                    tooltip=(
                                        "固定分段时长（秒）。按时间轴每 N 秒切一段，"
                                        "最后不足 N 秒的部分会作为最后一段保留。"
                                    ),
                                ),
                            ],
                        ),
                    ],
                    tooltip=(
                        "分段方式：分镜 = 模型检测镜头；时长 = 按固定时长切分。"
                        "切换后下方参数会跟着切换。"
                    ),
                ),
                io.Boolean.Input(
                    "output_csv",
                    default=False,
                    label_on="on",
                    label_off="off",
                    tooltip=(
                        "开启后把分段表写入 ComfyUI output/{}/shots_XXXXX_.csv"
                        "（UTF-8 BOM，Excel 可直接打开）。".format(CSV_SUBFOLDER)
                    ),
                ),
                io.Int.Input(
                    "seed",
                    optional=True,
                    default=0,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip=(
                        "本次运行的 seed，会写入 shot_info/CSV 方便标识。"
                        "TransNetV2 推理是确定性的，seed 不改变分段结果；"
                        "但它参与 ComfyUI 缓存键：fixed 会复用同一视频的缓存，"
                        "increment/randomize 会改变 seed，让同一个视频每次 queue 都重新处理。"
                    ),
                ),
            ],
            outputs=[
                io.Video.Output(
                    "videos",
                    display_name="videos",
                    is_output_list=True,
                    tooltip=(
                        "每个镜头一个懒加载 VIDEO 分段（列表），带原视频音频；"
                        "接 Save Video 一次导出全部分段。"
                    ),
                ),
                io.String.Output(
                    "shot_info",
                    display_name="shot_info",
                    tooltip="JSON：帧率、总帧数、分段起止帧/时间、边界分数、CSV 路径等。",
                ),
                io.Int.Output(
                    "shot_count",
                    display_name="shot_count",
                    tooltip="输出分段数量；开启短段合并后可能少于检测到的镜头数。",
                ),
                io.String.Output(
                    "video_name",
                    display_name="video_name",
                    tooltip=(
                        "加载视频的文件名，不含扩展名。没有文件路径的输入"
                        "（IMAGE 批次或内存型 VIDEO）为空字符串。"
                    ),
                ),
            ],
        )

    @classmethod
    def validate_inputs(
        cls,
        media=None,
        video_path="",
        split_mode=None,
        output_csv=False,
        seed=0,
    ):
        # ComfyUI validates prompts without executing upstream nodes, so linked
        # inputs arrive here as None.  Presence/type of the link is checked by
        # ComfyUI; only validate literals we can actually see.
        if media is not None and not isinstance(media, Input.Video):
            return (
                "video 输入只接受 ComfyUI 的 VIDEO，实际收到 {}。".format(
                    type(media).__name__
                )
            )
        try:
            kind, values = _split_mode_kind(split_mode)
        except ValueError as e:
            return str(e)
        if kind == "duration":
            segment_value = values.get("segment_sec", 3.0)
            try:
                segment_sec = 3.0 if segment_value is None else float(segment_value)
            except (TypeError, ValueError):
                return "segment_sec 必须是数字。"
            if segment_sec <= 0:
                return "segment_sec 必须大于 0。"
            return True
        threshold = values.get("threshold", 0.5)
        if threshold is not None:
            try:
                threshold = float(threshold)
            except (TypeError, ValueError):
                return "threshold 必须是 0-1 之间的浮点数。"
            if not 0.0 <= threshold <= 1.0:
                return "threshold 必须在 0-1 之间。"
        device = values.get("device", "auto")
        if device is not None and str(device) not in ("auto", "cuda", "cpu", "mps"):
            return "device 必须是 auto / cuda / cpu / mps 之一。"
        weights = values.get("weights", BUNDLED_WEIGHTS)
        if (
            weights is not None
            and weights != BUNDLED_WEIGHTS
            and weights not in _weights_options()
        ):
            return "权重文件 '{}' 不存在，请重新选择或放入 models/{}。".format(
                weights, WEIGHTS_FOLDER
            )
        return True

    @classmethod
    def execute(
        cls,
        media=None,
        video_path="",
        split_mode=None,
        output_csv=False,
        seed=0,
    ):
        seed = 0 if seed is None else int(seed)
        kind, values = _split_mode_kind(split_mode)

        if kind == "transnetv2":
            weights = values.get("weights", BUNDLED_WEIGHTS) or BUNDLED_WEIGHTS
            threshold_value = values.get("threshold", 0.5)
            threshold = 0.5 if threshold_value is None else float(threshold_value)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    "[XiNodes TransNetV2] threshold must be between 0 and 1."
                )
            device = values.get("device", "auto") or "auto"
            drop_transition_frames = bool(
                values.get("drop_transition_frames", False)
            )
            merge_short_sec = (
                3.0
                if values.get("merge_short_sec") is None
                else max(0.0, float(values.get("merge_short_sec")))
            )
            merge_max_sec = (
                6.0
                if values.get("merge_max_sec") is None
                else max(0.0, float(values.get("merge_max_sec")))
            )
            segment_sec = None
        else:
            segment_value = values.get("segment_sec", 3.0)
            segment_sec = 3.0 if segment_value is None else float(segment_value)
            if segment_sec <= 0:
                raise ValueError(
                    "[XiNodes TransNetV2] segment_sec must be greater than 0."
                )
            weights = BUNDLED_WEIGHTS
            threshold = None
            device = "auto"
            drop_transition_frames = False
            merge_short_sec = 0.0
            merge_max_sec = 0.0

        resolved_video_path = _resolve_video_path(video_path)
        if resolved_video_path is not None:
            media = InputImpl.VideoFromFile(resolved_video_path)
        elif media is None:
            raise ValueError(
                "[XiNodes TransNetV2] connect media or fill video_path with a "
                "local video file."
            )
        if not isinstance(media, Input.Video):
            raise ValueError(
                "[XiNodes TransNetV2] this node expects a ComfyUI VIDEO, got {}.".format(
                    type(media).__name__
                )
            )
        video_name = _video_basename(media, resolved_video_path)

        if isinstance(media, InputImpl.VideoFromFile):
            streaming = True
            components = None
            frame_rate = media.get_frame_rate()
        else:
            # Create Video / other in-memory sources: frames already exist.
            streaming = False
            components = media.get_components()
            frame_rate = components.frame_rate
        if frame_rate is None or float(frame_rate) <= 0:
            frame_rate = Fraction(24, 1)
        fps_value = float(frame_rate)

        # One tick of the source time base is much smaller than a frame.  Add
        # half a tick when handing the start time to the trimmer so that
        # int(start_time / time_base) lands on the intended frame even when
        # float rounding puts the value a hair below the frame's PTS.
        time_tick = (_video_time_base(media) if streaming else 0.0) or (
            1.0 / fps_value
        )

        if kind == "transnetv2":
            # ---- model + predictions ------------------------------------
            resolved_device = _resolve_device(device)
            model, weights_path = _load_model(resolved_device, weights)
            if streaming:
                predictions, total_frames, timestamps = _stream_predict_video(
                    model, media
                )
            else:
                images = components.images
                if not isinstance(images, torch.Tensor) or images.ndim != 4:
                    raise ValueError(
                        "[XiNodes TransNetV2] expected frames [N,H,W,C], got {}.".format(
                            type(images)
                        )
                    )
                total_frames = int(images.shape[0])
                if total_frames < 1:
                    raise ValueError("[XiNodes TransNetV2] input has no frames.")
                predictions = _predict_frames(model, _images_to_model_frames(images))
                timestamps = None
            if len(predictions) != total_frames:
                raise RuntimeError(
                    "[XiNodes TransNetV2] prediction count does not match frame count "
                    "({} vs {}).".format(len(predictions), total_frames)
                )

            # ---- boundaries ---------------------------------------------
            detected = _detect_shots(predictions, threshold, drop_transition_frames)
            rows = _shot_rows(detected, fps_value)
            if streaming:
                # VFR sources: report and trim by the actual frame timestamps,
                # not by frame_index / average_fps.
                for row in rows:
                    start = int(row["start_frame"])
                    end = int(row["end_frame"]) + 1
                    row["start_time"] = timestamps[start]
                    if end < total_frames:
                        row["end_time"] = timestamps[end]
                    else:
                        try:
                            end_ts = float(media.get_duration())
                        except Exception:
                            end_ts = 0.0
                        if end_ts <= timestamps[-1]:
                            if total_frames > 1:
                                tail = timestamps[-1] - timestamps[-2]
                            else:
                                tail = 1.0 / fps_value
                            end_ts = timestamps[-1] + max(tail, time_tick)
                        row["end_time"] = end_ts
                    row["duration"] = row["end_time"] - row["start_time"]
            detected_count = len(rows)
            rows, merge_operations = _merge_short_shots(
                rows, merge_short_sec, merge_max_sec
            )
            weights_label = (
                os.path.basename(weights_path)
                if weights_path is not None
                else BUNDLED_WEIGHTS
            )
            device_label = str(model.device)
            predictions_for_info = predictions
        else:
            # ---- fixed duration: no model at all -----------------------
            model = None
            weights_path = None
            predictions_for_info = None
            merge_operations = 0
            weights_label = ""
            device_label = ""
            if streaming:
                _, total_frames, timestamps = _stream_predict_video(None, media)
                try:
                    total_end_time = float(media.get_duration())
                except Exception:
                    total_end_time = None
            else:
                images = components.images
                if not isinstance(images, torch.Tensor) or images.ndim != 4:
                    raise ValueError(
                        "[XiNodes TransNetV2] expected frames [N,H,W,C], got {}.".format(
                            type(images)
                        )
                    )
                total_frames = int(images.shape[0])
                if total_frames < 1:
                    raise ValueError("[XiNodes TransNetV2] input has no frames.")
                timestamps = None
                total_end_time = total_frames / fps_value
            rows = _fixed_duration_rows(
                total_frames,
                fps_value,
                segment_sec,
                timestamps=timestamps,
                total_end_time=total_end_time,
            )
            detected_count = len(rows)

        videos = []
        for row in rows:
            start = int(row["start_frame"])
            end = int(row["end_frame"]) + 1
            if streaming:
                start_ts = row["start_time"]
                end_ts = row["end_time"]
                duration = max(end_ts - start_ts, time_tick)
                segment = media.as_trimmed(
                    start_ts + 0.5 * time_tick,
                    duration,
                    strict_duration=False,
                )
                if segment is None:
                    raise RuntimeError(
                        "[XiNodes TransNetV2] could not trim VIDEO segment {}-{}.".format(
                            start, end - 1
                        )
                    )
                videos.append(segment)
            else:
                shot_images = components.images[start:end]
                video_images = (
                    shot_images[..., :3]
                    if shot_images.shape[-1] > 3
                    else shot_images
                )
                videos.append(
                    InputImpl.VideoFromComponents(
                        Types.VideoComponents(
                            images=video_images,
                            audio=_slice_audio(
                                components.audio, start, end, fps_value
                            ),
                            frame_rate=frame_rate,
                        )
                    )
                )

        if not rows:
            raise RuntimeError(
                "[XiNodes TransNetV2] no shots were produced from this input."
            )

        kept_frames = sum(row["frame_count"] for row in rows)
        if predictions_for_info is not None:
            transition_frames = total_frames - kept_frames
            boundary_frames = int((predictions_for_info > threshold).sum())
            drop_flag = bool(drop_transition_frames)
        else:
            transition_frames = 0
            boundary_frames = 0
            drop_flag = False
        mode = "video_stream" if streaming else "video_components"
        meta = {
            "fps": fps_value,
            "threshold": threshold,
            "total_frames": total_frames,
            "source": "video",
            "weights": weights_label,
            "device": device_label,
            "mode": mode,
            "split_mode": kind,
            "segment_sec": segment_sec,
            "video_name": video_name,
            "seed": seed,
        }

        csv_path = None
        if output_csv:
            csv_path = _write_csv(rows, meta)

        info = {
            "shot_count": len(rows),
            "detected_shots": detected_count,
            "total_frames": total_frames,
            "kept_frames": kept_frames,
            "transition_frames": transition_frames,
            "boundary_frames": boundary_frames,
            "split_mode": kind,
            "segment_sec": (
                round(segment_sec, 3) if segment_sec is not None else None
            ),
            "drop_transition_frames": drop_flag,
            "merge_short_sec": round(merge_short_sec, 3),
            "merge_max_sec": round(merge_max_sec, 3),
            "merge_operations": merge_operations,
            "video_path": resolved_video_path or "",
            "video_name": video_name,
            "seed": seed,
            "fps": round(fps_value, 6),
            "threshold": threshold,
            "source": "video",
            "mode": mode,
            "images_decoded": False,
            "device": device_label,
            "weights": weights_label,
            "csv_path": csv_path,
            "shots": [
                {
                    "shot_id": row["shot_id"],
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "frame_count": row["frame_count"],
                    "merged_shots": int(row.get("merged_shots", 1)),
                    "start_time": round(row["start_time"], 3),
                    "end_time": round(row["end_time"], 3),
                    "duration": round(row["duration"], 3),
                    "boundary_score": (
                        None
                        if row["boundary_score"] is None
                        else round(row["boundary_score"], 4)
                    ),
                }
                for row in rows
            ],
        }
        info_json = json.dumps(info, ensure_ascii=False, indent=2)

        if kind == "transnetv2":
            summary = (
                "TransNetV2: {} shots, {} frames @ {:.3f} fps, "
                "device={}, threshold={:g}".format(
                    len(rows), total_frames, fps_value, device_label, threshold
                )
            )
        else:
            summary = (
                "Fixed duration: {} shots, {} frames @ {:.3f} fps, "
                "segment_sec={:g}".format(
                    len(rows), total_frames, fps_value, segment_sec
                )
            )
        if csv_path:
            summary += "\nCSV: {}".format(csv_path)
        if streaming:
            summary += (
                "\nStreaming VIDEO: decoded once at 48x27; videos are lazily trimmed."
                if kind == "transnetv2"
                else "\nStreaming VIDEO: frames scanned for timestamps only; "
                "videos are lazily trimmed."
            )
        else:
            summary += "\nVIDEO components were already in memory; videos are sliced lazily."
        print("[XiNodes Shot Split] {}".format(summary))
        for row in rows[:10]:
            print(
                "  shot {:>3}: frames {}-{} ({}f, {:.2f}-{:.2f}s)".format(
                    row["shot_id"],
                    row["start_frame"],
                    row["end_frame"],
                    row["frame_count"],
                    row["start_time"],
                    row["end_time"],
                )
            )
        if len(rows) > 10:
            print("  ... {} more shots".format(len(rows) - 10))

        return io.NodeOutput(
            videos,
            info_json,
            len(rows),
            video_name,
            ui=ui.PreviewText(summary),
        )


NODES = [XiTransNetV2ShotSplitVideo]
