import os
import json


class XiSaveAudio:
    """Save an AUDIO input to the ComfyUI output directory.

    Input/Output follows ComfyUI's AUDIO type: a dict with
        "waveform": torch.Tensor of shape (batch, channels, samples)
        "sample_rate": int
    """

    # Opus (libopus) only accepts these sample rates.
    OPUS_RATES = [8000, 12000, 16000, 24000, 48000]

    # container_format, codec, supports_bitrate
    FORMATS = {
        "m4a": ("mp4", "aac", True),
        "wav": ("wav", "pcm_s16le", False),
        "flac": ("flac", "flac", False),
        "mp3": ("mp3", "libmp3lame", True),
        "ogg": ("ogg", "libvorbis", True),
        "opus": ("opus", "libopus", True),
    }

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING", {"default": "audio/ComfyUI"}),
                "format": (["m4a", "wav", "flac", "mp3", "ogg", "opus"], {"default": "m4a"}),
                "sample_rate": ("INT", {"default": 44100, "min": 4000, "max": 384000, "step": 100}),
                "bitrate_kbps": ("INT", {"default": 128, "min": 8, "max": 512, "step": 1}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "save_audio"
    OUTPUT_NODE = True
    CATEGORY = "XiNodes"
    DESCRIPTION = "保存 AUDIO 到 ComfyUI output 目录，可自定义文件名前缀、格式、采样率、码率。支持 m4a/wav/flac/mp3/ogg/opus，并透传输入 AUDIO。"

    # ------------------------------------------------------------------
    def _normalize_sample_rate(self, sample_rate, fmt):
        """Clamp/repair sample rate for codecs that have strict limits."""
        sample_rate = int(sample_rate)
        if fmt == "opus":
            if sample_rate > 48000:
                return 48000
            if sample_rate in self.OPUS_RATES:
                return sample_rate
            # Same behaviour as ComfyUI core: pick the next supported rate.
            for rate in sorted(self.OPUS_RATES):
                if rate > sample_rate:
                    return rate
            return 48000
        if fmt == "mp3":
            return max(8000, min(48000, sample_rate))
        if fmt == "m4a":
            return max(8000, min(96000, sample_rate))
        if fmt == "ogg":
            return max(8000, min(192000, sample_rate))
        return sample_rate

    def _resample(self, waveform, orig_freq, new_freq):
        """Resample a (channels, samples) float tensor.

        Prefer torchaudio (ComfyUI standard). If it is missing, fall back to
        numpy linear interpolation so the node can still run.
        """
        try:
            import torchaudio
            return torchaudio.functional.resample(waveform, orig_freq, new_freq)
        except ImportError:
            pass

        import numpy as np
        import torch

        arr = waveform.detach().cpu().float().numpy()
        new_len = max(1, int(round(arr.shape[-1] * new_freq / float(orig_freq))))
        x_old = np.linspace(0.0, 1.0, num=arr.shape[-1], dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, num=new_len, dtype=np.float64)
        resampled = np.stack(
            [np.interp(x_new, x_old, arr[ch]) for ch in range(arr.shape[0])],
            axis=0,
        ).astype(np.float32)
        print(f"[XiNodes SaveAudio Warning] torchaudio not available; used numpy linear interpolation for resampling {orig_freq}->{new_freq}.")
        return torch.from_numpy(resampled)

    def _resolve_output_path(self, filename_prefix, ext):
        """Return (full_output_folder, filename, counter, subfolder).

        Uses ComfyUI's folder_paths when available (standard behaviour).
        """
        filename_prefix = filename_prefix.strip() or "audio/ComfyUI"

        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
            full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
                filename_prefix, output_dir
            )
            os.makedirs(full_output_folder, exist_ok=True)
            return full_output_folder, filename, counter, subfolder
        except ImportError:
            # Fallback for running outside a full ComfyUI install.
            output_dir = os.path.abspath("./output")
            subfolder = os.path.dirname(os.path.normpath(filename_prefix))
            filename = os.path.basename(os.path.normpath(filename_prefix)) or "ComfyUI"
            if subfolder and not os.path.isabs(subfolder):
                full_output_folder = os.path.join(output_dir, subfolder)
            else:
                full_output_folder = output_dir
            os.makedirs(full_output_folder, exist_ok=True)

            counter = 1
            while True:
                candidate = f"{filename}_{counter:05}.{ext}"
                if not os.path.exists(os.path.join(full_output_folder, candidate)):
                    break
                counter += 1
            return full_output_folder, filename, counter, subfolder

    # ------------------------------------------------------------------
    def save_audio(
        self,
        audio,
        filename_prefix="audio/ComfyUI",
        format="m4a",
        sample_rate=44100,
        bitrate_kbps=128,
        prompt=None,
        extra_pnginfo=None,
    ):
        import torch

        # ---- 1. Validate the ComfyUI AUDIO dict ----------------------
        if not isinstance(audio, dict) or "waveform" not in audio or "sample_rate" not in audio:
            raise ValueError(
                "[XiNodes SaveAudio] input must be a ComfyUI AUDIO dict: {'waveform': torch.Tensor, 'sample_rate': int}. "
                "If the source video has no audio track, the upstream node may pass None."
            )

        waveform = audio["waveform"]
        if not isinstance(waveform, torch.Tensor):
            raise ValueError("[XiNodes SaveAudio] audio['waveform'] must be a torch.Tensor.")

        if waveform.numel() == 0:
            raise ValueError("[XiNodes SaveAudio] audio['waveform'] is empty.")

        input_sample_rate = int(audio["sample_rate"])
        target_sample_rate = int(sample_rate)
        fmt = str(format).strip().lower()
        bitrate = int(bitrate_kbps) * 1000

        if fmt not in self.FORMATS:
            raise ValueError(
                f"[XiNodes SaveAudio] unsupported format '{fmt}'. Supported: {list(self.FORMATS.keys())}"
            )

        container_format, codec, supports_bitrate = self.FORMATS[fmt]
        if not supports_bitrate:
            print(f"[XiNodes SaveAudio] format '{fmt}' is lossless/uncompressed; bitrate_kbps is ignored.")

        # ---- 2. Normalize shape to (batch, channels, samples) --------
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.dim() == 2:
            # A single audio clip of shape (channels, samples).
            waveform = waveform.unsqueeze(0)
        elif waveform.dim() != 3:
            raise ValueError(
                f"[XiNodes SaveAudio] waveform must have shape (batch, channels, samples). Got {tuple(waveform.shape)}."
            )

        batch_size, channels, samples = waveform.shape
        if channels not in (1, 2):
            raise ValueError(
                f"[XiNodes SaveAudio] ComfyUI audio is expected to be mono (1) or stereo (2); got {channels} channels."
            )
        layout = "mono" if channels == 1 else "stereo"

        # ---- 3. Repair/resample sample rate --------------------------
        target_sample_rate = self._normalize_sample_rate(target_sample_rate, fmt)
        if target_sample_rate != int(sample_rate):
            print(
                f"[XiNodes SaveAudio] sample_rate {int(sample_rate)} is not valid for '{fmt}'; using {target_sample_rate} instead."
            )

        # ---- 4. Resolve output path -----------------------------------
        ext = "m4a" if fmt == "m4a" else fmt
        full_output_folder, filename, counter, subfolder = self._resolve_output_path(filename_prefix, ext)

        # ---- 5. Import PyAV -------------------------------------------
        try:
            import av
        except ImportError as e:
            raise RuntimeError(
                "[XiNodes SaveAudio] This node requires PyAV. Install it with: pip install av"
            ) from e

        # ---- 6. Metadata (same keys as ComfyUI core SaveImage) --------
        metadata = {}
        if prompt is not None:
            metadata["prompt"] = json.dumps(prompt, ensure_ascii=False)
        if extra_pnginfo is not None:
            for key, value in extra_pnginfo.items():
                metadata[key] = json.dumps(value, ensure_ascii=False)

        # ---- 7. Save each batch item as its own file ------------------
        results = []
        for batch_number in range(batch_size):
            wav = waveform[batch_number].detach().cpu().float()

            if target_sample_rate != input_sample_rate:
                wav = self._resample(wav, input_sample_rate, target_sample_rate)

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}.{ext}"
            output_path = os.path.join(full_output_folder, file)

            output_container = None
            try:
                output_container = av.open(output_path, mode="w", format=container_format)

                # Metadata must be set before muxing starts.
                for key, value in metadata.items():
                    try:
                        output_container.metadata[key] = value
                    except Exception:
                        pass

                out_stream = output_container.add_stream(codec, rate=target_sample_rate, layout=layout)
                if supports_bitrate:
                    out_stream.bit_rate = bitrate

                # ComfyUI core uses this exact packed-frame layout for AUDIO tensors.
                frame = av.AudioFrame.from_ndarray(
                    wav.movedim(0, 1).reshape(1, -1).numpy(),
                    format="flt",
                    layout=layout,
                )
                frame.sample_rate = target_sample_rate
                frame.pts = 0

                output_container.mux(out_stream.encode(frame))
                output_container.mux(out_stream.encode(None))
            except Exception as e:
                raise RuntimeError(f"[XiNodes SaveAudio] failed to encode '{output_path}': {e}") from e
            finally:
                if output_container is not None:
                    try:
                        output_container.close()
                    except Exception:
                        pass

            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
            counter += 1

        return {
            "ui": {
                "audio": results,
            },
            "result": (audio,),
        }


NODE_CLASS_MAPPINGS = {
    "XiSaveAudio": XiSaveAudio,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "XiSaveAudio": "Save Audio",
}
