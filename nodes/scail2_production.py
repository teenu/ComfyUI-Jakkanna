from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile

import av
import cv2
import torch

import comfy.model_management
import comfy.model_sampling
import comfy.samplers
import folder_paths
from comfy.ldm.sam3.tracker import unpack_masks

from .scail2_flow_unipc import FlowUniPCMultistepScheduler


UPSTREAM_COMMIT = "5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5"

_VIDEO_FORMATS_DIR = str(Path(__file__).resolve().parents[1] / "video_formats")
_vhs_entry = folder_paths.folder_names_and_paths.get("VHS_video_formats")
if _vhs_entry is None:
    folder_paths.folder_names_and_paths["VHS_video_formats"] = ([_VIDEO_FORMATS_DIR], {".json"})
elif _VIDEO_FORMATS_DIR not in _vhs_entry[0]:
    folder_paths.folder_names_and_paths["VHS_video_formats"] = (
        list(_vhs_entry[0]) + [_VIDEO_FORMATS_DIR],
        set(_vhs_entry[1]) | {".json"},
    )


def _hash_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_HASH_CACHE_PATH = Path(folder_paths.get_temp_directory()) / "jakkanna_scail2_model_hash_cache.json"


def _hash_file_cached(path):
    path = os.path.realpath(path)
    stat = os.stat(path)
    try:
        cache = json.loads(_HASH_CACHE_PATH.read_text())
    except (OSError, ValueError):
        cache = {}
    entry = cache.get(path)
    if (
        isinstance(entry, dict)
        and entry.get("size") == stat.st_size
        and entry.get("mtime_ns") == stat.st_mtime_ns
    ):
        return entry["sha256"]
    digest = _hash_file(path)
    cache[path] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=_HASH_CACHE_PATH.parent,
        prefix=f".{_HASH_CACHE_PATH.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(cache, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    try:
        temp_path.replace(_HASH_CACHE_PATH)
    finally:
        temp_path.unlink(missing_ok=True)
    return digest


def _hash_stream(source):
    if isinstance(source, (str, os.PathLike)):
        return _hash_file(source)
    if not isinstance(source, io.BytesIO):
        raise TypeError(f"Unsupported video source type: {type(source).__name__}")
    position = source.tell()
    source.seek(0)
    digest = hashlib.sha256()
    for chunk in iter(lambda: source.read(16 * 1024 * 1024), b""):
        digest.update(chunk)
    source.seek(position)
    return digest.hexdigest()


def _hash_tensor(tensor, binary=False):
    digest = hashlib.sha256()
    frames = tensor if tensor.ndim > 3 else tensor.unsqueeze(0)
    for frame in frames:
        frame = frame.detach().to(device="cpu")
        if binary:
            data = (frame > 0.5).to(torch.uint8).contiguous()
        else:
            data = frame.float().clamp(0, 1).mul(65535).round().to(torch.uint16).contiguous()
        digest.update(data.numpy().tobytes())
    return digest.hexdigest()


def _json(data):
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _comfyui_commit():
    root = Path(folder_paths.__file__).resolve().parent
    try:
        head = (root / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            head = (root / ".git" / head[5:]).read_text(encoding="utf-8").strip()
        if len(head) == 40 and all(character in "0123456789abcdef" for character in head.lower()):
            return head
    except OSError:
        pass
    return "unknown"


class JakkannaSCAIL2ProductionInputValidate:
    HDR_TRANSFERS = {16, 18}
    CANONICAL_RATE = Fraction(16)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_video": ("VIDEO",),
                "reference_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("width", "height", "length", "fps", "input_report")
    FUNCTION = "validate"
    CATEGORY = "Jakkanna/SCAIL-2"

    def validate(self, source_video, reference_image):
        source = source_video.get_stream_source()
        if isinstance(source, io.BytesIO):
            source.seek(0)

        with av.open(source, mode="r") as container:
            if len(container.streams.video) != 1:
                raise ValueError(f"Source must contain exactly one video stream, got {len(container.streams.video)}")
            stream = container.streams.video[0]
            rate = Fraction(stream.average_rate) if stream.average_rate else None
            if rate is None:
                raise ValueError("Source has no detectable frame rate")
            if rate != self.CANONICAL_RATE:
                raise ValueError(
                    f"SCAIL-2 production input must be exactly {self.CANONICAL_RATE} fps — the canonical "
                    f"training rate; prepare the source at 16 fps before running this workflow — got {rate}"
                )

            pts = []
            rotations = set()
            decoded_hash = hashlib.sha256()
            color_transfers = set()
            color_primaries = set()
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("Source contains a frame without a presentation timestamp")
                pts.append(frame.pts)
                rotations.add(frame.rotation)
                color_transfers.add(frame.color_trc)
                color_primaries.add(frame.color_primaries)
                decoded_hash.update(frame.to_ndarray(format="rgb24").tobytes())

            width, height = stream.width, stream.height
            time_base = Fraction(stream.time_base)
            stream_metadata = dict(stream.metadata)
            pixel_format = stream.codec_context.pix_fmt
            sample_aspect_ratio = stream.codec_context.sample_aspect_ratio

        if isinstance(source, io.BytesIO):
            source.seek(0)

        if len(pts) != 81:
            raise ValueError(f"SCAIL-2 production input must contain exactly 81 frames, got {len(pts)}")
        if (width, height) not in {(704, 1280), (1280, 704)}:
            raise ValueError(
                f"SCAIL-2 production input must be 704x1280 or 1280x704, got {width}x{height}"
            )
        if any(rotation != 0 for rotation in rotations) or stream_metadata.get("rotate", "0") != "0":
            raise ValueError("Apply video rotation physically before SCAIL-2; rotation metadata is not accepted")
        if sample_aspect_ratio is not None and Fraction(sample_aspect_ratio) != 1:
            raise ValueError(f"Source must use square pixels, got sample aspect ratio {sample_aspect_ratio}")
        if reference_image.shape[0] != 1:
            raise ValueError(f"Replacement reference must contain exactly one image, got {reference_image.shape[0]}")
        if (reference_image.shape[2], reference_image.shape[1]) != (width, height):
            raise ValueError(
                "Replacement reference must match the source canvas exactly: "
                f"source={width}x{height}, reference={reference_image.shape[2]}x{reference_image.shape[1]}"
            )

        nominal_delta = Fraction(1, 1) / rate
        deltas = [(pts[i + 1] - pts[i]) * time_base for i in range(len(pts) - 1)]
        tolerance = max(float(time_base) * 1.1, 1e-6)
        max_timing_error = max(abs(float(delta - nominal_delta)) for delta in deltas)
        if max_timing_error > tolerance:
            raise ValueError(
                "Source must be constant-frame-rate: "
                f"maximum frame-interval error is {max_timing_error:.8f}s"
            )
        if self.HDR_TRANSFERS.intersection(color_transfers):
            raise ValueError("HDR PQ/HLG source video is not accepted; convert it to SDR BT.709 first")

        color_warning = None
        if color_transfers == {2} or color_primaries == {2}:
            color_warning = "Color metadata is unspecified; no HDR transfer was detected"

        report = {
            "source_sha256": _hash_stream(source_video.get_stream_source()),
            "decoded_rgb8_sha256": decoded_hash.hexdigest(),
            "width": width,
            "height": height,
            "frames": len(pts),
            "fps": f"{rate.numerator}/{rate.denominator}",
            "canonical_fps_enforced": int(self.CANONICAL_RATE),
            "duration_seconds": float(len(pts) / rate),
            "constant_frame_rate": True,
            "maximum_frame_interval_error_seconds": max_timing_error,
            "pixel_format": pixel_format,
            "bit_depth": source_video.get_bit_depth(),
            "color_transfers": sorted(color_transfers),
            "color_primaries": sorted(color_primaries),
            "color_warning": color_warning,
        }
        return width, height, len(pts), float(rate), _json(report)


class JakkannaSCAIL2SingleSubjectMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "masks": ("MASK",),
                "minimum_area_percent": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "maximum_area_percent": ("FLOAT", {"default": 95.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("mask", "mask_report")
    FUNCTION = "validate"
    CATEGORY = "Jakkanna/SCAIL-2"

    def validate(self, masks, minimum_area_percent, maximum_area_percent):
        if masks.ndim == 2:
            masks = masks.unsqueeze(0)
        if masks.shape[0] != 1:
            raise ValueError(f"SAM 3.1 must detect exactly one subject, got {masks.shape[0]}")
        binary = (masks > 0.5).float()
        area_percent = float(binary[0].mean().item() * 100)
        if not minimum_area_percent <= area_percent <= maximum_area_percent:
            raise ValueError(
                f"SAM 3.1 subject mask covers {area_percent:.3f}% of the image; "
                f"expected {minimum_area_percent:.3f}% to {maximum_area_percent:.3f}%"
            )
        report = {
            "objects": 1,
            "area_percent": area_percent,
            "binary_mask_sha256": _hash_tensor(binary, binary=True),
        }
        return binary, _json(report)



class JakkannaSCAIL2ReferencePNG16Save:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filename_prefix": ("STRING", {"default": "scail2_prep/reference/SCAIL2_reference_704x1280"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("save_report",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Jakkanna/SCAIL-2"

    def save(self, images, filename_prefix):
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            images.shape[2],
            images.shape[1],
        )
        results = []
        files = []
        for index in range(images.shape[0]):
            # Same 16-bit quantization as _hash_tensor, so the saved pixels hash to
            # output_rgb16_sha256 and the production loader's decode reproduces it exactly.
            data = images[index].detach().cpu().float().clamp(0, 1).mul(65535).round().to(torch.uint16).numpy()
            file = f"{filename}_{counter + index:05}_.png"
            path = os.path.join(full_output_folder, file)
            if not cv2.imwrite(path, data[:, :, :3][:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 4]):
                raise RuntimeError(f"Failed to write 16-bit PNG: {path}")
            files.append({"file": file, "file_sha256": _hash_file(path)})
            results.append({"filename": file, "subfolder": subfolder, "type": "output"})
        report = {
            "files": files,
            "bit_depth": 16,
            "width": images.shape[2],
            "height": images.shape[1],
            "output_rgb16_sha256": _hash_tensor(images),
        }
        return {"ui": {"images": results}, "result": (_json(report),)}



class JakkannaSCAIL2TrackValidate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "track_data": ("SAM3_TRACK_DATA",),
                "reference_mask": ("MASK",),
                "expected_frames": ("INT", {"forceInput": True}),
                "maximum_area_step_ratio": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 100.0, "step": 0.1}),
                "maximum_centroid_jump": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "minimum_adjacent_iou": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("object_indices", "track_report")
    FUNCTION = "validate"
    CATEGORY = "Jakkanna/SCAIL-2"

    def validate(
        self,
        track_data,
        reference_mask,
        expected_frames,
        maximum_area_step_ratio,
        maximum_centroid_jump,
        minimum_adjacent_iou,
    ):
        packed = track_data.get("packed_masks")
        if packed is None:
            raise ValueError("SAM 3.1 produced no source track")
        if packed.shape[0] != expected_frames:
            raise ValueError(f"SAM 3.1 track has {packed.shape[0]} frames; expected {expected_frames}")
        if packed.shape[1] != 1:
            raise ValueError(f"SAM 3.1 track must contain exactly one subject, got {packed.shape[1]}")

        masks = unpack_masks(packed)[:, 0].bool()
        areas = masks.sum(dim=(-1, -2)).float()
        if (areas == 0).any():
            empty = (areas == 0).nonzero(as_tuple=True)[0].tolist()
            raise ValueError(f"SAM 3.1 source track is empty on frames {empty}")

        smaller = torch.minimum(areas[:-1], areas[1:]).clamp(min=1)
        larger = torch.maximum(areas[:-1], areas[1:])
        area_step_ratio = larger / smaller

        height, width = masks.shape[-2:]
        grid_y = torch.arange(height, device=masks.device, dtype=torch.float32).view(1, height, 1)
        grid_x = torch.arange(width, device=masks.device, dtype=torch.float32).view(1, 1, width)
        centroids_y = (masks * grid_y).sum(dim=(-1, -2)) / areas
        centroids_x = (masks * grid_x).sum(dim=(-1, -2)) / areas
        dx = (centroids_x[1:] - centroids_x[:-1]) / width
        dy = (centroids_y[1:] - centroids_y[:-1]) / height
        centroid_jump = torch.sqrt(dx.square() + dy.square())

        intersections = (masks[:-1] & masks[1:]).sum(dim=(-1, -2)).float()
        unions = (masks[:-1] | masks[1:]).sum(dim=(-1, -2)).float().clamp(min=1)
        adjacent_iou = intersections / unions

        max_area_step = float(area_step_ratio.max().item())
        max_jump = float(centroid_jump.max().item())
        min_iou = float(adjacent_iou.min().item())
        if max_area_step > maximum_area_step_ratio:
            raise ValueError(
                f"SAM 3.1 track area changes by {max_area_step:.3f}x between adjacent frames; "
                f"limit is {maximum_area_step_ratio:.3f}x"
            )
        if max_jump > maximum_centroid_jump:
            raise ValueError(
                f"SAM 3.1 track centroid jumps {max_jump:.3f} frame diagonals; "
                f"limit is {maximum_centroid_jump:.3f}"
            )
        if min_iou < minimum_adjacent_iou:
            raise ValueError(
                f"SAM 3.1 minimum adjacent-frame IoU is {min_iou:.4f}; "
                f"minimum is {minimum_adjacent_iou:.4f}"
            )
        if reference_mask.shape[0] != 1 or not (reference_mask > 0.5).any():
            raise ValueError("Validated reference mask must contain exactly one nonempty subject")

        report = {
            "frames": masks.shape[0],
            "objects": 1,
            "minimum_area_percent": float((areas.min() / (height * width) * 100).item()),
            "maximum_area_percent": float((areas.max() / (height * width) * 100).item()),
            "maximum_adjacent_area_ratio": max_area_step,
            "maximum_centroid_jump": max_jump,
            "minimum_adjacent_iou": min_iou,
            "track_mask_sha256": _hash_tensor(masks, binary=True),
        }
        return "0", _json(report)


def _scheduler(steps, shift, device):
    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=1000,
        shift=1,
        use_dynamic_shifting=False,
    )
    scheduler.set_timesteps(steps, device=device, shift=shift)
    return scheduler


def sample_flow_unipc(model, noise, sigmas, extra_args=None, callback=None, disable=None, steps=40, shift=3.0):
    extra_args = {} if extra_args is None else extra_args
    scheduler = _scheduler(steps, shift, noise.device)
    expected_sigmas = scheduler.sigmas.to(device=sigmas.device, dtype=sigmas.dtype)
    if sigmas.shape != expected_sigmas.shape or not torch.equal(sigmas, expected_sigmas):
        raise ValueError("Flow-UniPC sampler requires the matching upstream sigma schedule")

    denoise_mask = extra_args.get("denoise_mask")

    def apply_history(x):
        if denoise_mask is None:
            return x
        return x * denoise_mask + model.latent_image * (1.0 - denoise_mask)

    x = apply_history(model.noise.clone())
    for i, timestep in enumerate(scheduler.timesteps):
        scheduler_sigma = scheduler.sigmas[i].to(device=x.device, dtype=x.dtype)
        model_sigma = (timestep.to(device=x.device, dtype=x.dtype) / scheduler.config.num_train_timesteps).expand(x.shape[0])
        denoised = model(x, model_sigma, **extra_args)
        model_output = (x - denoised) / model_sigma.view((-1,) + (1,) * (x.ndim - 1))
        upstream_denoised = x - scheduler_sigma * model_output
        x = apply_history(scheduler.step(model_output, timestep, x, return_dict=False)[0])
        if callback is not None:
            callback({"x": x, "i": i, "sigma": scheduler_sigma, "sigma_hat": scheduler_sigma, "denoised": upstream_denoised})
    return x


class JakkannaSCAIL2FlowUniPC:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "shift": ("FLOAT", {"default": 3.0, "min": 0.01, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("SAMPLER", "SIGMAS")
    RETURN_NAMES = ("sampler", "sigmas")
    FUNCTION = "build"
    CATEGORY = "Jakkanna/SCAIL-2"

    def build(self, steps, shift):
        scheduler = _scheduler(steps, shift, "cpu")
        sampler = comfy.samplers.KSAMPLER(sample_flow_unipc, {"steps": steps, "shift": shift})
        return sampler, scheduler.sigmas.clone()



class UpstreamNoise:
    def __init__(self, noise, seed):
        self.noise = noise
        self.seed = seed

    def generate_noise(self, input_latent):
        expected = input_latent["samples"].shape
        if self.noise.shape != expected:
            raise ValueError(
                f"Upstream noise shape {tuple(self.noise.shape)} does not match latent shape {tuple(expected)}"
            )
        return self.noise


class JakkannaSCAIL2UpstreamNoise:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "noise_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("NOISE", "STRING")
    RETURN_NAMES = ("noise", "noise_report")
    FUNCTION = "generate"
    CATEGORY = "Jakkanna/SCAIL-2"

    def generate(self, latent, noise_seed):
        shape = latent["samples"].shape
        device = comfy.model_management.get_torch_device()
        generator = torch.Generator(device=device)
        generator.manual_seed(noise_seed)
        noise = torch.randn(shape, dtype=torch.float32, generator=generator, device=device)
        digest = hashlib.sha256(noise.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        report = {
            "seed": noise_seed,
            "shape": list(shape),
            "dtype": "float32",
            "generator_device": str(device),
            "noise_sha256": digest,
        }
        return UpstreamNoise(noise, noise_seed), _json(report)



def _simple_flow_sigmas(shift, steps):
    """The sigma schedule BasicScheduler("simple") produces for a discrete-flow model at `shift`."""
    model_sampling = comfy.model_sampling.ModelSamplingDiscreteFlow()
    model_sampling.set_parameters(shift=shift)
    return comfy.samplers.simple_scheduler(model_sampling, steps)


class JakkannaSCAIL2ProductionManifest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "output_images": ("IMAGE",),
                "pose_video_mask": ("IMAGE",),
                "reference_image_mask": ("IMAGE",),
                "input_report": ("STRING", {"forceInput": True}),
                "reference_report": ("STRING", {"forceInput": True}),
                "source_initial_mask_report": ("STRING", {"forceInput": True}),
                "reference_mask_report": ("STRING", {"forceInput": True}),
                "track_report": ("STRING", {"forceInput": True}),
                "noise_report": ("STRING", {"forceInput": True}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "seed": ("INT", {"forceInput": True}),
                "diffusion_model": (folder_paths.get_filename_list("diffusion_models"),),
                "dpo_lora": (folder_paths.get_filename_list("loras"),),
                "sam_checkpoint": (folder_paths.get_filename_list("checkpoints"),),
                "text_encoder": (folder_paths.get_filename_list("text_encoders"),),
                "clip_vision": (folder_paths.get_filename_list("clip_vision"),),
                "vae": (folder_paths.get_filename_list("vae"),),
                "attention_backend": (["PyTorch SDPA", "FlashAttention", "SageAttention", "Other"],),
                "hash_model_files": ("BOOLEAN", {"default": True}),
                "filename_prefix": ("STRING", {"default": "scail2_production/manifests/SCAIL2_replacement_production"}),
            },
            "optional": {
                "acceleration_lora": (["None"] + folder_paths.get_filename_list("loras"),),
                "acceleration_lora_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "sampler_name": ("STRING", {"default": "Flow-UniPC"}),
                "scheduler_name": ("STRING", {"default": "upstream Flow-UniPC"}),
                "steps": ("INT", {"default": 40, "min": 1, "max": 100}),
                "flow_shift": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "effective_flow_shift": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.01, "tooltip": "Shift actually realized in the sigma schedule; 0 = same as flow_shift. When sigmas is connected, the run fails unless the wired schedule is the Euler/simple flow schedule at this shift."}),
                "mode": (["replacement", "animation"], {"default": "replacement"}),
                "sigmas": ("SIGMAS", {"tooltip": "Wire the sampled sigma schedule to record it verbatim and validate effective_flow_shift against it."}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("manifest_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "Jakkanna/SCAIL-2"

    def save(
        self,
        output_images,
        pose_video_mask,
        reference_image_mask,
        input_report,
        reference_report,
        source_initial_mask_report,
        reference_mask_report,
        track_report,
        noise_report,
        positive_prompt,
        negative_prompt,
        seed,
        diffusion_model,
        dpo_lora,
        sam_checkpoint,
        text_encoder,
        clip_vision,
        vae,
        attention_backend,
        hash_model_files,
        filename_prefix,
        acceleration_lora="None",
        acceleration_lora_strength=0.0,
        sampler_name="Flow-UniPC",
        scheduler_name="upstream Flow-UniPC",
        steps=40,
        flow_shift=3.0,
        cfg=5.0,
        effective_flow_shift=0.0,
        mode="replacement",
        sigmas=None,
        prompt=None,
        extra_pnginfo=None,
    ):
        effective_shift = effective_flow_shift if effective_flow_shift > 0 else flow_shift
        sigma_schedule = None
        if sigmas is not None:
            expected = _simple_flow_sigmas(effective_shift, steps)
            wired = sigmas.detach().cpu().float()
            if wired.shape != expected.shape or not torch.equal(wired, expected):
                raise ValueError(
                    "Wired sigma schedule is not the Euler/simple flow schedule at "
                    f"effective_flow_shift={effective_shift:g} with {steps} steps: "
                    f"got {[round(float(s), 6) for s in wired]}, "
                    f"expected {[round(float(s), 6) for s in expected]}"
                )
            sigma_schedule = [float(s) for s in wired]
        model_files = {
            "diffusion_model": ("diffusion_models", diffusion_model),
            "dpo_lora": ("loras", dpo_lora),
            "sam_checkpoint": ("checkpoints", sam_checkpoint),
            "text_encoder": ("text_encoders", text_encoder),
            "clip_vision": ("clip_vision", clip_vision),
            "vae": ("vae", vae),
        }
        if acceleration_lora != "None":
            model_files["acceleration_lora"] = ("loras", acceleration_lora)
        artifacts = {}
        for key, (category, name) in model_files.items():
            path = folder_paths.get_full_path(category, name)
            if path is None:
                raise ValueError(f"Could not resolve {category} artifact: {name}")
            artifacts[key] = {
                "name": name,
                "size_bytes": os.path.getsize(path),
                "sha256": _hash_file_cached(path) if hash_model_files else None,
            }

        workflow = (extra_pnginfo or {}).get("workflow")
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "pipeline": f"SCAIL-2 single-subject 81-frame character {mode}",
            "upstream_scail2_commit": UPSTREAM_COMMIT,
            "comfyui_commit": _comfyui_commit(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "attention_backend": attention_backend,
            "settings": {
                "resolution": "704x1280 or 1280x704",
                "frames": 81,
                "sampler": sampler_name,
                "scheduler": scheduler_name,
                "steps": steps,
                "flow_shift": flow_shift,
                "effective_flow_shift": effective_shift,
                "sigma_schedule": sigma_schedule,
                "cfg": cfg,
                "dpo_lora_strength": 1.0,
                "acceleration_lora": None if acceleration_lora == "None" else {
                    "name": acceleration_lora,
                    "strength": acceleration_lora_strength,
                },
                "replacement_mode": mode == "replacement",
                "pose_strength": 1.0,
                "pose_start": 0.0,
                "pose_end": 1.0,
                "clip_vision_crop": "none",
                "seed": seed,
            },
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "input": json.loads(input_report),
            "reference": json.loads(reference_report),
            "source_initial_mask": json.loads(source_initial_mask_report),
            "reference_mask": json.loads(reference_mask_report),
            "source_track": json.loads(track_report),
            "noise": json.loads(noise_report),
            "pose_video_mask_rgb16_sha256": _hash_tensor(pose_video_mask),
            "reference_image_mask_rgb16_sha256": _hash_tensor(reference_image_mask),
            "decoded_output_rgb16_sha256": _hash_tensor(output_images),
            "decoded_output_frames": output_images.shape[0],
            "artifacts": artifacts,
            "prompt_graph_sha256": hashlib.sha256(_json(prompt or {}).encode()).hexdigest(),
            "workflow_sha256": hashlib.sha256(_json(workflow or {}).encode()).hexdigest(),
            "production_node_sha256": _hash_file(__file__),
            "flow_unipc_node_sha256": _hash_file(Path(__file__).with_name("scail2_flow_unipc.py")),
        }

        full_output_folder, filename, counter, _, _ = folder_paths.get_save_image_path(
            filename_prefix,
            folder_paths.get_output_directory(),
            output_images.shape[2],
            output_images.shape[1],
        )
        output_path = os.path.join(full_output_folder, f"{filename}_{counter:05}_.json")
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        return {"ui": {"text": [output_path]}, "result": (output_path,)}

NODE_CLASS_MAPPINGS = {
    "JakkannaSCAIL2FlowUniPC": JakkannaSCAIL2FlowUniPC,
    "JakkannaSCAIL2ProductionInputValidate": JakkannaSCAIL2ProductionInputValidate,
    "JakkannaSCAIL2SingleSubjectMask": JakkannaSCAIL2SingleSubjectMask,
    "JakkannaSCAIL2ReferencePNG16Save": JakkannaSCAIL2ReferencePNG16Save,
    "JakkannaSCAIL2TrackValidate": JakkannaSCAIL2TrackValidate,
    "JakkannaSCAIL2UpstreamNoise": JakkannaSCAIL2UpstreamNoise,
    "JakkannaSCAIL2ProductionManifest": JakkannaSCAIL2ProductionManifest,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "JakkannaSCAIL2FlowUniPC": "Jakkanna SCAIL-2 Flow-UniPC",
    "JakkannaSCAIL2ProductionInputValidate": "Jakkanna SCAIL-2 Validate Input",
    "JakkannaSCAIL2SingleSubjectMask": "Jakkanna SCAIL-2 Validate Subject Mask",
    "JakkannaSCAIL2ReferencePNG16Save": "Jakkanna SCAIL-2 Save 16-bit Reference",
    "JakkannaSCAIL2TrackValidate": "Jakkanna SCAIL-2 Validate Track",
    "JakkannaSCAIL2UpstreamNoise": "Jakkanna SCAIL-2 Upstream Noise",
    "JakkannaSCAIL2ProductionManifest": "Jakkanna SCAIL-2 Save Manifest",
}
