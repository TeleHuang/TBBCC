#!/usr/bin/env python3
"""Collect and compare canonical model-suite GPU/NPU artifacts.

The generated 175-case benchmark remains the broad compatibility suite. This
module handles the smaller canonical model suite used for layer-wise numerical
figures and task-level drift analysis.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# MindSpore 2.7.2 ships generated protobuf modules that are incompatible with
# protobuf 4+ C++ descriptors when Transformers is imported first.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import tbbcc  # noqa: E402


SCHEMA_REGISTRY = "tbbcc.model_zoo.registry.v1"
SCHEMA_SUITE = "tbbcc.model_suite.v1"
SCHEMA_ARTIFACT = "tbbcc.model_artifact.v1"
SCHEMA_MANIFEST = "tbbcc.model_artifact_manifest.v1"
SCHEMA_COMPARISON = "tbbcc.model_suite_comparison.v1"
TENSOR_KEY = "__tbbcc_model_tensor__"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise SystemExit(f"Expected JSON object in {path}")
    return data


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _model_by_id(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["model_id"]): item for item in registry.get("models", []) if isinstance(item, dict)}


def _selected_models(registry_path: Path, suite_path: Path) -> list[dict[str, Any]]:
    registry = _load_json(registry_path)
    suite = _load_json(suite_path)
    validate_registry_payload(registry, suite)
    models = _model_by_id(registry)
    return [models[str(model_id)] for model_id in suite["model_ids"]]


def _input_artifact_path(root: Path, model_id: str) -> Path:
    return root / model_id / "inputs" / "input.npy"


def _input_dir(root: Path, model_id: str) -> Path:
    return root / model_id / "inputs"


def _has_input_artifacts(root: Path, model: dict[str, Any]) -> bool:
    model_id = str(model["model_id"])
    directory = _input_dir(root, model_id)
    kind = _model_input_kind(model)
    if kind == "language_tokens":
        return (directory / "input_ids.npy").exists() and (directory / "attention_mask.npy").exists()
    if kind == "diffusion_latent":
        return (directory / "sample.npy").exists() and (directory / "timestep.npy").exists()
    return _input_artifact_path(root, model_id).exists()


def _manifest_passed_model_ids(root: Path) -> set[str]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return set()
    manifest = _load_json(manifest_path)
    out: set[str] = set()
    for item in manifest.get("cases") or []:
        if not isinstance(item, dict) or item.get("status") != "passed" or not item.get("model_id"):
            continue
        model_id = str(item["model_id"])
        if _artifact_path(root, model_id).exists():
            out.add(model_id)
    return out


def _model_input_kind(model: dict[str, Any]) -> str:
    return str(model.get("input_kind") or "image_tensor")


def validate_registry_payload(registry: dict[str, Any], suite: dict[str, Any] | None = None) -> list[str]:
    errors: list[str] = []
    if registry.get("schema_version") != SCHEMA_REGISTRY:
        errors.append(f"registry.schema_version must be {SCHEMA_REGISTRY}")
    models = registry.get("models")
    if not isinstance(models, list) or not models:
        errors.append("registry.models must be a non-empty list")
        return errors
    seen: set[str] = set()
    for index, model in enumerate(models):
        if not isinstance(model, dict):
            errors.append(f"models[{index}] must be an object")
            continue
        model_id = str(model.get("model_id") or "")
        if not model_id:
            errors.append(f"models[{index}].model_id is required")
        if model_id in seen:
            errors.append(f"duplicate model_id: {model_id}")
        seen.add(model_id)
        for field in ("display_name", "task", "architecture_family", "pretrained_weights", "metrics", "hooks", "figure_role"):
            if not model.get(field):
                errors.append(f"{model_id or index} missing {field}")
        weights = model.get("pretrained_weights") or {}
        if not isinstance(weights, dict) or not weights.get("source") or not weights.get("locator"):
            errors.append(f"{model_id or index} pretrained_weights must include source and locator")
        hooks = model.get("hooks") or {}
        if not isinstance(hooks, dict) or not hooks.get("activation_layers") or not hooks.get("gradient_layers"):
            errors.append(f"{model_id or index} hooks must include activation_layers and gradient_layers")
        if not isinstance(model.get("metrics"), list) or not model.get("metrics"):
            errors.append(f"{model_id or index} metrics must be a non-empty list")

    if suite is not None:
        if suite.get("schema_version") != SCHEMA_SUITE:
            errors.append(f"suite.schema_version must be {SCHEMA_SUITE}")
        model_ids = suite.get("model_ids")
        if not isinstance(model_ids, list) or not model_ids:
            errors.append("suite.model_ids must be a non-empty list")
        else:
            missing = [str(model_id) for model_id in model_ids if str(model_id) not in seen]
            if missing:
                errors.append(f"suite references unknown model_ids: {missing}")
    return errors


def _artifact_rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _summarize_tensor(value: Any, artifact_dir: Path, artifact_root: Path, name: str) -> dict[str, Any]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    artifact_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(value, torch.Tensor):
        arr = value.detach().float().cpu().numpy()
        dtype = str(value.dtype)
        device = str(value.device)
    else:
        arr = np.asarray(value)
        dtype = str(arr.dtype)
        device = None
    arr = np.ascontiguousarray(arr)
    safe = tbbcc._slug(name)
    path = artifact_dir / f"{safe}.npy"
    np.save(path, arr, allow_pickle=False)
    flat = arr.reshape(-1)
    finite = flat[np.isfinite(flat)] if np.issubdtype(arr.dtype, np.number) else np.array([])
    summary: dict[str, Any] = {
        TENSOR_KEY: True,
        "shape": [int(x) for x in arr.shape],
        "dtype": dtype,
        "device": device,
        "numel": int(arr.size),
        "sha256": hashlib.sha256(arr.view(np.uint8)).hexdigest(),
        "artifact_path": _artifact_rel(path, artifact_root),
        "artifact_format": "npy",
        "sample": flat[:8].tolist(),
    }
    if finite.size:
        summary.update(
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "mean": float(finite.mean()),
                "std": float(finite.std()),
            }
        )
    return summary


def _environment(role: str, device: str) -> dict[str, Any]:
    env: dict[str, Any] = {
        "role": role,
        "device_requested": device,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    try:
        import torch  # type: ignore

        env["torch"] = getattr(torch, "__version__", "present")
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            env["gpu_count"] = int(torch.cuda.device_count())
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        env["torch_error"] = repr(exc)
    return env


class _UNetBlock:
    pass


def _build_unet_small() -> Any:
    import torch  # type: ignore

    class DoubleConv(torch.nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
                torch.nn.BatchNorm2d(out_ch),
                torch.nn.ReLU(inplace=True),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class UNetSmall(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = DoubleConv(3, 16)
            self.pool1 = torch.nn.MaxPool2d(2)
            self.enc2 = DoubleConv(16, 32)
            self.pool2 = torch.nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(32, 64)
            self.up2 = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.dec2 = DoubleConv(96, 32)
            self.up1 = torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            self.dec1 = DoubleConv(48, 16)
            self.out = torch.nn.Conv2d(16, 2, 1)

        def forward(self, x: Any) -> Any:
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            b = self.bottleneck(self.pool2(e2))
            d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)

    return UNetSmall()


def _hf_local_files_only() -> bool | None:
    value = os.environ.get("TBBCC_HF_LOCAL_FILES_ONLY")
    if value is None:
        return None
    return value not in {"0", "false", "False"}


def _minimind_load_kwargs(role: str) -> dict[str, Any]:
    import torch  # type: ignore

    kwargs: dict[str, Any] = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
    }
    if role == "gpu-reference":
        kwargs.update(
            {
                "device_map": os.environ.get("TBBCC_LLM_DEVICE_MAP", "auto"),
                "low_cpu_mem_usage": True,
            }
        )
    else:
        # Accelerate's meta initialization is incompatible with the torch4ms
        # intercept path, which materializes model parameters through MindSpore.
        kwargs["low_cpu_mem_usage"] = False
    local_files_only = _hf_local_files_only()
    if local_files_only is not None:
        kwargs["local_files_only"] = local_files_only
    return kwargs


def _build_model(
    model: dict[str, Any], *, pretrained: bool, cache_dir: Path, role: str = "gpu-reference"
) -> tuple[Any, dict[str, Any]]:
    import torch  # type: ignore

    os.environ.setdefault("TORCH_HOME", str(cache_dir / "torch"))
    os.environ.setdefault("HF_HOME", str(cache_dir / "hf"))
    os.environ.setdefault("TIMM_HOME", str(cache_dir / "timm"))
    model_id = str(model["model_id"])
    checkpoint: dict[str, Any] = {
        "source": (model.get("pretrained_weights") or {}).get("source"),
        "locator": (model.get("pretrained_weights") or {}).get("locator"),
        "pretrained_requested": pretrained,
        "sha256": None,
    }
    if model_id == "resnet18_imagenet_224":
        from torchvision.models import ResNet18_Weights, resnet18  # type: ignore

        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        return resnet18(weights=weights).eval(), checkpoint
    if model_id == "mobilenetv2_imagenet_224":
        from torchvision.models import MobileNet_V2_Weights, mobilenet_v2  # type: ignore

        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        return mobilenet_v2(weights=weights).eval(), checkpoint
    if model_id == "vit_tiny_imagenet_224":
        import timm  # type: ignore

        return timm.create_model("vit_tiny_patch16_224", pretrained=pretrained).eval(), checkpoint
    if model_id == "squeezenet1_1_imagenet_224":
        from torchvision.models import SqueezeNet1_1_Weights, squeezenet1_1  # type: ignore

        weights = SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        return squeezenet1_1(weights=weights).eval(), checkpoint
    if model_id == "shufflenet_v2_x1_0_imagenet_224":
        from torchvision.models import ShuffleNet_V2_X1_0_Weights, shufflenet_v2_x1_0  # type: ignore

        weights = ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        return shufflenet_v2_x1_0(weights=weights).eval(), checkpoint
    if model_id == "efficientnet_b0_imagenet_224":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0  # type: ignore

        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        return efficientnet_b0(weights=weights).eval(), checkpoint
    if model_id == "vgg11_bn_imagenet_224":
        from torchvision.models import VGG11_BN_Weights, vgg11_bn  # type: ignore

        weights = VGG11_BN_Weights.IMAGENET1K_V1 if pretrained else None
        return vgg11_bn(weights=weights).eval(), checkpoint
    if model_id == "bert_tiny_uncased_seq128":
        from transformers import AutoModel  # type: ignore

        locator = str(checkpoint["locator"])
        return AutoModel.from_pretrained(locator) if pretrained else AutoModel.from_config(_bert_tiny_config()), checkpoint
    if model_id == "minimind3o_moe_seq128":
        from transformers import AutoModelForCausalLM  # type: ignore

        locator = os.environ.get("TBBCC_LM_LOCATOR", str(checkpoint["locator"]))
        if not pretrained:
            raise SystemExit("minimind3o_moe_seq128 requires pretrained weights or a local checkpoint path.")
        load_kwargs = _minimind_load_kwargs(role)
        return AutoModelForCausalLM.from_pretrained(locator, **load_kwargs).eval(), checkpoint
    if model_id == "ddpm_cifar10_unet_32":
        if pretrained:
            from diffusers import UNet2DModel  # type: ignore

            locator = str(checkpoint["locator"])
            load_kwargs = {}
            local_files_only = _hf_local_files_only()
            if local_files_only is not None:
                load_kwargs["local_files_only"] = local_files_only
            return UNet2DModel.from_pretrained(locator, **load_kwargs).eval(), checkpoint
        return _build_tiny_ddpm_unet().eval(), checkpoint
    if model_id == "unet_small_biosample_256":
        net = _build_unet_small().eval()
        checkpoint_path = cache_dir / "unet_small_biosample_256" / "checkpoint.pt"
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location="cpu")
            net.load_state_dict(state.get("state_dict", state))
            checkpoint["local_path"] = str(checkpoint_path)
            checkpoint["sha256"] = _sha256_file(checkpoint_path)
        elif pretrained:
            checkpoint["warning"] = "No UNet checkpoint found; using deterministic initialized weights for bootstrap only."
        return net, checkpoint
    raise SystemExit(f"Unsupported model_id: {model_id}")


def _bert_tiny_config() -> Any:
    from transformers import BertConfig  # type: ignore

    return BertConfig(
        vocab_size=30522,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=512,
    )


def _build_tiny_ddpm_unet() -> Any:
    from diffusers import UNet2DModel  # type: ignore

    return UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    )


def _resolve_device(role: str, requested: str) -> str:
    if requested != "auto":
        return requested
    return "cuda" if role == "gpu-reference" else "cpu"


def _move_input_to_device(value: Any, device: str) -> Any:
    if isinstance(value, dict):
        return {key: _move_input_to_device(item, device) for key, item in value.items()}
    if hasattr(value, "to"):
        return value.to(device)
    return value


def _set_input_requires_grad(value: Any) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _set_input_requires_grad(item)
        return
    if hasattr(value, "is_floating_point") and value.is_floating_point():
        value.requires_grad_(True)


def _input_batch_size(value: Any) -> int:
    if isinstance(value, dict):
        for item in value.values():
            if hasattr(item, "shape") and len(item.shape) > 0:
                return int(item.shape[0])
        return 1
    if hasattr(value, "shape") and len(value.shape) > 0:
        return int(value.shape[0])
    return 1


def _forward_model(net: Any, value: Any, model: dict[str, Any]) -> Any:
    kind = _model_input_kind(model)
    if kind == "language_tokens":
        return net(**value, output_hidden_states=True, use_cache=False)
    if kind == "diffusion_latent":
        return net(value["sample"], value["timestep"])
    return net(value)


def _extract_output_tensor(output: Any) -> Any:
    if hasattr(output, "logits") and output.logits is not None:
        return output.logits
    if hasattr(output, "sample") and output.sample is not None:
        return output.sample
    if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
        return output.last_hidden_state
    if isinstance(output, (tuple, list)) and output:
        return output[0]
    return output


def _extra_activation_channels(output: Any, model: dict[str, Any]) -> dict[str, Any]:
    if not bool(model.get("capture_model_outputs")):
        return {}
    out: dict[str, Any] = {}
    if hasattr(output, "hidden_states") and output.hidden_states:
        hidden_states = list(output.hidden_states)
        indices = model.get("hidden_state_indices") or [0, len(hidden_states) // 2, len(hidden_states) - 1]
        for index in indices:
            idx = int(index)
            if -len(hidden_states) <= idx < len(hidden_states):
                out[f"hidden_state_{idx}"] = hidden_states[idx]
    return out


def _prepare_input(
    model: dict[str, Any],
    *,
    model_dir: Path,
    input_root: Path | None,
    device: str,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    import numpy as np  # type: ignore
    import torch  # type: ignore

    model_id = str(model["model_id"])
    kind = _model_input_kind(model)
    if input_root is not None:
        source_dir = _input_dir(input_root, model_id)
        if kind == "language_tokens":
            input_ids = source_dir / "input_ids.npy"
            attention_mask = source_dir / "attention_mask.npy"
            for source in (input_ids, attention_mask):
                if not source.exists():
                    raise SystemExit(f"Missing shared input artifact for {model_id}: {source}")
            return (
                {
                    "input_ids": torch.from_numpy(np.load(input_ids, allow_pickle=False)).long().to(device),
                    "attention_mask": torch.from_numpy(np.load(attention_mask, allow_pickle=False)).long().to(device),
                },
                {
                    "source_dir": str(source_dir.resolve()),
                    "files": {"input_ids": _sha256_file(input_ids), "attention_mask": _sha256_file(attention_mask)},
                    "shared_input": True,
                    "input_kind": kind,
                },
            )
        if kind == "diffusion_latent":
            sample = source_dir / "sample.npy"
            timestep = source_dir / "timestep.npy"
            for source in (sample, timestep):
                if not source.exists():
                    raise SystemExit(f"Missing shared input artifact for {model_id}: {source}")
            return (
                {
                    "sample": torch.from_numpy(np.load(sample, allow_pickle=False)).float().to(device),
                    "timestep": torch.from_numpy(np.load(timestep, allow_pickle=False)).long().to(device),
                },
                {
                    "source_dir": str(source_dir.resolve()),
                    "files": {"sample": _sha256_file(sample), "timestep": _sha256_file(timestep)},
                    "shared_input": True,
                    "input_kind": kind,
                },
            )
        source = source_dir / "input.npy"
        if not source.exists():
            raise SystemExit(f"Missing shared input artifact for {model_id}: {source}")
        arr = np.load(source, allow_pickle=False)
        tensor = torch.from_numpy(arr).float().to(device)
        return tensor, {"source": str(source.resolve()), "sha256": _sha256_file(source), "shared_input": True, "input_kind": kind}

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    input_dir = model_dir / "inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    if kind == "language_tokens":
        seq_len = int(model.get("sequence_length", 128))
        ids = torch.zeros((1, seq_len), dtype=torch.long)
        tokens = [101, 2023, 2003, 1037, 3722, 3793, 6251, 2005, 2898, 2224, 1012, 102]
        ids[0, : min(seq_len, len(tokens))] = torch.tensor(tokens[:seq_len], dtype=torch.long)
        mask = (ids != 0).long()
        ids_path = input_dir / "input_ids.npy"
        mask_path = input_dir / "attention_mask.npy"
        np.save(ids_path, ids.numpy(), allow_pickle=False)
        np.save(mask_path, mask.numpy(), allow_pickle=False)
        return (
            {"input_ids": ids.to(device), "attention_mask": mask.to(device)},
            {
                "source_dir": _artifact_rel(input_dir, model_dir.parent),
                "files": {"input_ids": _sha256_file(ids_path), "attention_mask": _sha256_file(mask_path)},
                "shared_input": False,
                "input_kind": kind,
            },
        )
    if kind == "diffusion_latent":
        shape = [int(x) for x in model.get("input_shape", [1, 3, 32, 32])]
        sample_cpu = torch.randn(*shape, generator=generator, dtype=torch.float32)
        timestep_cpu = torch.tensor([int(model.get("diffusion_timestep", 10))], dtype=torch.long)
        sample_path = input_dir / "sample.npy"
        timestep_path = input_dir / "timestep.npy"
        np.save(sample_path, sample_cpu.numpy(), allow_pickle=False)
        np.save(timestep_path, timestep_cpu.numpy(), allow_pickle=False)
        return (
            {"sample": sample_cpu.to(device), "timestep": timestep_cpu.to(device)},
            {
                "source_dir": _artifact_rel(input_dir, model_dir.parent),
                "files": {"sample": _sha256_file(sample_path), "timestep": _sha256_file(timestep_path)},
                "shared_input": False,
                "input_kind": kind,
            },
        )
    shape = [int(x) for x in model.get("input_shape", [1, 3, 224, 224])]
    tensor_cpu = torch.randn(*shape, generator=generator, dtype=torch.float32)
    path = input_dir / "input.npy"
    np.save(path, tensor_cpu.numpy(), allow_pickle=False)
    return tensor_cpu.to(device), {"source": _artifact_rel(path, model_dir.parent), "sha256": _sha256_file(path), "shared_input": False, "input_kind": kind}


def _register_activation_hooks(model: Any, layers: list[str], store: dict[str, Any]) -> list[Any]:
    modules = dict(model.named_modules())
    handles = []
    for name in layers:
        module = modules.get(name)
        if module is None:
            store[name] = {"missing": True}
            continue

        def _hook(_module: Any, _inputs: Any, output: Any, *, layer_name: str = name) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            store[layer_name] = value

        handles.append(module.register_forward_hook(_hook))
    return handles


def _register_gradient_hooks(model: Any, layers: list[str], store: dict[str, Any]) -> list[Any]:
    modules = dict(model.named_modules())
    handles = []
    for name in layers:
        module = modules.get(name)
        if module is None:
            store[name] = {"missing": True}
            continue
        param = next(module.parameters(recurse=True), None)
        if param is None:
            store[name] = {"missing": True, "reason": "no_parameter"}
            continue

        def _hook(grad: Any, *, layer_name: str = name) -> None:
            store[layer_name] = grad

        handles.append(param.register_hook(_hook))
    return handles


def _task_metrics(model: dict[str, Any], output: Any, duration_seconds: float, batch_size: int) -> dict[str, Any]:
    import torch  # type: ignore

    metrics: dict[str, Any] = {
        "latency_ms": duration_seconds * 1000.0,
        "throughput_img_s": batch_size / duration_seconds if duration_seconds > 0 else None,
    }
    if isinstance(output, torch.Tensor):
        metrics["output_shape"] = [int(x) for x in output.shape]
        if output.ndim == 2 and output.shape[1] >= 5:
            _, top = output.detach().float().cpu().topk(5, dim=1)
            metrics["top5_indices_sample"] = top[: min(4, top.shape[0])].tolist()
    return metrics


def _run_one_model(args: argparse.Namespace, model: dict[str, Any], registry_path: Path, suite_path: Path) -> dict[str, Any]:
    import torch  # type: ignore

    model_started = time.perf_counter()
    role = str(args.role)
    device_name = _resolve_device(role, str(args.device))
    if role == "gpu-reference" and device_name == "cuda" and not torch.cuda.is_available() and not bool(args.allow_cpu_fallback):
        raise SystemExit("CUDA is not available. Use --device cpu only for smoke tests.")

    adapter_path = Path(args.adapter).resolve() if args.adapter else None

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    torch.set_grad_enabled(True)

    cache_dir = Path(os.environ.get("TBBCC_MODEL_CACHE") or args.model_cache).expanduser().resolve()
    model_id = str(model["model_id"])
    out_dir = Path(args.out).resolve()
    model_dir = out_dir / model_id
    artifact_dir = model_dir / "artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    net, checkpoint = _build_model(
        model,
        pretrained=not bool(args.no_pretrained),
        cache_dir=cache_dir,
        role=role,
    )
    device = torch.device(device_name)
    if not bool(model.get("device_map_managed")):
        net.to(device)
    x, input_meta = _prepare_input(
        model,
        model_dir=model_dir,
        input_root=Path(args.input_root).resolve() if args.input_root else None,
        device=device_name,
        seed=int(args.seed),
    )
    collect_gradients = bool(model.get("collect_gradients", True))
    if collect_gradients:
        _set_input_requires_grad(x)
    activations_raw: dict[str, Any] = {}
    gradients_raw: dict[str, Any] = {}
    handles = []
    hooks = model.get("hooks") or {}
    handles.extend(_register_activation_hooks(net, [str(x) for x in hooks.get("activation_layers", [])], activations_raw))
    if collect_gradients:
        handles.extend(_register_gradient_hooks(net, [str(x) for x in hooks.get("gradient_layers", [])], gradients_raw))

    bridge_namespace: dict[str, Any] | None = None
    try:
        if adapter_path is not None:
            adapter = tbbcc.load_adapter(adapter_path)
            bridge_namespace = {"__name__": "__tbbcc_model_bridge_preamble__"}
            exec(adapter.preamble, bridge_namespace, bridge_namespace)
            torch.manual_seed(int(args.seed))

        started = time.perf_counter()
        with torch.set_grad_enabled(collect_gradients):
            output_obj = _forward_model(net, x, model)
            output = _extract_output_tensor(output_obj)
            loss = output.float().mean()
            if collect_gradients:
                loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize()
        duration = time.perf_counter() - started

        activations = {
            name: value if isinstance(value, dict) else _summarize_tensor(value, artifact_dir / "activations", out_dir, f"activation_{name}")
            for name, value in {**activations_raw, **_extra_activation_channels(output_obj, model)}.items()
        }
        gradients = {
            name: value if isinstance(value, dict) else _summarize_tensor(value, artifact_dir / "gradients", out_dir, f"gradient_{name}")
            for name, value in gradients_raw.items()
        }
        result = _summarize_tensor(output, artifact_dir / "result", out_dir, "output")
        metrics = _task_metrics(model, output, duration, _input_batch_size(x))
        if device.type == "cuda":
            metrics["peak_memory_mb"] = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
        else:
            metrics["peak_memory_mb"] = None
        metrics["loss_scalar"] = float(loss.detach().float().cpu())
        artifact = {
            "schema_version": SCHEMA_ARTIFACT,
            "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
            "model_id": model_id,
            "display_name": model.get("display_name"),
            "role": role,
            "status": "passed",
            "registry_path": str(registry_path),
            "suite_path": str(suite_path),
            "environment": _environment(role, device_name),
            "checkpoint": checkpoint,
            "input": input_meta,
            "channels": {
                "result": result,
                "activations": activations,
                "gradients": gradients,
                "task_metrics": metrics,
            },
            "figure_role": model.get("figure_role"),
        }
        artifact_path = model_dir / "model_artifact.json"
        artifact_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
        return {
            "model_id": model_id,
            "status": "passed",
            "artifact_json": str(artifact_path.resolve()),
            "figure_role": model.get("figure_role"),
            "activation_layers": len(activations),
            "gradient_layers": len(gradients),
            "duration_seconds": time.perf_counter() - model_started,
        }
    finally:
        for handle in handles:
            try:
                handle.remove()
            except Exception:
                pass
        if bridge_namespace is not None:
            env = bridge_namespace.get("_env")
            if hasattr(env, "__exit__"):
                env.__exit__(None, None, None)


def cmd_validate(args: argparse.Namespace) -> int:
    registry = _load_json(Path(args.registry).resolve())
    suite = _load_json(Path(args.suite).resolve()) if args.suite else None
    errors = validate_registry_payload(registry, suite)
    print(json.dumps({"ok": not errors, "errors": errors}, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


def cmd_plan(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry).resolve()
    suite_path = Path(args.suite).resolve()
    models = _selected_models(registry_path, suite_path)
    plan = {
        "schema_version": "tbbcc.model_suite.plan.v1",
        "registry": str(registry_path),
        "suite": str(suite_path),
        "model_count": len(models),
        "models": [
            {
                "model_id": item["model_id"],
                "display_name": item.get("display_name"),
                "figure_role": item.get("figure_role"),
                "activation_layers": item.get("hooks", {}).get("activation_layers", []),
                "gradient_layers": item.get("hooks", {}).get("gradient_layers", []),
                "metrics": item.get("metrics", []),
            }
            for item in models
        ],
        "figure_contract": {
            "core_conclusion": "Canonical models identify where NPU bridge execution first diverges from GPU PyTorch reference and whether that drift propagates to gradients or short training.",
            "archetype": "quantitative grid",
            "primary_panels": ["layerwise_fne_curve", "first_divergence_layer", "gradient_consistency_curve", "loss_curve_dtw"],
            "exports": ["pdf", "svg", "tiff"],
        },
    }
    if args.out:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(plan, indent=2, ensure_ascii=False))
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry).resolve()
    suite_path = Path(args.suite).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = _selected_models(registry_path, suite_path)
    requested = set(args.model_id or [])
    if requested:
        selected = [item for item in selected if item["model_id"] in requested]
    skipped: list[dict[str, Any]] = []
    if args.input_root and not bool(getattr(args, "strict_input_root", False)):
        input_root = Path(args.input_root).resolve()
        passed_ids = _manifest_passed_model_ids(input_root)
        kept: list[dict[str, Any]] = []
        for model in selected:
            model_id = str(model["model_id"])
            input_path = _input_artifact_path(input_root, model_id)
            if (passed_ids and model_id not in passed_ids) or not _has_input_artifacts(input_root, model):
                skipped.append(
                    {
                        "model_id": model_id,
                        "status": "skipped",
                        "reason": "MissingGPUReferenceInput",
                        "input_artifact": str(_input_dir(input_root, model_id)),
                    }
                )
                continue
            kept.append(model)
        selected = kept
    started = _dt.datetime.now(_dt.UTC)
    budget_seconds = float(args.time_budget_seconds) if args.time_budget_seconds is not None else None
    max_models = int(args.max_models) if args.max_models is not None else None
    wall_started = time.perf_counter()
    cases: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, model in enumerate(selected):
        if max_models is not None and len(cases) >= max_models:
            skipped.extend(
                {
                    "model_id": str(item["model_id"]),
                    "status": "skipped",
                    "reason": "MaxModelsReached",
                }
                for item in selected[index:]
            )
            break
        if budget_seconds is not None and (time.perf_counter() - wall_started) >= budget_seconds:
            skipped.extend(
                {
                    "model_id": str(item["model_id"]),
                    "status": "skipped",
                    "reason": "TimeBudgetExceededBeforeStart",
                    "time_budget_seconds": budget_seconds,
                }
                for item in selected[index:]
            )
            break
        try:
            if bool(getattr(args, "isolate_models", False)):
                remaining_budget = None
                if budget_seconds is not None:
                    remaining_budget = max(1.0, budget_seconds - (time.perf_counter() - wall_started))
                cases.append(_run_one_model_isolated(args, model, out_dir, remaining_budget))
            else:
                cases.append(_run_one_model(args, model, registry_path, suite_path))
        except Exception as exc:
            failures.append({"model_id": model.get("model_id"), "status": "failed", "error": repr(exc)})
            if not bool(args.keep_going):
                raise
        if budget_seconds is not None and (time.perf_counter() - wall_started) >= budget_seconds:
            skipped.extend(
                {
                    "model_id": str(item["model_id"]),
                    "status": "skipped",
                    "reason": "TimeBudgetExceededAfterModel",
                    "time_budget_seconds": budget_seconds,
                }
                for item in selected[index + 1 :]
            )
            break
    manifest = {
        "schema_version": SCHEMA_MANIFEST,
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "role": args.role,
        "registry": str(registry_path),
        "suite": str(suite_path),
        "totals": {
            "total": len(cases) + len(failures) + len(skipped),
            "passed": len(cases),
            "failed": len(failures),
            "skipped": len(skipped),
            "duration_seconds": (_dt.datetime.now(_dt.UTC) - started).total_seconds(),
            "time_budget_seconds": budget_seconds,
            "max_models": max_models,
        },
        "cases": cases + failures + skipped,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"manifest_json": str((out_dir / "manifest.json").resolve()), "totals": manifest["totals"]}, indent=2))
    return 0 if not failures else 1


def _run_one_model_isolated(
    args: argparse.Namespace,
    model: dict[str, Any],
    out_dir: Path,
    remaining_budget: float | None,
) -> dict[str, Any]:
    model_id = str(model["model_id"])
    with tempfile.TemporaryDirectory(prefix=f"tbbcc-{model_id}-", dir=out_dir) as temporary:
        child_out = Path(temporary)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "collect",
            "--registry",
            str(Path(args.registry).resolve()),
            "--suite",
            str(Path(args.suite).resolve()),
            "--out",
            str(child_out),
            "--role",
            str(args.role),
            "--device",
            str(args.device),
            "--model-id",
            model_id,
            "--model-cache",
            str(args.model_cache),
            "--seed",
            str(args.seed),
            "--keep-going",
        ]
        for option, value in (("--adapter", args.adapter), ("--input-root", args.input_root)):
            if value:
                command.extend([option, str(value)])
        if remaining_budget is not None:
            command.extend(["--time-budget-seconds", str(remaining_budget)])
        if bool(args.strict_input_root):
            command.append("--strict-input-root")
        if bool(args.no_pretrained):
            command.append("--no-pretrained")
        if bool(args.allow_cpu_fallback):
            command.append("--allow-cpu-fallback")

        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        manifest_path = child_out / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(
                f"Isolated model process failed for {model_id} (exit {completed.returncode}): "
                f"{completed.stderr[-2000:]}"
            )
        child_manifest = _load_json(manifest_path)
        child_cases = child_manifest.get("cases") or []
        if len(child_cases) != 1 or child_cases[0].get("status") != "passed":
            detail = child_cases[0] if child_cases else completed.stderr[-2000:]
            raise RuntimeError(f"Isolated model process failed for {model_id}: {detail}")

        source_model_dir = child_out / model_id
        target_model_dir = out_dir / model_id
        shutil.copytree(source_model_dir, target_model_dir, dirs_exist_ok=True)
        result = dict(child_cases[0])
        result["artifact_json"] = str((target_model_dir / "model_artifact.json").resolve())
        result["isolated_process"] = True
        return result


def _artifact_path(root: Path, model_id: str) -> Path:
    return root / model_id / "model_artifact.json"


def _load_array(summary: dict[str, Any], root: Path) -> Any:
    import numpy as np  # type: ignore

    artifact = summary.get("artifact_path")
    if not artifact:
        return None
    path = Path(str(artifact))
    if not path.is_absolute():
        path = root / path
    return np.load(path, allow_pickle=False)


def _manifest_cases_by_model(root: Path) -> dict[str, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = _load_json(manifest_path)
    return {
        str(item["model_id"]): item
        for item in manifest.get("cases") or []
        if isinstance(item, dict) and item.get("model_id")
    }


def _numeric_compare(a: Any, b: Any, atol: float, rtol: float) -> dict[str, Any]:
    import numpy as np  # type: ignore

    if a is None or b is None:
        return {"passed": False, "reason": "MissingArtifact"}
    if tuple(a.shape) != tuple(b.shape):
        return {"passed": False, "reason": "ShapeMismatch", "gpu_shape": list(a.shape), "npu_shape": list(b.shape)}
    av = a.reshape(-1).astype("float64", copy=False)
    bv = b.reshape(-1).astype("float64", copy=False)
    diff = np.abs(av - bv)
    finite_mask = np.isfinite(av) & np.isfinite(bv) & np.isfinite(diff)
    finite = diff[finite_mask]
    close = diff <= (atol + rtol * np.abs(bv))
    dot = float(np.dot(av, bv))
    norm = float(np.linalg.norm(av) * np.linalg.norm(bv))
    cosine = dot / norm if norm > 0 else None
    ref_scale = float(np.mean(np.abs(av[finite_mask]))) if np.any(finite_mask) else 0.0
    mae = float(np.mean(finite)) if finite.size else float("inf")
    p95 = float(np.percentile(finite, 95)) if finite.size else float("inf")
    p99 = float(np.percentile(finite, 99)) if finite.size else float("inf")
    outlier_ratio = (float(np.max(finite)) / p95) if finite.size and p95 > 0 else None
    return {
        "passed": bool(np.all(close)),
        "reason": None if bool(np.all(close)) else "NumericMismatch",
        "count": int(av.size),
        "mae": mae,
        "relative_mae": (mae / ref_scale) if ref_scale > 0 else None,
        "max_error": float(np.max(diff)) if diff.size else 0.0,
        "p50": float(np.percentile(finite, 50)) if finite.size else float("inf"),
        "p95": p95,
        "p99": p99,
        "cosine": cosine,
        "failed_count": int(np.size(close) - np.count_nonzero(close)),
        "finite_count": int(np.count_nonzero(finite_mask)),
        "nonfinite_count": int(av.size - np.count_nonzero(finite_mask)),
        "finite_fraction": float(np.count_nonzero(finite_mask) / av.size) if av.size else None,
        "outlier_ratio_max_to_p95": outlier_ratio,
        "outlier_dominated": bool(outlier_ratio is not None and outlier_ratio > 1e6),
        "atol": atol,
        "rtol": rtol,
    }


def _numeric_quality(metrics: dict[str, Any], *, aligned_cosine: float, usable_cosine: float, aligned_p95: float, usable_p95: float) -> str:
    if metrics.get("passed") is True:
        return "strict_pass"
    if metrics.get("reason") in {"MissingArtifact", "ShapeMismatch", "MissingLayer"}:
        return "unavailable"
    if metrics.get("nonfinite_count"):
        return "invalid_nonfinite"
    if metrics.get("outlier_dominated"):
        return "outlier_dominated"
    cosine = metrics.get("cosine")
    p95 = metrics.get("p95")
    if not isinstance(cosine, (int, float)) or not isinstance(p95, (int, float)):
        return "unknown"
    if cosine >= aligned_cosine and p95 <= aligned_p95:
        return "aligned_with_tolerance"
    if cosine >= usable_cosine and p95 <= usable_p95:
        return "usable_drift"
    return "diverged"


def _compare_named_channel(
    gpu: dict[str, Any],
    npu: dict[str, Any],
    gpu_root: Path,
    npu_root: Path,
    atol: float,
    rtol: float,
    order: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordered = [str(item) for item in (order or [])]
    extras = sorted((set(gpu) | set(npu)) - set(ordered))
    names = ordered + extras
    for name in names:
        gv = gpu.get(name)
        nv = npu.get(name)
        if not isinstance(gv, dict) or not isinstance(nv, dict) or gv.get("missing") or nv.get("missing"):
            rows.append({"name": name, "passed": False, "reason": "MissingLayer"})
            continue
        metrics = _numeric_compare(_load_array(gv, gpu_root), _load_array(nv, npu_root), atol, rtol)
        rows.append({"name": name, **metrics})
    return rows


def _annotate_numeric_rows(
    rows: list[dict[str, Any]],
    *,
    aligned_cosine: float,
    usable_cosine: float,
    aligned_p95: float,
    usable_p95: float,
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "quality": _numeric_quality(
                row,
                aligned_cosine=aligned_cosine,
                usable_cosine=usable_cosine,
                aligned_p95=aligned_p95,
                usable_p95=usable_p95,
            ),
        }
        for row in rows
    ]


def _first_divergence(rows: list[dict[str, Any]], threshold: float) -> str | None:
    for row in rows:
        cosine = row.get("cosine")
        if row.get("passed") is False or (isinstance(cosine, (int, float)) and cosine < threshold):
            return str(row.get("name"))
    return None


def _first_quality_drop(rows: list[dict[str, Any]], allowed: set[str]) -> str | None:
    for row in rows:
        if str(row.get("quality")) not in allowed:
            return str(row.get("name"))
    return None


def _summarize_quality(rows: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        quality = str(row.get("quality") or "unknown")
        out[quality] = out.get(quality, 0) + 1
    return out


def _latency_ratio(gpu_metrics: dict[str, Any], npu_metrics: dict[str, Any]) -> float | None:
    gpu = gpu_metrics.get("latency_ms")
    npu = npu_metrics.get("latency_ms")
    if isinstance(gpu, (int, float)) and isinstance(npu, (int, float)) and gpu > 0:
        return float(npu / gpu)
    return None


def _write_model_suite_source_data(summary: dict[str, Any], out_dir: Path) -> None:
    source_dir = out_dir / "source_data"
    source_dir.mkdir(parents=True, exist_ok=True)
    fne_path = source_dir / "layerwise_fne.csv"
    with fne_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "layer",
                "quality",
                "passed",
                "cosine",
                "mae",
                "relative_mae",
                "p95",
                "p99",
                "max_error",
                "failed_count",
                "count",
                "outlier_dominated",
            ],
        )
        writer.writeheader()
        for model in summary.get("models") or []:
            for row in model.get("fne_curve") or []:
                writer.writerow(
                    {
                        "model_id": model.get("model_id"),
                        "layer": row.get("name"),
                        "quality": row.get("quality"),
                        "passed": row.get("passed"),
                        "cosine": row.get("cosine"),
                        "mae": row.get("mae"),
                        "relative_mae": row.get("relative_mae"),
                        "p95": row.get("p95"),
                        "p99": row.get("p99"),
                        "max_error": row.get("max_error"),
                        "failed_count": row.get("failed_count"),
                        "count": row.get("count"),
                        "outlier_dominated": row.get("outlier_dominated"),
                    }
                )
    model_path = source_dir / "model_summary.csv"
    with model_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model_id",
                "status",
                "numerical_verdict",
                "output_quality",
                "first_divergence_layer",
                "first_quality_drop_layer",
                "latency_ratio_npu_over_gpu",
                "npu_error",
            ],
        )
        writer.writeheader()
        missing = {item["model_id"]: item for item in summary.get("missing_models") or [] if item.get("model_id")}
        for model in summary.get("models") or []:
            writer.writerow(
                {
                    "model_id": model.get("model_id"),
                    "status": "compared",
                    "numerical_verdict": model.get("numerical_verdict"),
                    "output_quality": (model.get("output_drift") or {}).get("quality"),
                    "first_divergence_layer": model.get("first_divergence_layer"),
                    "first_quality_drop_layer": model.get("first_quality_drop_layer"),
                    "latency_ratio_npu_over_gpu": model.get("latency_ratio_npu_over_gpu"),
                    "npu_error": "",
                }
            )
        for model_id, item in missing.items():
            writer.writerow(
                {
                    "model_id": model_id,
                    "status": "missing",
                    "numerical_verdict": "unavailable",
                    "output_quality": "",
                    "first_divergence_layer": "",
                    "first_quality_drop_layer": "",
                    "latency_ratio_npu_over_gpu": "",
                    "npu_error": item.get("npu_error") or item.get("reason"),
                }
            )


def _fmt_float(value: Any, digits: int = 4) -> str:
    if isinstance(value, (int, float)):
        if value == 0:
            return "0"
        if abs(float(value)) < 1e-3 or abs(float(value)) >= 1e4:
            return f"{float(value):.{digits}e}"
        return f"{float(value):.{digits}f}"
    return "NA"


def _manuscript_result_text(summary: dict[str, Any], *, chinese: bool) -> str:
    verdict_labels = {
        "aligned": ("stable alignment", "稳定对齐"),
        "usable_with_drift": ("usable, bounded drift", "存在可用且有界的漂移"),
        "outlier_dominated": ("outlier-dominated drift", "异常值主导的漂移"),
        "diverged": ("numerical divergence", "数值发散"),
    }
    results = []
    for model in summary.get("models") or []:
        name = str(model.get("display_name") or model.get("model_id"))
        verdict = str(model.get("numerical_verdict"))
        label = verdict_labels.get(verdict, (verdict, verdict))[1 if chinese else 0]
        output = model.get("output_drift") or {}
        cosine = _fmt_float(output.get("cosine"), 6)
        p95 = _fmt_float(output.get("p95"), 4)
        if chinese:
            results.append(f"{name} 表现为{label}（输出余弦相似度 {cosine}，P95 绝对误差 {p95}）")
        else:
            results.append(f"{name} shows {label} (output cosine {cosine}; P95 absolute error {p95})")
    for model in summary.get("missing_models") or []:
        name = str(model.get("display_name") or model.get("model_id"))
        error = str(model.get("npu_error") or model.get("reason") or "missing artifact")
        if chinese:
            results.append(f"{name} 未生成可比 artifact（{error}）")
        else:
            results.append(f"{name} has no comparable artifact ({error})")
    separator = "；" if chinese else "; "
    return separator.join(results) + ("。" if chinese and results else "." if results else "")


def _write_model_suite_markdown(summary: dict[str, Any], out_dir: Path) -> None:
    totals = summary.get("totals") or {}
    lines = [
        "# Numeric Alignment Report",
        "",
        "## Summary",
        "",
        f"- Benchmark verdict: `{summary.get('benchmark_verdict')}`",
        f"- Expected models: {totals.get('expected_models')}",
        f"- Compared models: {totals.get('models')}",
        f"- Aligned: {totals.get('aligned')}",
        f"- Usable with drift: {totals.get('usable_with_drift')}",
        f"- Outlier dominated: {totals.get('outlier_dominated')}",
        f"- Diverged: {totals.get('diverged')}",
        f"- Missing: {totals.get('missing')}",
        "",
        "## Model Results",
        "",
        "| Model | Verdict | Output cosine | Output P95 | First quality drop | NPU/GPU latency |",
        "| --- | --- | ---: | ---: | --- | ---: |",
    ]
    for model in summary.get("models") or []:
        output = model.get("output_drift") or {}
        latency = model.get("latency_ratio_npu_over_gpu")
        lines.append(
            "| {model} | `{verdict}` | {cosine} | {p95} | {drop} | {latency} |".format(
                model=model.get("display_name") or model.get("model_id"),
                verdict=model.get("numerical_verdict"),
                cosine=_fmt_float(output.get("cosine"), 6),
                p95=_fmt_float(output.get("p95"), 4),
                drop=model.get("first_quality_drop_layer") or "none",
                latency=f"{latency:.1f}x" if isinstance(latency, (int, float)) else "NA",
            )
        )
    if summary.get("missing_models"):
        lines.extend(["", "## Missing Or Failed Models", "", "| Model | NPU status | Error |", "| --- | --- | --- |"])
        for item in summary.get("missing_models") or []:
            lines.append(
                f"| {item.get('display_name') or item.get('model_id')} | {item.get('npu_status') or 'missing'} | `{item.get('npu_error') or item.get('reason')}` |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This report separates strict numerical pass/fail from benchmark-grade interpretation. "
            "`aligned` indicates stable layer-wise agreement under the configured benchmark thresholds; "
            "`outlier_dominated` indicates that most elements remain close but a small number of extreme values dominate max/MAE statistics; "
            "`unavailable` indicates that the NPU bridge did not produce a comparable artifact.",
            "",
            "## Source Data",
            "",
            "- `summary.json`",
            "- `source_data/model_summary.csv`",
            "- `source_data/layerwise_fne.csv`",
            "- `figures/figure_1_numeric_alignment.pdf`",
            "- `figures/figure_1_numeric_alignment.svg`",
            "- `figures/figure_1_numeric_alignment.tiff`",
            "",
            "## Manuscript Draft",
            "",
            (
                "TorchBridgeBench's numeric-only comparison distinguishes stable numerical alignment, outlier-dominated drift, and bridge coverage failures from the same GPU-reference/NPU-artifact evidence chain. "
                + _manuscript_result_text(summary, chinese=False)
                + " Together, these outcomes demonstrate that the benchmark reports not only whether a bridge matches strict tolerances, but also where and why numerical evidence becomes unreliable."
            ),
            "",
            "## 中文结果段落草稿",
            "",
            (
                "TBBCC 的仅数值比对流程能够从同一组 GPU reference 与 NPU bridge artifact 中区分稳定对齐、异常值主导漂移和桥接覆盖失败三类结果。"
                + _manuscript_result_text(summary, chinese=True)
                + "这些结果表明，该系统不仅给出严格容差下的通过/失败判断，还能定位数值证据何处开始失效，并将执行失败与真实数值漂移分离。"
            ),
        ]
    )
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quality_color(quality: str | None) -> str:
    colors = {
        "aligned_with_tolerance": "#3B8C5A",
        "strict_pass": "#3B8C5A",
        "usable_drift": "#D7A12B",
        "usable_with_drift": "#D7A12B",
        "outlier_dominated": "#C4513F",
        "diverged": "#8C2D2D",
        "unavailable": "#BDBDBD",
    }
    return colors.get(str(quality), "#8F8F8F")


def _select_diagnostic_model(models: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str]:
    outlier = next((m for m in models if m.get("numerical_verdict") == "outlier_dominated"), None)
    if outlier is not None:
        return outlier, "outlier"
    drift = [m for m in models if m.get("numerical_verdict") == "usable_with_drift"]
    if drift:
        return max(drift, key=lambda m: float((m.get("output_drift") or {}).get("p95") or 0.0)), "drift"
    return None, "none"


def _plot_model_suite_numeric(summary: dict[str, Any], out_dir: Path) -> list[str]:
    try:
        import matplotlib as mpl  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return []

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[0.82, 1.18])
    ax_workflow = fig.add_subplot(gs[0, 0])
    ax_matrix = fig.add_subplot(gs[0, 1])
    ax_fne = fig.add_subplot(gs[1, 0])
    ax_outlier = fig.add_subplot(gs[1, 1])

    fig.suptitle("Benchmark-grade GPU-NPU numeric alignment", x=0.02, ha="left", fontsize=10, fontweight="bold")

    # Panel A: workflow
    ax_workflow.axis("off")
    boxes = [
        (0.05, 0.62, "GPU PyTorch\nreference"),
        (0.38, 0.62, "NPU bridge\nartifact"),
        (0.71, 0.62, "TBBCC\ncompare"),
        (0.34, 0.20, "Verdict +\nsource data"),
    ]
    for x, y, label in boxes:
        width = 0.32 if "Verdict" in label else 0.24
        ax_workflow.add_patch(plt.Rectangle((x, y), width, 0.2, facecolor="#F3F5F6", edgecolor="#5D6A72", linewidth=0.8))
        ax_workflow.text(x + width / 2, y + 0.1, label, ha="center", va="center", fontsize=6.6)
    for start, end in [((0.29, 0.72), (0.38, 0.72)), ((0.62, 0.72), (0.71, 0.72)), ((0.83, 0.62), (0.50, 0.42))]:
        ax_workflow.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "#4D4D4D"})
    ax_workflow.text(0.0, 1.02, "A", transform=ax_workflow.transAxes, fontweight="bold", fontsize=9)
    ax_workflow.text(0.0, 0.03, "No adapter generation or eval-suite rerun", transform=ax_workflow.transAxes, fontsize=6.5, color="#555555")

    # Panel B: verdict matrix
    rows: list[tuple[str, str]] = []
    for model in summary.get("models") or []:
        rows.append((str(model.get("display_name") or model.get("model_id")), str(model.get("numerical_verdict"))))
    for item in summary.get("missing_models") or []:
        rows.append((str(item.get("display_name") or item.get("model_id")), "unavailable"))
    labels = [r[0] for r in rows]
    verdicts = [r[1] for r in rows]
    ax_matrix.barh(range(len(rows)), [1] * len(rows), color=[_quality_color(v if v != "aligned" else "aligned_with_tolerance") for v in verdicts], height=0.58)
    ax_matrix.set_yticks(range(len(rows)), labels)
    ax_matrix.set_xlim(0, 1.0)
    ax_matrix.set_xticks([])
    ax_matrix.invert_yaxis()
    for i, verdict in enumerate(verdicts):
        ax_matrix.text(0.03, i, verdict, va="center", ha="left", color="white" if verdict != "unavailable" else "#333333", fontsize=7, fontweight="bold")
    ax_matrix.set_title("Model-level numeric verdicts", loc="left", fontsize=8, fontweight="bold")
    ax_matrix.text(-0.08, 1.02, "B", transform=ax_matrix.transAxes, fontweight="bold", fontsize=9)

    # Panel C: ResNet FNE
    aligned_model = next((m for m in summary.get("models") or [] if m.get("numerical_verdict") == "aligned"), None)
    if aligned_model:
        layers = [str(r.get("name")) for r in aligned_model.get("fne_curve") or []]
        cosines = [float(r.get("cosine")) if isinstance(r.get("cosine"), (int, float)) else np.nan for r in aligned_model.get("fne_curve") or []]
        p95 = [float(r.get("p95")) if isinstance(r.get("p95"), (int, float)) else np.nan for r in aligned_model.get("fne_curve") or []]
        x = np.arange(len(layers))
        one_minus_cosine = np.maximum(1 - np.asarray(cosines, dtype=float), 1e-12)
        p95_values = np.maximum(np.asarray(p95, dtype=float), 1e-12)
        ax_fne.plot(x, one_minus_cosine, marker="o", color="#2F6B9A", linewidth=1.4, markersize=3.5, label="1 - cosine")
        ax_fne.plot(x, p95_values, marker="s", color="#D7A12B", linewidth=1.1, markersize=3.2, label="P95 error")
        ax_fne.set_yscale("log")
        ax_fne.set_xticks(x, layers, rotation=35, ha="right")
        ax_fne.set_ylabel("metric value (log)")
        ax_fne.set_title(f"{aligned_model.get('display_name')} remains aligned", loc="left", fontsize=8, fontweight="bold")
        ax_fne.legend(loc="lower right", fontsize=6)
    else:
        ax_fne.text(0.5, 0.5, "No aligned model available", ha="center", va="center")
    ax_fne.text(-0.16, 1.04, "C", transform=ax_fne.transAxes, fontweight="bold", fontsize=9)

    # Panel D: strongest diagnostic case, preferring outlier evidence.
    diagnostic_model, diagnostic_kind = _select_diagnostic_model(summary.get("models") or [])
    if diagnostic_model and diagnostic_kind == "outlier":
        rows = diagnostic_model.get("fne_curve") or []
        layers = [str(r.get("name")) for r in rows]
        p95 = np.asarray([float(r.get("p95")) if isinstance(r.get("p95"), (int, float)) else np.nan for r in rows])
        max_err = np.asarray([float(r.get("max_error")) if isinstance(r.get("max_error"), (int, float)) else np.nan for r in rows])
        ratio = np.divide(max_err, p95, out=np.full_like(max_err, np.nan), where=p95 > 0)
        log_ratio = np.log10(ratio, out=np.full_like(ratio, np.nan), where=ratio > 0)
        x = np.arange(len(layers))
        ax_outlier.bar(x, log_ratio, width=0.58, color=["#C4513F" if str(r.get("quality")) == "outlier_dominated" else "#B9C6CF" for r in rows])
        finite_ratio = log_ratio[np.isfinite(log_ratio)]
        if finite_ratio.size:
            ax_outlier.set_ylim(0, max(1.0, float(np.nanmax(finite_ratio)) + 3.0))
        ax_outlier.set_xticks(x, layers, rotation=35, ha="right")
        ax_outlier.set_ylabel("log10(max / P95 error)")
        ax_outlier.set_title(f"{diagnostic_model.get('display_name')} is outlier dominated", loc="left", fontsize=8, fontweight="bold")
        drop = diagnostic_model.get("first_quality_drop_layer")
        if drop in layers:
            idx = layers.index(str(drop))
            ax_outlier.annotate("extreme outlier\nwith small P95", xy=(idx, log_ratio[idx]), xytext=(idx + 0.75, max(log_ratio[idx] - 7, 1.0)), arrowprops={"arrowstyle": "->", "lw": 0.7}, fontsize=6.5)
            ax_outlier.text(idx + 0.05, max(log_ratio[idx] - 12, 1.0), f"P95={p95[idx]:.1e}", fontsize=6.2, color="#555555")
    elif diagnostic_model:
        rows = diagnostic_model.get("fne_curve") or []
        layers = [str(r.get("name")) for r in rows]
        cosines = np.asarray([float(r.get("cosine")) if isinstance(r.get("cosine"), (int, float)) else np.nan for r in rows])
        p95 = np.asarray([float(r.get("p95")) if isinstance(r.get("p95"), (int, float)) else np.nan for r in rows])
        x = np.arange(len(layers))
        ax_outlier.plot(x, np.maximum(1 - cosines, 1e-12), marker="o", color="#2F6B9A", linewidth=1.4, markersize=3.5, label="1 - cosine")
        ax_outlier.plot(x, np.maximum(p95, 1e-12), marker="s", color="#D7A12B", linewidth=1.1, markersize=3.2, label="P95 error")
        ax_outlier.set_yscale("log")
        ax_outlier.set_xticks(x, layers, rotation=35, ha="right")
        ax_outlier.set_ylabel("metric value (log)")
        ax_outlier.set_title(f"{diagnostic_model.get('display_name')}: bounded drift", loc="left", fontsize=8, fontweight="bold")
        ax_outlier.legend(loc="lower right", fontsize=6)
    else:
        ax_outlier.text(0.5, 0.5, "No diagnostic model available", ha="center", va="center")
    ax_outlier.text(-0.12, 1.04, "D", transform=ax_outlier.transAxes, fontweight="bold", fontsize=9)

    base = fig_dir / "figure_1_numeric_alignment"
    outputs = []
    for suffix, kwargs in [
        (".svg", {}),
        (".pdf", {}),
        (".tiff", {"dpi": 600}),
        (".png", {"dpi": 300}),
    ]:
        path = base.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", **kwargs)
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def cmd_compare(args: argparse.Namespace) -> int:
    registry_path = Path(args.registry).resolve()
    suite_path = Path(args.suite).resolve()
    gpu_root = Path(args.gpu_reference).resolve()
    npu_root = Path(args.npu_bridge).resolve()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = _selected_models(registry_path, suite_path)
    npu_manifest_cases = _manifest_cases_by_model(npu_root)
    results: list[dict[str, Any]] = []
    missing_models: list[dict[str, Any]] = []
    for model in models:
        model_id = str(model["model_id"])
        gpu_path = _artifact_path(gpu_root, model_id)
        npu_path = _artifact_path(npu_root, model_id)
        if not gpu_path.exists() or not npu_path.exists():
            npu_manifest_case = npu_manifest_cases.get(model_id) or {}
            missing_models.append(
                {
                    "model_id": model_id,
                    "display_name": model.get("display_name"),
                    "figure_role": model.get("figure_role"),
                    "reason": "MissingModelArtifact",
                    "gpu_artifact_present": gpu_path.exists(),
                    "npu_artifact_present": npu_path.exists(),
                    "npu_status": npu_manifest_case.get("status"),
                    "npu_error": npu_manifest_case.get("error"),
                    "gpu_artifact": str(gpu_path),
                    "npu_artifact": str(npu_path),
                }
            )
            continue
        gpu = _load_json(gpu_path)
        npu = _load_json(npu_path)
        channels_g = gpu.get("channels") or {}
        channels_n = npu.get("channels") or {}
        output = _numeric_compare(
            _load_array(channels_g.get("result") or {}, gpu_root),
            _load_array(channels_n.get("result") or {}, npu_root),
            float(args.atol),
            float(args.rtol),
        )
        output["quality"] = _numeric_quality(
            output,
            aligned_cosine=float(args.aligned_cosine),
            usable_cosine=float(args.usable_cosine),
            aligned_p95=float(args.aligned_p95),
            usable_p95=float(args.usable_p95),
        )
        hooks = model.get("hooks") or {}
        fne = _annotate_numeric_rows(
            _compare_named_channel(
                channels_g.get("activations") or {},
                channels_n.get("activations") or {},
                gpu_root,
                npu_root,
                float(args.atol),
                float(args.rtol),
                [str(item) for item in hooks.get("activation_layers", [])],
            ),
            aligned_cosine=float(args.aligned_cosine),
            usable_cosine=float(args.usable_cosine),
            aligned_p95=float(args.aligned_p95),
            usable_p95=float(args.usable_p95),
        )
        gc = _annotate_numeric_rows(
            _compare_named_channel(
                channels_g.get("gradients") or {},
                channels_n.get("gradients") or {},
                gpu_root,
                npu_root,
                float(args.atol),
                float(args.rtol),
                [str(item) for item in hooks.get("gradient_layers", [])],
            ),
            aligned_cosine=float(args.aligned_cosine),
            usable_cosine=float(args.usable_cosine),
            aligned_p95=float(args.aligned_p95),
            usable_p95=float(args.usable_p95),
        )
        quality_counts = _summarize_quality(fne)
        allowed = {"strict_pass", "aligned_with_tolerance", "usable_drift"}
        fne_qualities = {str(row.get("quality")) for row in fne}
        usable_forward = output.get("quality") in allowed and all(str(row.get("quality")) in allowed for row in fne)
        all_aligned = output.get("quality") in {"strict_pass", "aligned_with_tolerance"} and all(
            str(row.get("quality")) in {"strict_pass", "aligned_with_tolerance"} for row in fne
        )
        numerical_verdict = (
            "aligned"
            if all_aligned
            else ("usable_with_drift" if usable_forward else ("outlier_dominated" if "outlier_dominated" in fne_qualities else "diverged"))
        )
        gpu_metrics = channels_g.get("task_metrics") or {}
        npu_metrics = channels_n.get("task_metrics") or {}
        results.append(
            {
                "model_id": model_id,
                "display_name": model.get("display_name"),
                "figure_role": model.get("figure_role"),
                "passed": bool(output.get("passed") and all(row.get("passed") is not False for row in fne)),
                "benchmark_usable": usable_forward,
                "numerical_verdict": numerical_verdict,
                "output_drift": output,
                "fne_curve": fne,
                "gc_curve": gc,
                "fne_quality_counts": quality_counts,
                "first_divergence_layer": _first_divergence(fne, float(args.cosine_threshold)),
                "first_quality_drop_layer": _first_quality_drop(fne, allowed),
                "latency_ratio_npu_over_gpu": _latency_ratio(gpu_metrics, npu_metrics),
                "gpu_task_metrics": gpu_metrics,
                "npu_task_metrics": npu_metrics,
            }
        )
    figure_candidates = [
        {
            "figure_id": "canonical_layerwise_fne",
            "title": "Layer-wise GPU-NPU forward numerical equivalence",
            "source": "Canonical Model Suite",
            "models": [item["model_id"] for item in results],
            "required_fields": ["fne_curve", "first_divergence_layer"],
        },
        {
            "figure_id": "canonical_gradient_consistency",
            "title": "Gradient consistency across canonical models",
            "source": "Canonical Model Suite",
            "models": [item["model_id"] for item in results if item.get("gc_curve")],
            "required_fields": ["gc_curve"],
        },
    ]
    summary = {
        "schema_version": SCHEMA_COMPARISON,
        "created_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "registry": str(registry_path),
        "suite": str(suite_path),
        "gpu_reference": str(gpu_root),
        "npu_bridge": str(npu_root),
        "totals": {
            "expected_models": len(models),
            "models": len(results),
            "passed": sum(1 for item in results if item.get("passed")),
            "failed": sum(1 for item in results if not item.get("passed")),
            "benchmark_usable": sum(1 for item in results if item.get("benchmark_usable")),
            "aligned": sum(1 for item in results if item.get("numerical_verdict") == "aligned"),
            "usable_with_drift": sum(1 for item in results if item.get("numerical_verdict") == "usable_with_drift"),
            "outlier_dominated": sum(1 for item in results if item.get("numerical_verdict") == "outlier_dominated"),
            "diverged": sum(1 for item in results if item.get("numerical_verdict") == "diverged"),
            "missing": len(missing_models),
        },
        "quality_thresholds": {
            "strict": {"atol": float(args.atol), "rtol": float(args.rtol)},
            "aligned": {"cosine": float(args.aligned_cosine), "p95": float(args.aligned_p95)},
            "usable": {"cosine": float(args.usable_cosine), "p95": float(args.usable_p95)},
        },
        "benchmark_verdict": "usable_partial" if results else "unusable_no_overlap",
        "figure_candidates": figure_candidates,
        "missing_models": missing_models,
        "models": results,
    }
    if len(results) == len(models) and all(item.get("benchmark_usable") for item in results):
        summary["benchmark_verdict"] = "usable_complete"
    elif not results:
        summary["benchmark_verdict"] = "unusable_no_overlap"
    elif len(results) < len(models):
        summary["benchmark_verdict"] = "usable_partial"
    elif any(not item.get("benchmark_usable") for item in results):
        summary["benchmark_verdict"] = "needs_drift_review"
    _write_model_suite_source_data(summary, out_dir)
    _write_model_suite_markdown(summary, out_dir)
    figure_paths = _plot_model_suite_numeric(summary, out_dir)
    summary["human_report"] = str((out_dir / "summary.md").resolve())
    summary["figure_outputs"] = [str(Path(path).resolve()) for path in figure_paths]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_json": str((out_dir / "summary.json").resolve()),
                "summary_md": summary["human_report"],
                "figures": summary["figure_outputs"],
                "totals": summary["totals"],
            },
            indent=2,
        )
    )
    return 0 if results and summary["totals"]["failed"] == 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Canonical model-suite GPU/NPU artifact workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    default_registry = "benchmarks/model_zoo/registry.json"
    default_suite = "benchmarks/model_zoo/suites/canonical_models.json"

    p_validate = sub.add_parser("validate", help="Validate model registry and suite metadata")
    p_validate.add_argument("--registry", default=default_registry)
    p_validate.add_argument("--suite", default=default_suite)
    p_validate.set_defaults(func=cmd_validate)

    p_plan = sub.add_parser("plan", help="Print the canonical model experiment and figure plan")
    p_plan.add_argument("--registry", default=default_registry)
    p_plan.add_argument("--suite", default=default_suite)
    p_plan.add_argument("--out", default=None)
    p_plan.set_defaults(func=cmd_plan)

    p_collect = sub.add_parser("collect", help="Collect model artifacts for GPU reference or NPU bridge")
    p_collect.add_argument("--registry", default=default_registry)
    p_collect.add_argument("--suite", default=default_suite)
    p_collect.add_argument("--out", required=True)
    p_collect.add_argument("--role", choices=["gpu-reference", "npu-bridge"], required=True)
    p_collect.add_argument("--device", default="auto", help="cuda/cpu/auto. auto=cuda for GPU reference, cpu for bridge preamble.")
    p_collect.add_argument("--adapter", default=None, help="Required for --role npu-bridge in formal runs")
    p_collect.add_argument("--input-root", default=None, help="GPU reference artifact root whose saved inputs should be reused")
    p_collect.add_argument(
        "--strict-input-root",
        action="store_true",
        help="Fail if any suite model is missing from --input-root instead of skipping unavailable models",
    )
    p_collect.add_argument("--model-id", action="append", default=[])
    p_collect.add_argument("--model-cache", default="~/.cache/torchbridgebench/models")
    p_collect.add_argument("--seed", type=int, default=20260624)
    p_collect.add_argument(
        "--time-budget-seconds",
        type=float,
        default=None,
        help="Stop starting new models once the collection stage reaches this wall-clock budget",
    )
    p_collect.add_argument("--max-models", type=int, default=None, help="Maximum number of models to execute from the selected suite")
    p_collect.add_argument("--no-pretrained", action="store_true", help="Smoke/debug only; formal runs should use pretrained weights")
    p_collect.add_argument("--allow-cpu-fallback", action="store_true", help="Allow GPU reference smoke collection without CUDA")
    p_collect.add_argument(
        "--isolate-models",
        action="store_true",
        help="Run each selected model in a fresh process to prevent cross-framework state leakage",
    )
    p_collect.add_argument("--keep-going", action="store_true")
    p_collect.set_defaults(func=cmd_collect)

    p_compare = sub.add_parser("compare", help="Compare GPU reference and NPU bridge model artifacts")
    p_compare.add_argument("--registry", default=default_registry)
    p_compare.add_argument("--suite", default=default_suite)
    p_compare.add_argument("--gpu-reference", required=True)
    p_compare.add_argument("--npu-bridge", required=True)
    p_compare.add_argument("--out", required=True)
    p_compare.add_argument("--atol", type=float, default=1e-5)
    p_compare.add_argument("--rtol", type=float, default=1e-5)
    p_compare.add_argument("--cosine-threshold", type=float, default=0.999)
    p_compare.add_argument("--aligned-cosine", type=float, default=0.9999)
    p_compare.add_argument("--usable-cosine", type=float, default=0.99)
    p_compare.add_argument("--aligned-p95", type=float, default=1e-2)
    p_compare.add_argument("--usable-p95", type=float, default=5e-2)
    p_compare.set_defaults(func=cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
