#!/usr/bin/env python3
"""Prepare persistent model-weight caches for TBBCC model-suite runs.

The runner uses:
- TORCH_HOME=$TBBCC_MODEL_CACHE/torch for torchvision checkpoints
- HF_HOME=$TBBCC_MODEL_CACHE/hf for Hugging Face and diffusers checkpoints

Preferred use is to run this directly on the NPU server so the persistent cache
is created in place. If the NPU server cannot access the model hosts, run the
same command on a networked PC, archive the whole cache directory, and unpack it
to the same path on the NPU server.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


TORCHVISION_WEIGHTS = {
    "resnet18_imagenet_224": ("torchvision.models", "ResNet18_Weights", "IMAGENET1K_V1"),
    "mobilenetv2_imagenet_224": ("torchvision.models", "MobileNet_V2_Weights", "IMAGENET1K_V1"),
    "squeezenet1_1_imagenet_224": ("torchvision.models", "SqueezeNet1_1_Weights", "IMAGENET1K_V1"),
    "shufflenet_v2_x1_0_imagenet_224": ("torchvision.models", "ShuffleNet_V2_X1_0_Weights", "IMAGENET1K_V1"),
    "efficientnet_b0_imagenet_224": ("torchvision.models", "EfficientNet_B0_Weights", "IMAGENET1K_V1"),
    "vgg11_bn_imagenet_224": ("torchvision.models", "VGG11_BN_Weights", "IMAGENET1K_V1"),
}

HF_REPOS = {
    "minimind3o_moe_seq128": "jingyaogong/minimind-3o-moe",
    "ddpm_cifar10_unet_32": "google/ddpm-cifar10-32",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _suite_model_ids(suite_path: Path) -> list[str]:
    payload = _load_json(suite_path)
    return [str(item) for item in payload.get("model_ids", [])]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_torchvision(model_id: str) -> dict[str, Any]:
    import importlib

    module_name, enum_name, member_name = TORCHVISION_WEIGHTS[model_id]
    module = importlib.import_module(module_name)
    weights = getattr(getattr(module, enum_name), member_name)
    weights.get_state_dict(progress=True, check_hash=True)
    return {
        "model_id": model_id,
        "status": "downloaded",
        "source": "torchvision",
        "locator": f"{module_name}.{enum_name}.{member_name}",
    }


def _download_hf(model_id: str, hf_home: Path) -> dict[str, Any]:
    from huggingface_hub import snapshot_download

    repo_id = HF_REPOS[model_id]
    snapshot_path = snapshot_download(repo_id=repo_id, cache_dir=str(hf_home))
    return {
        "model_id": model_id,
        "status": "downloaded",
        "source": "huggingface_hub",
        "locator": repo_id,
        "snapshot_path": snapshot_path,
    }


def _cached_torch_checkpoints(torch_home: Path) -> list[dict[str, Any]]:
    checkpoint_dir = torch_home / "hub" / "checkpoints"
    if not checkpoint_dir.exists():
        return []
    out = []
    for path in sorted(checkpoint_dir.glob("*")):
        if path.is_file():
            out.append(
                {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Download model weights into a persistent TBBCC cache.")
    parser.add_argument("--suite", default="benchmarks/model_zoo/suites/mixed_alignment_30min.json")
    parser.add_argument("--cache-dir", default=os.environ.get("TBBCC_MODEL_CACHE", "tbbcc_model_cache"))
    parser.add_argument("--model-id", action="append", help="Download only the given model id. May be repeated.")
    parser.add_argument("--include-optional-torchvision", action="store_true")
    parser.add_argument("--skip-hf", action="store_true", help="Only download torchvision weights.")
    parser.add_argument("--skip-torchvision", action="store_true", help="Only download Hugging Face/diffusers weights.")
    args = parser.parse_args()

    root = Path(args.cache_dir).expanduser().resolve()
    torch_home = root / "torch"
    hf_home = root / "hf"
    timm_home = root / "timm"
    for path in (torch_home, hf_home, timm_home):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["TIMM_HOME"] = str(timm_home)

    model_ids = args.model_id or _suite_model_ids(Path(args.suite))
    if args.include_optional_torchvision:
        model_ids = list(dict.fromkeys([*model_ids, *TORCHVISION_WEIGHTS.keys()]))

    results: list[dict[str, Any]] = []
    for model_id in model_ids:
        try:
            if model_id in TORCHVISION_WEIGHTS and not args.skip_torchvision:
                results.append(_download_torchvision(model_id))
            elif model_id in HF_REPOS and not args.skip_hf:
                results.append(_download_hf(model_id, hf_home))
            else:
                results.append({"model_id": model_id, "status": "skipped", "reason": "no downloader enabled"})
        except Exception as exc:  # Keep going so partial caches can still be archived.
            results.append({"model_id": model_id, "status": "failed", "error": repr(exc)})

    manifest = {
        "schema_version": "tbbcc.weight_cache.v1",
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "cache_dir": str(root),
        "torch_home": str(torch_home),
        "hf_home": str(hf_home),
        "timm_home": str(timm_home),
        "models": results,
        "torch_checkpoints": _cached_torch_checkpoints(torch_home),
    }
    manifest_path = root / "tbbcc_weight_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"manifest: {manifest_path}")
    return 1 if any(item.get("status") == "failed" for item in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
