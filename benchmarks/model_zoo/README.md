# Canonical Model Suite

The canonical model suite is the primary GPU-vs-NPU numerical characterization
asset. It is separate from the 175-case generated benchmark:

- `benchmarks/v1.0.0/suites/all_noop.json` audits broad compatibility and
  failure modes.
- `benchmarks/model_zoo/registry.json` defines a small set of real models for
  deep numerical analysis: layer-wise FNE, layer-wise GC, task metrics,
  latency, memory and training-path drift.

## Selected Models

The initial suite contains four small models suitable for Ascend 910B memory
and for publication-quality case studies:

| model_id | Model | Role |
| --- | --- | --- |
| `resnet18_imagenet_224` | ResNet-18 | Primary CNN case study |
| `mobilenetv2_imagenet_224` | MobileNetV2 | Depthwise/lightweight CNN case |
| `vit_tiny_imagenet_224` | ViT-Tiny | Attention/Transformer path case |
| `unet_small_biosample_256` | UNet-Small | Segmentation and training-path case |

An optional second-batch suite is available at
`benchmarks/model_zoo/suites/torchvision_candidates_v1.json`. It uses only
torchvision weight locators to reduce Hugging Face download failures:

| model_id | Model | Role |
| --- | --- | --- |
| `squeezenet1_1_imagenet_224` | SqueezeNet1_1 | Fire-module CNN candidate |
| `shufflenet_v2_x1_0_imagenet_224` | ShuffleNetV2 x1.0 | Channel-shuffle candidate |
| `efficientnet_b0_imagenet_224` | EfficientNet-B0 | MBConv/SE candidate |
| `vgg11_bn_imagenet_224` | VGG11-BN | Plain deep CNN candidate |

## Weight Storage

Weights are not stored in this repository. Set:

```bash
export TBBCC_MODEL_CACHE=/path/to/tbbcc_models
```

Recommended sources:

- ResNet-18: `torchvision.models.ResNet18_Weights.IMAGENET1K_V1`
- MobileNetV2: `torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1`
- ViT-Tiny: `timm.create_model("vit_tiny_patch16_224", pretrained=True)`
- UNet-Small: use the team-adapted torch4ms checkpoint or a public small UNet
  checkpoint; record URL, checksum and license before final experiments.
- SqueezeNet1_1: `torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1`
- ShuffleNetV2 x1.0: `torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1`
- EfficientNet-B0: `torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1`
- VGG11-BN: `torchvision.models.VGG11_BN_Weights.IMAGENET1K_V1`

The current runner looks for the UNet checkpoint at:

```text
$TBBCC_MODEL_CACHE/unet_small_biosample_256/checkpoint.pt
```

If this file is absent, the runner can still perform a bootstrap smoke run with
deterministic initialized weights, but such a run is not acceptable for final
paper claims.

For final paper experiments, every downloaded checkpoint must have:

- source URL or library locator
- version
- SHA256 checksum
- license
- preprocessing recipe
- model cache path

## Probe Data

Probe datasets are also external. The repository should contain only metadata
and loading code, not dataset payloads.

### imagenet-style-probe

Use a small deterministic image-classification subset with ImageNet-compatible
normalization. Suggested size: 32-256 images. Store file lists and labels under
the configured model/data cache, not in Git.

### segmentation-probe

Use a small deterministic segmentation subset. Suggested size: 8-32 images and
masks. Record dataset source, license and preprocessing.

## Metrics

Inference metrics:

- final output cosine / MAE / max error
- layer-wise FNE curve
- first divergence layer
- top-1/top-5 or task metric drift
- latency, throughput and peak memory

Training/post-training metrics:

- layer-wise GC curve
- loss curve DTW distance
- final task metric relative deviation
- optimizer/AMP/BatchNorm/Dropout path notes

## Figure Candidates

The main paper should use a small number of figure panels from this suite:

- ResNet-18: layer-wise FNE curve and first divergence layer.
- MobileNetV2: operator-family drift around depthwise separable blocks.
- ViT-Tiny: attention block drift.
- UNet-Small: training loss curve plus gradient consistency curve.

The 175-case benchmark remains useful for failure-taxonomy plots and coverage
audits, but it should not be treated as the main GPU-vs-NPU numerical evidence.

## Commands

Validate the registry and suite:

```bash
python scripts/tbbcc_model_suite.py validate
```

Plan the evidence chain and figure candidates:

```bash
python scripts/tbbcc_model_suite.py plan --out reports/canonical_model_plan.json
```

GPU reference collection:

```bash
export TBBCC_MODEL_CACHE=/path/to/model_cache
python scripts/tbbcc_model_suite.py collect \
  --role gpu-reference \
  --device cuda \
  --out reports/canonical_models_gpu
```

Second-batch torchvision candidate collection:

```bash
python scripts/tbbcc_model_suite.py collect \
  --suite benchmarks/model_zoo/suites/torchvision_candidates_v1.json \
  --role gpu-reference \
  --device cuda \
  --out reports/torchvision_candidates_gpu
```

NPU bridge collection with shared GPU inputs:

```bash
python scripts/tbbcc_model_suite.py collect \
  --role npu-bridge \
  --adapter reports/torch4ms_eval/adapter.generated.json \
  --input-root reports/canonical_models_gpu \
  --time-budget-seconds 1800 \
  --out reports/canonical_models_torch4ms_npu
```

NPU collection skips models missing from `--input-root` by default. Use
`--strict-input-root` when a complete suite is required and missing GPU inputs
should fail the run.

The NPU-side numerical-alignment stage should stay within 30 minutes:

```bash
python scripts/tbbcc_model_suite.py collect \
  --role npu-bridge \
  --adapter reports/torch4ms_eval/adapter.generated.json \
  --input-root reports/canonical_models_gpu \
  --time-budget-seconds 1800 \
  --max-models 4 \
  --out reports/canonical_models_torch4ms_npu
```

GPU reference collection is outside this budget.

Comparison:

```bash
python scripts/tbbcc_model_suite.py compare \
  --gpu-reference reports/canonical_models_gpu \
  --npu-bridge reports/canonical_models_torch4ms_npu \
  --out reports/canonical_models_gpu_vs_npu
```
