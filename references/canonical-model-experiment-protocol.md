# Canonical Model Experiment Protocol

This protocol defines the focused numerical experiments for the paper. It
replaces the idea of treating all 175 benchmark cases as equally important
GPU-vs-NPU numerical comparisons.

## Roles

- Broad benchmark: `benchmarks/v1.0.0/suites/all_noop.json`
  - Used for compatibility coverage, failure taxonomy and anti-regression.
  - Not the primary source for layer-wise numerical claims.
- Canonical model suite: `benchmarks/model_zoo/suites/canonical_models.json`
  - Used for GPU-vs-NPU numerical characterization and case-study figures.

## Models

The initial canonical suite contains:

1. ResNet-18: standard residual CNN.
2. MobileNetV2: lightweight depthwise-separable CNN.
3. ViT-Tiny: attention/Transformer path.
4. UNet-Small: encoder-decoder segmentation and short post-training path.

## GPU Reference Run

For each model:

1. Load pretrained weights from `TBBCC_MODEL_CACHE` or download them according
   to `benchmarks/model_zoo/README.md`.
2. Save checkpoint checksum and preprocessing metadata.
3. Run native PyTorch on CUDA.
4. Use deterministic probe inputs, stored as artifacts and reused by NPU runs.
5. Capture:
   - final logits/output
   - named layer activations
   - named layer gradients for post-training mode
   - task metrics
   - latency, throughput and peak memory
   - loss curve for short training/fine-tuning mode

## NPU Bridge Run

For each model:

1. Load the same checkpoint and the exact same input artifacts.
2. Run through the bridge adapter on Ascend NPU.
3. Use the same hook names and channel schema as the GPU reference run.
4. Capture the same outputs, activations, gradients, task metrics and runtime
   metrics.

## Required Metrics

Inference:

- final output cosine, MAE, max error
- FNE curve: cosine per named layer
- first divergence layer
- task metric drift: top-1/top-5, mIoU, Dice or task-specific metric
- latency, throughput and peak memory

Post-training:

- GC curve: gradient cosine per named layer
- loss curve DTW distance
- final metric relative deviation
- training-path notes for AMP, optimizer, BatchNorm and Dropout

## Figure Plan

Main paper figures should use selected model panels:

- ResNet-18: layer-wise FNE curve, first divergence layer.
- MobileNetV2: depthwise block drift panel.
- ViT-Tiny: attention block drift panel.
- UNet-Small: loss curve + GC curve panel.

The full 175-case benchmark should contribute failure-taxonomy and coverage
figures, not the main layer-wise numerical figure.

## Acceptance Criteria

- Registry validates against `tbbcc.model_zoo.registry.v1`.
- Every model has a weight locator and explicit external storage policy.
- Every model defines activation and gradient hook names.
- GPU and NPU artifacts use shared inputs and checkpoint hashes.
- The comparison report distinguishes true numerical drift from input mismatch,
  missing hooks, unsupported operators and harness failures.
