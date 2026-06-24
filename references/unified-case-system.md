# Unified Case System

TorchBridgeBench uses one benchmark case identity across GPU reference runs,
NPU bridge runs, reports, plots, and paper metrics.

## Canonical Identity

The canonical case id is the `id` field inside each
`benchmarks/v1.0.0/cases/**/*.json` file.

Example:

```text
bench_v1.0.0/L1/conv/conv2d_fp32
```

Do not introduce parallel GPU-only ids such as
`bench_v1.0.0/L1/conv2d/001` for new experiments. Those ids can be imported
only through an explicit mapping file.

## Canonical Source Of Truth

For v1.0.0, the source of truth is:

```text
benchmarks/v1.0.0/
├── manifest.json
├── suites/*.json
└── cases/**/*.json
```

Both GPU reference collection and NPU bridge evaluation must execute the same
case JSON code from the same suite JSON. This guarantees that case id, seed,
model shape, dtype policy, optimizer settings, activation channels, gradient
channels, and task metrics have the same intended semantics.

## Artifact Roles

`gpu-reference` artifacts are generated on a CUDA host by native PyTorch. They
are the formal source baseline for paper numeric metrics.

`npu-bridge` artifacts are generated on an Ascend host by executing the same
case under a bridge adapter.

`local-pair` reports compare local source execution with local adapter target
execution. They are useful diagnostics, but they are not formal GPU-vs-NPU
numeric evidence.

## Required Artifact Schema

Each GPU reference run writes:

```text
<out>/
├── manifest.json
├── summary.json
└── cases/<case-slug>/
    ├── reference.json
    └── artifacts/gpu_reference/*.npy
```

`reference.json` must contain:

- `schema_version`: `tbbcc.gpu_reference.case.v1`
- `case_id`: canonical benchmark case id
- `case_sha256`: hash of the exact case JSON
- `suite_id`
- `status`: `passed`, `failed`, or `skipped`
- `environment`: Python, PyTorch, CUDA, cuDNN, GPU, driver, platform
- `channels.result`: normalized output tensor/tree summary
- `channels.activations`: optional activation summaries
- `channels.gradients`: optional gradient summaries
- `channels.task_metrics`: optional task metric summaries

Tensor artifacts are stored as `.npy` files. JSON summaries hold shape, dtype,
sample, hash, numeric statistics, and relative artifact paths.

## Mapping Legacy GPU Artifacts

Existing GPU artifacts with non-canonical ids must not be merged silently. Use
a reviewed mapping file when importing them:

```json
{
  "schema_version": "tbbcc.case_mapping.v1",
  "source": "legacy-gpu-ground-truth",
  "target_suite": "bench_v1.0.0/all_noop",
  "mappings": [
    {
      "canonical_case_id": "bench_v1.0.0/L1/conv/conv2d_fp32",
      "legacy_case_id": "bench_v1.0.0/L1/conv2d/001",
      "equivalence": "exact",
      "checked_fields": ["operator", "shape", "dtype", "seed", "parameters", "output", "gradients"]
    }
  ]
}
```

If there is no direct id overlap and no mapping file, the system must report
`mapping_required=true` and refuse to produce GPU-vs-NPU numeric conclusions.

## Practical Workflow

On the GPU host:

```bash
python scripts/tbbcc_gpu_reference.py \
  --suite benchmarks/v1.0.0/suites/all_noop.json \
  --out reports/gpu_reference_all_noop \
  --device cuda
```

On the NPU host, run the bridge evaluation against the same suite:

```bash
python scripts/tbbcc.py eval-suite \
  --suite benchmarks/v1.0.0/suites/all_noop.json \
  --out reports/torch4ms_eval
```

Only artifacts sharing the same canonical `case_id` and compatible
`case_sha256` should be compared as formal numeric evidence.
