# Model support and functional admission

MLX-One follows the practical MLX-LM admission approach: native model correctness,
real checkpoint task execution and working relevant interfaces. All **40 registered
model types** have passing selected primary workflows in the
[current support inventory](../status_models_testing.md). **0.1.1 is approved for
release on 8 November 2026; publication is pending.**
[Release notes](releases/0.1.1.md).

## What supported means

A support entry identifies a model type, exact checkpoint revision, storage and
execution precision, primary operations and saved real-checkpoint result. Model
architecture/state tests, relevant interfaces, native regressions and clean
package checks back the entry. Other sizes, quantizations, tasks and hardware
require evidence for any additional advertised claim.

The machine-readable source is
[`model_support_release_20261108.json`](../src/mlx_one/data/model_support_release_20261108.json).
It is checked against the native registry and saved result hashes during the
website build. Historical manifests retain their original statuses and do not
override this current inventory.

## Required engineering checks

- Correct config, weight names/shapes and quantization mapping; reject unsupported
  or malformed inputs.
- Forward shape/dtype and finite outputs; correct masks, routing and task state.
- Applicable cache prefill/continuation/reset and independent request state.
- Real pinned-checkpoint output for the named task: text, images, transcripts,
  waveform generation, embeddings, scores, masks or classifications.
- Working advertised Python/CLI/HTTP/streaming interfaces and relevant lifecycle
  checks. A task does not need an interface that is not advertised.
- Regression and clean distribution checks, native Metal execution where claimed,
  isolated imports, and retained licenses/notices.

Upstream MLX-LM asks for new model tests and actual generation output. Its template
requests reference output if CUDA is available; it does not prescribe a universal
corpus accuracy floor or blanket broad-reliability certification.
[Contribution guide](https://github.com/ml-explore/mlx-lm/blob/main/CONTRIBUTING.md) ·
[new-model template](https://github.com/ml-explore/mlx-lm/blob/main/.github/PULL_REQUEST_TEMPLATE/new_model.md).

## Quality, performance and optional scopes

CER/WER/F1/PPL measurements, reference comparisons, long-context work and optional
task variants remain valuable diagnostics or evidence for specific claims. They
are not universal support gates. Actual loading, invalid output, broken state or
interface defects still block the affected advertised workflow.

Changing the support policy never turns a failed benchmark into a passed
benchmark. See [repairs and preserved diagnostic limits](qualification/model-release-20261108/RESOLUTIONS.md).
MLXBoost, training, TQ3 variants and other optional experiments retain their own
status; model support does not admit every extension.

## Release and publication

Functional support, measured quality/performance and publication are separate
records. The [maintainer release decision](releases/0.1.1.json) approves 0.1.1;
publishing the package and deploying the site happen separately. Published
installation examples target 0.1.1 and identify publication as pending until upload.

## Historical status vocabulary

Older manifests use `ARCHITECTURE_IMPLEMENTED`, `MODEL_SMOKE_TESTED`,
`EXPERIMENTAL_SUPPORTED`, `USER_VALIDATED`, `WIP` and blocked/unknown labels. These
continue to describe their original exact profiles. They do not impose a blanket
experimental label on the current supported models. The older 23/7/10 model audit
is preserved in the [historical report](qualification/history/model-status-before-20261108.md).
