---
title: Coverage and progress
---

# Coverage and progress

Explore the checkpoints, evidence and development history already in MLX-One.
This page refreshes from the repository manifests and benchmark JSON on each
site build. It does not download models or run new benchmarks.

<div data-insights-root data-source="../assets/data/insights.json"></div>

<!-- insights:summary -->

<p class="insights-provenance" data-insights-provenance>Loading the source revision…</p>

## Model-support growth

<div class="insights-panel" data-growth-chart aria-label="Model evidence history"></div>

The curve counts **checkpoint/task/precision profiles in the five manifests**
and profiles with **scoped smoke evidence**. Candidates are included in the
catalog line. Dates mark when records reached the branch's first-parent history,
not the original test date, a release date, or the first implementation of an
architecture. Support that predates these manifests is not reconstructed.

<details class="insights-history"><summary>View historical counts and source commits</summary><div data-history-table></div></details>

## Model explorer

The catalog covers text, vision-language, audio, OCR and additional native task
profiles. It is not a count of every implemented architecture. Each row retains
its pinned revision, task, precision, raw status and evidence boundary.

<div class="insights-controls">
<label>Find a checkpoint or task<input type="search" data-model-search placeholder="Try Qwen, ASR, Q4 or blocked" disabled></label>
<label>Modality<select data-model-modality disabled><option value="">All modalities</option></select></label>
<label>Evidence status<select data-model-status disabled><option value="">All statuses</option></select></label>
<label>Precision<select data-model-precision disabled><option value="">All precisions</option></select></label>
<button type="button" data-model-reset disabled>Reset filters</button>
</div>

<p data-model-count role="status" aria-live="polite"></p>
<div class="insights-table" data-model-table></div>

<!-- insights:catalog -->

### What the labels mean

<div data-status-definitions></div>

**Measured performance is not release qualification.** A blocked profile stays
blocked even when a diagnostic run has numeric results. Architecture-only and
quantization tests do not establish that a full checkpoint has passed a smoke
test. A revision or interface with no recorded evidence stays unqualified.

## Performance dashboard

Choose a measurement cohort. Rates use different units and timing scopes across
text, vision, speech and OCR, so cohorts are never combined into one ranking.
Missing or failed results stay visible without zero-valued speed bars.

<div class="insights-controls"><label>Measurement cohort<select data-performance-cohort disabled></select></label></div>
<p data-performance-note></p>
<div class="insights-panel" data-performance-chart></div>
<div class="insights-table" data-performance-table></div>

The [all-modality reports](../performance/index.md) retain detailed methods,
limitations and the separate historical GPT-OSS HTTP context. Every measured row
below links to its original JSON; expand its details for hardware, software,
workload and quality metrics.

## MLXBoost comparison

MLXBoost is **experimental and opt-in**. These are historical **complete HTTP SSE
request rates**, not direct-model decode rates or evidence of faster kernels.
The default reference execution path is unchanged.

<div class="insights-panel" data-boost-chart></div>
<div class="insights-table" data-boost-table></div>
<p data-boost-note></p>

[Scope, numerical failures and rollback](../experimental/mlxboost.md)

## Weekly development milestones

Weeks run Monday–Sunday using Git's recorded commit dates. Commit activity and
dated changelog milestones describe development, not model certification or
package publication. Expand a week to inspect its commits.

<div data-weekly-milestones></div>

## Sources and refresh

[Download the dashboard JSON](../assets/data/insights.json) or inspect the
[source manifest](../source-manifest.json). These static files update on each
GitHub Pages build; no account or running inference server is required.

If interactive data cannot load, the full model catalog above remains available.
The [checkpoint inventory](../reference/checkpoints.md),
[performance reports](../performance/index.md), and
[changelog](../reference/changelog.md) provide the original documentation.
