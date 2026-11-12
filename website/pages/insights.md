---
title: Models, coverage and progress
---

# Models, coverage and progress

Explore **45 native model types across seven modalities**, represented by
**50 verified checkpoint profiles**. These include T5 text-to-text, supervised
ViT classification, Kokoro speech and multilingual retrieval alongside the
original 40-type release cohort.
[Model workflows and examples](../../docs/models/batch-1-sub-6b/README.md).

<div data-insights-root data-source="../assets/data/insights.json"></div>

<!-- insights:summary -->

<p class="insights-provenance" data-insights-provenance>Loading source provenance…</p>

## Supported model explorer

Every row identifies a checkpoint profile, its architecture type, pinned revision,
precision, tested primary operations and saved result. Multiple checkpoints can
share one model type. Search by model, task or precision.

<div class="insights-controls">
<label>Find a model or task<input type="search" data-support-search placeholder="Try GPT-OSS, speech, SAM3 or Q8" disabled></label>
<label>Modality<select data-support-modality disabled><option value="">All modalities</option></select></label>
<label>Precision<select data-support-precision disabled><option value="">All precisions</option></select></label>
<button type="button" data-support-reset disabled>Reset filters</button>
</div>

<p data-support-count role="status" aria-live="polite"></p>
<div class="insights-table" data-support-table></div>

<details markdown="1" class="insights-fallback">
<summary>All verified checkpoint profiles (also available without JavaScript)</summary>

<!-- insights:supported -->

</details>

**Supported** means the selected checkpoint passes native task execution and
relevant interface checks, backed by model tests, regressions and packaging
checks. Accuracy and speed remain measurements of the recorded workload.
The availability label reads the current package release decision, keeping
approval and PyPI publication separate. All profiles are included in the published **0.1.2** release;
[the release record](../../docs/releases/0.1.2.json) records the verified artifacts.
[Install the release](../getting-started.md#install-from-pypi).

[Quality and scope limitations](../models/support.md#known-quality-and-scope-limitations)
include Florence page OCR accuracy, NER sample accuracy, LFM speech precision and
the optional SAM MPS reference discrepancy. Optional failed tasks are excluded.
ViT classification passes while its fixed hidden-state diagnostic remains failed;
Kokoro is qualified for bounded `af_heart` English speech. The
[workflow guide](../../docs/models/batch-1-sub-6b/README.md) records these limits.

## Model coverage over time

<div class="insights-panel" data-registration-chart aria-label="Registered model types and dated functional admission records"></div>
<div class="insights-table" data-registration-table></div>

Registered types follow dated Git snapshots. Functional admissions are a separate
series: the 8 November record covers 40 model types and the 10 November record
covers 45. Ten additional checkpoint profiles add five architecture types;
variants do not inflate the type count. The historical 23/7/10 audit used a
different policy and is not connected to this series. Monthly rows retain the
actual registry snapshot date.

## Performance dashboard

Choose a measurement cohort. Text, vision, speech and OCR use their original
units, timing scopes and hardware. Missing or failed results remain visible.

<div class="insights-controls"><label>Measurement cohort<select data-performance-cohort disabled></select></label></div>
<p data-performance-note></p>
<div class="insights-panel" data-performance-chart></div>
<div class="insights-table" data-performance-table></div>

The [all-modality reports](../performance/index.md) retain detailed methods,
limitations and historical GPT-OSS HTTP context. Each measured row links to its
original JSON and includes hardware, software, workload and quality details.

## MLXBoost comparison

MLXBoost is **experimental and opt-in**. These historical **complete HTTP SSE
request rates** describe the recorded workloads; direct-model decode rates and
kernel performance are separate measurements.

<div class="insights-panel" data-boost-chart></div>
<div class="insights-table" data-boost-table></div>
<p data-boost-note></p>

[Scope, numerical failures and rollback](../experimental/mlxboost.md)

## Weekly development milestones

Weeks run Monday–Sunday using Git's recorded commit dates. Activity and changelog
milestones describe development; model support and publication have their own
dated records.

<div data-weekly-milestones></div>

## Historical checkpoint profiles

The earlier five catalogs contain checkpoint/task/precision profiles, including
optional tasks and alternative revisions. Their counting unit and historical
labels differ from the current checkpoint profiles above. Original blockers remain
visible for those exact profiles.

<details markdown="1" class="insights-history">
<summary>Explore the historical catalogs and profile history</summary>

<div class="insights-controls">
<label>Find a historical profile<input type="search" data-model-search placeholder="Try Qwen, ASR, Q4 or blocked" disabled></label>
<label>Modality<select data-model-modality disabled><option value="">All modalities</option></select></label>
<label>Historical evidence status<select data-model-status disabled><option value="">All statuses</option></select></label>
<label>Precision<select data-model-precision disabled><option value="">All precisions</option></select></label>
<button type="button" data-model-reset disabled>Reset filters</button>
</div>

<p data-model-count role="status" aria-live="polite"></p>
<div class="insights-table" data-model-table></div>
<div data-status-definitions></div>

<!-- insights:catalog -->

<div class="insights-panel" data-growth-chart aria-label="Historical checkpoint profile catalog"></div>
<div data-history-table></div>

Dates show when profiles reached first-parent Git history. These are separate
from model implementation, campaign execution and release dates.

</details>

[Archived pre-campaign model audit](../history/model-support.md)

## Sources and refresh

[Dashboard JSON](../assets/data/insights.json) ·
[source manifest](../source-manifest.json) ·
[current support inventory](../models/support.md) ·
[changelog](../reference/changelog.md).

The site refreshes on each GitHub Pages build. It reads saved results without
downloading a model or running inference. Both current and historical inventories
remain available without JavaScript.
