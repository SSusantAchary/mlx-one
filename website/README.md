# MLX-One documentation website

The site is built with [Zensical](https://zensical.org), the same documentation
engine as the supplied TurboQuant-MLX reference. Its configured GitHub Pages
address is `https://ssusantachary.github.io/mlx-one/`.

## Preview locally

Use Python 3.12 for the documentation build. These tools are separate from
MLX-One’s runtime dependencies; building docs never loads a model or imports MLX.

```bash
python3.12 -m venv .venv-docs
source .venv-docs/bin/activate
python -m pip install -r website/requirements.txt
python website/prepare.py
cd website
zensical serve
```

Open the local address printed by the server. After changing an existing source
document, run `python website/prepare.py` again from the repository root; Zensical
reloads the generated page. Edits under `website/pages/` own site-specific guides
and the home page. Never edit `.generated/` or `dist/`.

## Build and check

From the repository root, with the docs environment active:

```bash
python -m unittest discover -s website/tests -v
python website/build.py
python website/check.py
```

The build stages the authoritative Markdown, builds HTML and checks local page,
asset and heading links. The output is `website/dist/`. The generated
`source-manifest.json` records the source commit, every input document/image
SHA-256 and the source-to-page mapping. No benchmark JSON or evidence is rewritten.

## Maintain content once

| Content | Maintained source |
| --- | --- |
| Home and short task guides | `website/pages/` |
| Current supported models | `src/mlx_one/data/model_support_release_20261108.json`, `src/mlx_one/data/batch_1_sub_6b.json` and their verified result files |
| Release approval and publication state | `docs/releases/0.1.1.json`, with notes in `docs/releases/0.1.1.md` |
| Support and known limits / historical audit | `status_models_testing.md` / `docs/qualification/history/model-status-before-20261108.md` |
| Installation, APIs, architecture and experiments | Existing Markdown under `docs/` |
| Family implementation guides | `src/mlx_one/models/*/*/README.md`, discovered automatically |
| All-modality performance page | Performance sections extracted from the current README |
| Coverage, explorer, history and performance dashboard | `website/insights.py`, versioned manifests, benchmark JSON and Git history |
| Detailed performance | Existing `docs/benchmarks/**/performance.md` reports |
| Citation | `CITATION.bib` |
| Navigation, theme and deployment URL | `website/zensical.toml` |

`website/prepare.py` maps the selected existing docs to friendly site URLs.
References to published pages become internal links. Original images are copied
unchanged. Supporting source files, JSON evidence and research reports outside
the selected site navigation link to the exact Git commit on GitHub. Code-fenced
examples are preserved. Every authored page links back to its maintained source.

To add a top-level guide, add its source/destination to `PAGES` in `prepare.py`
and add the destination to `nav` in `zensical.toml`. New model-family READMEs
automatically get pages and search entries; the canonical model index remains
their navigation entry point.

The homepage and model workflow guide install from the current source checkout.
Dated `0.1.1` installation examples identify publication as pending until its
artifacts are uploaded. The release record retains the last published version
separately. Historical measurements keep their original version
and workload identities. `README.pypi.md` is generated from the main README with
absolute links by `python scripts/release/pypi_readme.py`; use `--check` to verify it.

## Publish with GitHub Pages

1. In repository **Settings → Pages**, choose **GitHub Actions** as the source.
2. Merge the reviewed website changes into `main`.
3. The **Documentation site** workflow builds and publishes the site. It also
   runs on pull requests, but pull requests only build and validate.
4. Use **Actions → Documentation site → Run workflow** to rebuild manually.

Restrict the `github-pages` environment to the `main` branch. Keep deployment
permissions on the deploy job; the build job only needs repository read access.
The existing project CI and package dependencies remain unchanged.

### If the live page shows the README instead of the documentation

A plain README page without the navigation and search interface is the legacy
Jekyll/branch deployment, not the generated Zensical site. In **Settings → Pages**,
set **Build and deployment → Source** to **GitHub Actions**. Then rerun the failed
**Documentation site** workflow, or merge the reviewed site update to trigger it.
Check that both the build and deploy jobs succeed.

The `Configure GitHub Pages` step can return `404 Not Found` while Pages is still
configured for branch deployment. The default workflow token cannot switch that
repository setting by adding `enablement: true`; choose the Actions source in
repository settings instead.

## Coverage and progress dashboard

`website/pages/insights.md` owns the layout; `website/support.py` verifies the current
dated release catalog and the current-source checkpoint catalog against the
native registry, saved results, functional gates and installed execution receipts.
Result checksums and checkpoint identities must match. The homepage counters, all
seven modality cards and the primary model explorer use this catalog. A failed
quality diagnostic stays visible; it does not become a passed benchmark. The
release record reads the dated maintainer decision and keeps approval separate
from publication. Building the site does not publish the package.

`website/insights.py` also builds
`assets/data/insights.json` from five existing catalogs (text, VLM, audio, OCR and
native tasks), published benchmark summaries and the repository history. The
homepage modality cards count distinct registered native model types from
`models/registry.py`, with OCR metadata from `ocr/registry.py`. Each type is counted
once under its primary modality; checkpoint variants do not increase the count.
Classification has its own card, alongside the other six modality categories. This includes
retrieval and segmentation, which are outside the five checkpoint catalogs.
Registry metadata is evaluated in isolation without importing the MLX-One package
or model runtime. No model download, package release, network request or benchmark
execution is needed to generate the dashboard data.

The primary explorer shows 50 verified checkpoint profiles covering 45 distinct
architecture types. Availability labels distinguish the 40-type approved 0.1.1
cohort from ten additional source-checkout profiles awaiting the next package
release decision. BERT and XLM-R checkpoint variants count once per type.
The public workflow guide is staged under `models/workflows/`; archived evidence
paths retain their original identities. The historical explorer
retains the older checkpoint/task/precision profiles and unique pinned revisions;
its statuses preserve blockers, WIP and experimental scopes for those exact
profiles. Missing historical release-qualification fields remain unknown. No raw
manifest status or benchmark result is rewritten.

The current coverage chart uses first-parent native registry snapshots, with
monthly rows retaining their actual source dates and commits. Functional campaign
admissions are a separate dated series: the November 8 record covers 40 types,
and the November 10 record covers 45. These count architecture types with passing
primary workflows, not checkpoint variants. The older 23/7/10 audit is not plotted as an admission point.

The historical profile curve uses first-parent snapshots of the versioned manifests. Its
first-recorded and first-smoke dates are when evidence reached the branch, not
original test execution or general support admission dates. Counts may change
when profiles are removed or replaced. GitHub Actions checks out full history;
local shallow clones display a history limitation. Weekly milestones use
first-parent commit dates and dated changelog headings.

Performance cohorts retain distinct units, workload identities, precision,
hardware/software details, failures and source SHA-256 values. Missing metrics
stay null. MLXBoost's flag comparison uses existing complete HTTP SSE wall-time
rates, with its original request/process counts and experimental limitations;
it is not a direct-model or kernel speedup comparison.

The current explorer filters by model/task text, modality and precision; the
historical explorer also filters by its original evidence status. Native browser SVG
and CSS render charts without new frontend dependencies. The complete catalog
also renders in the HTML for users without JavaScript. All original evidence
links are pinned to the build's source commit, and `source-manifest.json` records
every input hash. The Pages workflow refreshes the site when its data inputs
change. Merge or push this website branch through your normal review workflow;
only `main` deploys to GitHub Pages.
