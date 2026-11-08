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

Published installation commands retain `0.1.0a1` until a newer package is
published. The source candidate and citation may report `0.1.1`; a source version
does not establish a PyPI release. Historical measurements retain their original
version and workload identities.

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

`website/pages/insights.md` owns the layout; `website/insights.py` builds
`assets/data/insights.json` from five existing catalogs (text, VLM, audio, OCR and
native tasks), published benchmark summaries and the repository history. The
homepage modality cards count distinct registered native model types from
`models/registry.py`, with OCR metadata from `ocr/registry.py`. Each type is counted
once under its primary modality; checkpoint variants do not increase the count.
Classification types appear alongside the six main modality cards. This includes
retrieval and segmentation, which are outside the five checkpoint catalogs.
Registry metadata is evaluated in isolation without importing the MLX-One package
or model runtime. No model download, package release, network request or benchmark
execution is needed to generate the dashboard data.

Dashboard counts are checkpoint/task/precision profiles and unique pinned
revisions; the homepage counts registered model types. Status labels preserve blockers, WIP and experimental
scopes. Missing release-qualification fields remain unknown. No manifest status
or benchmark result is rewritten.

The growth curve uses first-parent snapshots of the versioned manifests. Its
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

The explorer filters by text, modality, status and precision. Native browser SVG
and CSS render charts without new frontend dependencies. The complete catalog
also renders in the HTML for users without JavaScript. All original evidence
links are pinned to the build's source commit, and `source-manifest.json` records
every input hash. The Pages workflow refreshes the site when its data inputs
change. Merge or push this website branch through your normal review workflow;
only `main` deploys to GitHub Pages.
