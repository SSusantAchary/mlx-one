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
