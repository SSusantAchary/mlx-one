# MLX-One branding

The maintainer supplied the black-and-orange design on 15 November 2026.

- `mlx-one.png` is the supplied full wordmark, copied without image changes. The
  README, PyPI description source and documentation homepage use this file.
- `ui/static/mlx-one-mark.png` is the compact, white-background application mark,
  derived with built-in imagegen for readability in dark and light toolbars.
  The chat sidebar/favicon and website header/favicon use this mark. The website
  stages it as `assets/brand/mark-orange.png`; the chat build copies it into
  `src/mlx_one/ui/dist/`.

The original blue three-bar wordmark and earlier application mark remain in the
historical `v0.1.4` tag. Post-release branding changes do not replace that tag or
its published PyPI distribution bytes. Current Git installs include the updated
mark; a future package release is required to update the installed PyPI icon.

## Compact-mark edit prompt

Tool: built-in imagegen. Input: the maintainer-supplied full logo.

> Edit the provided existing logo into a compact application icon. Retain ONLY
> the angular black folded-ribbon M symbol and its two orange right-side shapes,
> exactly as in the source. Remove the 'mlx-one' text wordmark and surrounding
> empty area. Preserve the original symbol shape and proportions. Center it on
> a clean solid WHITE square background with about 7 percent padding at the
> left/right edges. White opaque background is required so the black symbol
> remains readable in dark browser toolbars. Crisp clean edges, no transparency,
> no stray pixels, no red fringe, no shadows, no border, no new shapes or text.
> The entire symbol fits inside the canvas; use the original colors and folded
> gray center. This is a close crop/extraction, not a redesign.

## Verification

The supplied wordmark was verified byte-for-byte. UI source and packaged mark
match. Fourteen UI tests, nine chat browser scenarios, 29 website tests and
24 asset/README regression tests passed. Svelte reports no errors or warnings;
strict website links and desktop/mobile branding checks pass. These checks
cover branding and publication-state changes, without rerunning model inference.

| Asset | SHA-256 |
| --- | --- |
| `mlx-one.png` | `9da8505434d2f6ae3e8f5c88734a9d95c339f0462807bf0615ecf1d42710b6e3` |
| `ui/static/mlx-one-mark.png` | `1bd9522170697aab572761e4276f5b88198e494fc13b00b81c764185bebebdaf` |
