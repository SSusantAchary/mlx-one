<script lang="ts">
  import { Box } from '@lucide/svelte';
  import type { ModelInfo } from '$lib/types/api';
  let { model = '', info }: { model?: string; info?: ModelInfo['mlx'] } = $props();
  const quant = $derived(info?.quantization?.mode ? String(info.quantization.mode).toUpperCase() : info?.quantization?.bits ? `${info.quantization.bits}-bit` : null);
</script>
<span class="model-badge" title={model}>
  <Box size={14} /><span class="name">{model || 'No model connected'}</span>
  {#if info?.parameters}<span class="tag">{info.parameters >= 1e9 ? `${(info.parameters / 1e9).toFixed(1)}B` : info.parameters >= 1e6 ? `${Math.round(info.parameters / 1e6)}M` : `${info.parameters} params`}</span>{/if}
  {#if quant}<span class="tag">{quant}</span>{/if}
  {#if info?.context_length}<span class="tag">{info.context_length.toLocaleString()} ctx</span>{/if}
  <span class="tag">MLX</span>
</span>
<style>
  .model-badge{display:inline-flex;align-items:center;gap:.4rem;padding:.3rem .5rem;border-radius:.5rem;background:var(--hover);color:var(--muted);font-size:.72rem;max-width:100%;min-width:0;vertical-align:middle}.name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.tag{border:1px solid var(--line);border-radius:.4rem;padding:0 .3rem;white-space:nowrap;font-family:ui-monospace,monospace;font-size:.65rem}.model-badge :global(svg){flex-shrink:0}@media(max-width:700px){.tag{display:none}}
</style>
