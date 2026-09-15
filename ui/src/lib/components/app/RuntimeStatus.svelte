<script lang="ts">
  import type { RuntimeInfo } from '$lib/types/api';
  let { runtime, context = 0 } = $props<{ runtime: RuntimeInfo | null; context?: number }>();
  const fmt = (value?: number) => value == null ? '—' : value >= 1e9 ? `${(value/1e9).toFixed(1)} GB` : `${(value/1e6).toFixed(0)} MB`;
</script>
<footer aria-label="Runtime statistics">
  <span class="live"></span><b>MLX</b><span>{runtime?.chip ?? 'Apple Silicon'}</span>
  <span>{runtime?.tokens_per_second?.toFixed(1) ?? '—'} tok/s</span>
  <span>TTFT {runtime?.time_to_first_token_ms?.toFixed(0) ?? '—'} ms</span>
  <span>{fmt(runtime?.memory_used_bytes)}</span>
  <span>{runtime?.context_used ?? 0} / {context || '—'} ctx</span>
  <span>slots {runtime?.scheduler?.active_slots ?? 0}/{runtime?.scheduler?.parallel_slots ?? '—'}</span>
  <span>queue {runtime?.scheduler?.queue_depth ?? 0}</span>
  <span>KV {runtime?.server?.cache_type_k ?? '—'}/{runtime?.server?.cache_type_v ?? '—'}</span>
  <span>prefix {runtime?.prompt_cache_hit ? `${runtime.reused_prompt_tokens ?? 0} hit` : 'miss'}</span>
  {#if runtime?.drafted_tokens}
    <span>MTP {runtime.accepted_draft_tokens ?? 0}/{runtime.drafted_tokens}</span>
  {/if}
</footer>
<style>footer{height:2.6rem;border-top:1px solid var(--line);display:flex;align-items:center;gap:1.15rem;padding:0 1.2rem;color:var(--muted);font-size:.72rem;overflow:auto;white-space:nowrap}b{color:var(--ink)}.live{width:.45rem;height:.45rem;border-radius:50%;background:var(--accent);box-shadow:0 0 0 .2rem var(--accent2)}</style>
