<script lang="ts">
  import type { Settings } from '$lib/types/api';
  let { settings = $bindable(), onconnect, architecture, onbudgetchange }: { settings: Settings; onconnect: () => void; architecture?: string; onbudgetchange?: () => void } = $props();
</script>
<div class="settings">
  <label>Temperature <input type="number" min="0" max="2" step="0.1" bind:value={settings.temperature} /></label>
  <label>Top-p <input type="number" min="0.01" max="1" step="0.05" bind:value={settings.top_p} /></label>
  <label>Max tokens <input type="number" min="1" max="32768" step="1" bind:value={settings.max_tokens} oninput={() => onbudgetchange?.()} /></label>
  {#if architecture === 'gpt_oss'}
    <label>Reasoning effort<select aria-label="Reasoning effort" bind:value={settings.reasoning_effort}><option value="low">Low · quicker answers</option><option value="medium">Medium</option><option value="high">High · more reasoning</option></select></label>
    <p class="help">GPT-OSS shares its output budget between reasoning and the answer. New sessions use 8,192 output tokens and low effort when context allows. Higher effort may need more tokens.</p>
  {:else}
    <label>Reasoning <select bind:value={settings.reasoning}><option value="auto">Auto</option><option value="on">On</option><option value="off">Off</option></select></label>
    <label>Reasoning budget <input type="number" min="-1" step="1" bind:value={settings.reasoning_budget} /></label>
  {/if}
  <label>API key <input type="password" autocomplete="off" placeholder="Optional" bind:value={settings.api_key} /></label>
  <button type="button" onclick={onconnect}>Connect</button>
</div>
<style>.settings{display:grid;gap:.8rem}label{display:grid;gap:.3rem;color:var(--muted);font-size:.75rem}input,select{width:100%;padding:.55rem;border:1px solid var(--line);border-radius:.55rem;background:var(--bg);color:var(--ink)}button{padding:.5rem;border:1px solid var(--line);border-radius:.55rem;background:var(--accent2);color:var(--ink)}.help{font-size:.7rem;color:var(--muted);line-height:1.6;margin:0}</style>
