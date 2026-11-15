<script lang="ts">
  import { CircleGauge, X } from '@lucide/svelte';
  import type { Message } from '$lib/types/api';
  let { messages = [], context = 0 }: { messages?: Message[]; context?: number } = $props();
  let open = $state(false);
  const turns = $derived(messages.filter(m => m.role === 'assistant' && m.metrics));
  const last = $derived(turns.at(-1)?.metrics);
  const used = $derived(last?.context_used ?? 0);
  const percent = $derived(context > 0 ? Math.min(100, used / context * 100) : 0);
  const fresh = $derived(last?.actual_prefill_tokens ?? (last?.prompt_tokens != null ? Math.max(0, last.prompt_tokens - (last.reused_prompt_tokens ?? 0)) : undefined));
  const evaluated = $derived(turns.reduce((sum, m) => sum + (m.metrics?.actual_prefill_tokens ?? Math.max(0, (m.metrics?.prompt_tokens ?? 0) - (m.metrics?.reused_prompt_tokens ?? 0))), 0));
  const generated = $derived(turns.reduce((sum, m) => sum + (m.metrics?.generated_tokens ?? 0), 0));
  const timed = $derived(turns.filter(m => m.metrics?.generation_latency_ms && m.metrics?.generated_tokens != null));
  const average = $derived(timed.length ? timed.reduce((s,m) => s + (m.metrics?.generated_tokens ?? 0), 0) * 1000 / timed.reduce((s,m) => s + (m.metrics?.generation_latency_ms ?? 0), 0) : null);
  const number = (v?: number) => v == null ? '—' : v.toLocaleString();
</script>
<svelte:window onkeydown={(e) => { if (e.key === 'Escape') open = false; }} />
<div class="context-wrap">
  <button class="context-button" class:active={open} onclick={() => open = !open} aria-expanded={open} aria-label="Context usage" title="Context usage"><CircleGauge size={18} /><span>{last ? `${Math.round(percent)}%` : 'Context'}</span></button>
  {#if open}
    <section class="context-card" aria-label="Token usage details">
      <div class="heading"><strong>Context</strong><span>{number(last?.context_used)} / {context ? number(context) : '—'}</span><button onclick={() => open = false} aria-label="Close context details"><X size={15}/></button></div>
      <div class="track"><div style={`width:${percent}%`}></div></div>
      <div class="remaining"><span>{percent.toFixed(1)}% used</span><span>{context && last ? number(Math.max(0, context-used)) : '—'} remaining</span></div>
      <h4>Across this conversation</h4>
      <dl><div><dt>Input tokens evaluated</dt><dd>{turns.length ? number(evaluated) : '—'}</dd></div><div><dt>Tokens generated</dt><dd>{turns.length ? number(generated) : '—'}</dd></div></dl>
      <h4>Last request · context</h4>
      <dl><div><dt>Input tokens</dt><dd>{number(last?.prompt_tokens)}</dd></div><div class="sub"><dt>Fresh input</dt><dd>{number(fresh)}</dd></div><div class="sub"><dt>Reused prefix</dt><dd>{last ? number(last.reused_prompt_tokens ?? 0) : '—'}</dd></div><div><dt>Output tokens</dt><dd>{number(last?.generated_tokens)}</dd></div><div class="total"><dt>Request total</dt><dd>{number(last?.context_used)}</dd></div><div><dt>Avg overall speed</dt><dd>{average == null ? '—' : `${average.toFixed(1)} tok/s`}</dd></div></dl>
      <p>Measured request tokens. Reuse depends on the server’s cache settings.</p>
    </section>
  {/if}
</div>
<style>
  .context-wrap{position:relative}.context-button{display:flex;align-items:center;gap:.4rem;border:0;background:none;color:var(--muted);padding:.5rem;border-radius:.5rem;font-size:.72rem}.context-button:hover,.context-button.active{background:var(--hover);color:var(--ink)}.context-card{max-height:calc(100dvh - 7rem);overflow:auto;position:absolute;bottom:3rem;right:0;width:min(22rem,calc(100vw - 2.5rem));padding:1.2rem;border:1px solid var(--line);border-radius:1rem;background:var(--panel);box-shadow:0 20px 80px #0005;color:var(--muted);font-size:.8rem;z-index:15}.heading{display:flex;align-items:center;gap:.5rem}.heading strong{color:var(--ink)}.heading span{margin-left:auto;font-family:ui-monospace,monospace}.heading button{background:none;color:var(--muted);border:0;padding:0}.track{height:5px;background:var(--hover);border-radius:4px;overflow:hidden;margin-top:1rem}.track>div{height:100%;background:var(--accent)}.remaining{display:flex;justify-content:space-between;margin-top:.5rem;font-size:.7rem}h4{font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;margin:1.4rem 0 .7rem;font-weight:500}dl{margin:0}dl>div{display:flex;justify-content:space-between;gap:.7rem;padding:.4rem 0}dd{margin:0;font-family:ui-monospace,monospace;color:var(--ink)}.sub{font-size:.7rem}.total{border-top:1px solid var(--line);margin-top:.4rem;padding-top:.7rem}p{border-top:1px solid var(--line);padding-top:.8rem;margin:.8rem 0 0;font-size:.65rem;line-height:1.5}
@media(max-width:480px){.context-button span{display:none}}
</style>
