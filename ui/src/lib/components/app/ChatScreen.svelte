<script lang="ts">
  import { onMount } from 'svelte';
  import { Moon, Plus, RotateCcw, Sun, Trash2 } from '@lucide/svelte';
  import ChatForm from './ChatForm.svelte';
  import ChatMessages from './ChatMessages.svelte';
  import GenerationSettings from './GenerationSettings.svelte';
  import ModelSelector from './ModelSelector.svelte';
  import RuntimeStatus from './RuntimeStatus.svelte';
  import { streamChat } from '$lib/services/chat';
  import { fetchModels } from '$lib/services/models';
  import { fetchRuntime } from '$lib/services/runtime';
  import { error, messages, streaming } from '$lib/stores/chat';
  import { models, selectedModel } from '$lib/stores/models';
  import { runtime } from '$lib/stores/runtime';
  import { settings } from '$lib/stores/settings';
  import type { Message } from '$lib/types/api';

  let dark = $state(false);
  let controller: AbortController | null = null;
  const context = $derived(
    $models.find((item) => item.id === $selectedModel)?.mlx.context_length ?? 0
  );

  onMount(async () => {
    dark = matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.classList.toggle('dark', dark);
    try {
      $models = await fetchModels();
      $selectedModel = $models[0]?.id ?? '';
      $runtime = await fetchRuntime();
    } catch (reason) {
      $error = reason instanceof Error ? reason.message : String(reason);
    }
  });

  const id = () => crypto.randomUUID();
  async function generate(history: Message[]) {
    if (!$selectedModel || $streaming) return;
    $streaming = true;
    $error = null;
    controller = new AbortController();
    const assistantId = id();
    $messages = [...history, { id: assistantId, role: 'assistant', content: '' }];
    try {
      const result = await streamChat($selectedModel, history, $settings, controller.signal, (text) => {
        $messages = $messages.map((item) =>
          item.id === assistantId ? { ...item, content: item.content + text } : item
        );
      });
      $runtime = { ...($runtime ?? { backend: 'mlx', device: 'gpu' }), ...result.metrics };
    } catch (reason) {
      if (!(reason instanceof DOMException && reason.name === 'AbortError')) {
        $error = reason instanceof Error ? reason.message : String(reason);
      }
      if (!$messages.find((item) => item.id === assistantId)?.content) {
        $messages = $messages.filter((item) => item.id !== assistantId);
      }
    } finally {
      $streaming = false;
      controller = null;
    }
  }
  function submit(content: string) {
    const history = [...$messages, { id: id(), role: 'user' as const, content }];
    $messages = history;
    void generate(history);
  }
  function stop() { controller?.abort(); }
  function clear() { stop(); $messages = []; $error = null; }
  function regenerate() {
    if ($streaming) return;
    const history = $messages.at(-1)?.role === 'assistant' ? $messages.slice(0, -1) : $messages;
    if (history.length) { $messages = history; void generate(history); }
  }
  function toggleTheme() { dark = !dark; document.documentElement.classList.toggle('dark', dark); }
</script>

<div class="shell">
  <aside class="sidebar">
    <div class="brand"><i></i><strong>mlx-one</strong></div>
    <button class="new" onclick={clear}><Plus size={17} /> New chat</button>
    <div class="local"><span>LOCAL SESSION</span><p>{$messages.length ? 'Current conversation' : 'No conversations yet'}</p></div>
    <GenerationSettings bind:settings={$settings} />
    <p class="privacy">Nothing leaves this Mac except model downloads you request.</p>
  </aside>
  <main>
    <header><div class="mobile-brand"><i></i><strong>mlx-one</strong></div><ModelSelector models={$models} value={$selectedModel} /><div class="actions"><button class="icon-button" onclick={regenerate} disabled={$streaming || !$messages.length} aria-label="Regenerate"><RotateCcw size={17} /></button><button class="icon-button" onclick={clear} aria-label="Clear conversation"><Trash2 size={17} /></button><button class="icon-button" onclick={toggleTheme} aria-label="Toggle theme">{#if dark}<Sun size={17} />{:else}<Moon size={17} />{/if}</button></div></header>
    {#if $error}<div class="error" role="alert">{$error}</div>{/if}
    <ChatMessages messages={$messages} />
    <div class="dock"><ChatForm busy={$streaming} onsubmit={submit} onstop={stop} /></div>
    <RuntimeStatus runtime={$runtime} {context} />
  </main>
</div>

<style>
  .shell{height:100vh;display:grid;grid-template-columns:16rem 1fr;overflow:hidden}.sidebar{border-right:1px solid var(--line);background:var(--panel);padding:1.2rem;display:flex;flex-direction:column;gap:1.6rem}.brand,.mobile-brand{display:flex;align-items:center;gap:.65rem}.brand i,.mobile-brand i{width:1rem;height:1rem;background:var(--accent);border-radius:35% 65% 55% 45%;transform:rotate(25deg)}.new{display:flex;align-items:center;justify-content:center;gap:.5rem;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:.75rem;padding:.7rem}.local{flex:1}.local span{font-size:.62rem;color:var(--muted);letter-spacing:.12em}.local p{font-size:.83rem}.privacy{font-size:.68rem;line-height:1.5;color:var(--muted)}main{min-width:0;display:flex;flex-direction:column;position:relative}header{height:4.5rem;border-bottom:1px solid var(--line);display:flex;align-items:center;justify-content:space-between;padding:0 1.2rem;background:color-mix(in srgb,var(--bg) 86%,transparent)}.actions{display:flex;gap:.45rem}.mobile-brand{display:none}.dock{position:absolute;left:0;right:0;bottom:2.6rem;padding:1rem clamp(1rem,4vw,4rem);background:linear-gradient(transparent,var(--bg) 35%)}.error{position:absolute;z-index:3;top:5.2rem;left:50%;transform:translateX(-50%);background:#812d2a;color:white;padding:.65rem 1rem;border-radius:.6rem;font-size:.8rem;max-width:80%}@media(max-width:760px){.shell{grid-template-columns:1fr}.mobile-brand{display:flex}header :global(label){display:none}}
</style>
