<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte';
  import { Menu, Moon, Plus, RotateCcw, Sun, Trash2, X } from '@lucide/svelte';
  import ChatForm from './ChatForm.svelte';
  import ChatMessages from './ChatMessages.svelte';
  import GenerationSettings from './GenerationSettings.svelte';
  import RuntimeStatus from './RuntimeStatus.svelte';
  import { transcribeAudio } from '$lib/services/audio';
  import { streamChat } from '$lib/services/chat';
  import { fetchModels } from '$lib/services/models';
  import { fetchRuntime } from '$lib/services/runtime';
  import { error, messages, streaming } from '$lib/stores/chat';
  import { models, selectedModel } from '$lib/stores/models';
  import { runtime } from '$lib/stores/runtime';
  import { settings } from '$lib/stores/settings';
  import type { Message } from '$lib/types/api';

  const MAX_AUDIO_BYTES = 25 * 1024 * 1024;
  let dark = $state(false);
  let drawerOpen = $state(false);
  let activePanel = $state<'settings' | 'system'>('settings');
  let draft = $state('');
  let systemMessage = $state('');
  let transcribing = $state(false);
  let composer: { focus: () => void } | undefined;
  let systemInput = $state<HTMLTextAreaElement>();
  let controller: AbortController | null = null;
  let audioController: AbortController | null = null;
  const context = $derived($models.find((item) => item.id === $selectedModel)?.mlx.context_length ?? 0);
  const transcriptionEnabled = $derived(Boolean($runtime?.transcription?.enabled));

  onMount(async () => {
    dark = matchMedia('(prefers-color-scheme: dark)').matches;
    document.documentElement.classList.toggle('dark', dark);
    try { await connect(); } catch (reason) { setError(reason); }
  });
  onDestroy(() => { controller?.abort(); audioController?.abort(); });

  async function connect() {
    $error = null;
    try {
      $models = await fetchModels($settings.api_key);
      $selectedModel = $models[0]?.id ?? '';
      $runtime = await fetchRuntime($settings.api_key);
    } catch (reason) { setError(reason); }
  }
  function setError(reason: unknown) { $error = reason instanceof Error ? reason.message : String(reason); }
  const id = () => crypto.randomUUID();
  function requestHistory(history: Message[]) {
    return systemMessage.trim()
      ? [{ id: 'system-message', role: 'system' as const, content: systemMessage.trim() }, ...history]
      : history;
  }
  async function generate(history: Message[]) {
    if (!$selectedModel || $streaming) return;
    $streaming = true;
    $error = null;
    controller = new AbortController();
    const assistantId = id();
    $messages = [...history, { id: assistantId, role: 'assistant', content: '' }];
    try {
      const result = await streamChat($selectedModel, requestHistory(history), $settings, controller.signal, (text, thought) => {
        $messages = $messages.map((item) => item.id === assistantId
          ? { ...item, content: item.content + text, reasoning_content: (item.reasoning_content ?? '') + thought }
          : item);
      });
      $runtime = { ...($runtime ?? { backend: 'mlx', device: 'gpu' }), ...result.metrics };
    } catch (reason) {
      if (!(reason instanceof DOMException && reason.name === 'AbortError')) setError(reason);
      if (!$messages.find((item) => item.id === assistantId)?.content) $messages = $messages.filter((item) => item.id !== assistantId);
    } finally { $streaming = false; controller = null; }
  }
  function submit(content: string) {
    const history = [...$messages, { id: id(), role: 'user' as const, content }];
    $messages = history;
    void generate(history);
  }
  async function handleAudio(file: File) {
    if (!$selectedModel || transcribing) return;
    if (!file.type.startsWith('audio/')) { setError('Choose a supported audio file.'); return; }
    if (!file.size) { setError('The audio recording is empty.'); return; }
    if (file.size > MAX_AUDIO_BYTES) { setError('Audio files are limited to 25 MiB.'); return; }
    transcribing = true;
    $error = null;
    audioController = new AbortController();
    try {
      const result = await transcribeAudio($selectedModel, file, $settings.api_key, audioController.signal);
      draft = [draft.trim(), result.text.trim()].filter(Boolean).join(draft.trim() ? ' ' : '');
      await tick();
      composer?.focus();
    } catch (reason) {
      if (!(reason instanceof DOMException && reason.name === 'AbortError')) setError(reason);
    } finally { transcribing = false; audioController = null; }
  }
  function stop() { controller?.abort(); }
  function cancelTranscription() { audioController?.abort(); }
  function clear() {
    stop(); cancelTranscription(); $messages = []; draft = ''; systemMessage = ''; $error = null;
  }
  function regenerate() {
    if ($streaming) return;
    const history = $messages.at(-1)?.role === 'assistant' ? $messages.slice(0, -1) : $messages;
    if (history.length) { $messages = history; void generate(history); }
  }
  function toggleTheme() { dark = !dark; document.documentElement.classList.toggle('dark', dark); }
  async function openDrawer(panel: 'settings' | 'system') {
    activePanel = panel; drawerOpen = true;
    if (panel === 'system') { await tick(); systemInput?.focus(); }
  }
</script>

<svelte:window onkeydown={(event) => { if (event.key === 'Escape') drawerOpen = false; }} />
<div class="shell" class:empty={!$messages.length}>
  {#if drawerOpen}<button class="backdrop" onclick={() => drawerOpen = false} aria-label="Close sidebar"></button>{/if}
  <aside class:open={drawerOpen} aria-hidden={!drawerOpen} aria-label="Chat settings">
    <div class="drawer-head"><div class="brand"><i></i><strong>mlx-one</strong></div><button class="icon-button" onclick={() => drawerOpen = false} aria-label="Close sidebar"><X size={18} /></button></div>
    <button class="new" onclick={clear}><Plus size={17} /> New chat</button>
    <div class="tabs" role="tablist">
      <button class:active={activePanel === 'settings'} onclick={() => activePanel = 'settings'}>Settings</button>
      <button class:active={activePanel === 'system'} onclick={() => activePanel = 'system'}>System</button>
    </div>
    {#if activePanel === 'settings'}
      <GenerationSettings bind:settings={$settings} onconnect={connect} />
    {:else}
      <label class="system-label">System message<textarea bind:this={systemInput} bind:value={systemMessage} rows="8" placeholder="Optional instructions for this conversation"></textarea></label>
    {/if}
    <div class="local"><span>LOCAL SESSION</span><p>{$messages.length ? 'Current conversation' : 'No conversations yet'}</p></div>
    <p class="privacy">Audio is transcribed locally. Nothing is persisted.</p>
  </aside>
  <main>
    <header>
      <button class="icon-button" onclick={() => drawerOpen = true} aria-label="Open sidebar" aria-expanded={drawerOpen}><Menu size={19} /></button>
      <div class="brand"><i></i><strong>mlx-one</strong></div>
      <div class="actions"><button class="icon-button" onclick={regenerate} disabled={$streaming || !$messages.length} aria-label="Regenerate"><RotateCcw size={17} /></button><button class="icon-button" onclick={clear} aria-label="Clear conversation"><Trash2 size={17} /></button><button class="icon-button" onclick={toggleTheme} aria-label="Toggle theme">{#if dark}<Sun size={17} />{:else}<Moon size={17} />{/if}</button></div>
    </header>
    {#if $error}<div class="error" role="alert">{$error}</div>{/if}
    <ChatMessages messages={$messages} />
    <div class="dock"><ChatForm bind:this={composer} bind:value={draft} busy={$streaming} {transcribing} {transcriptionEnabled} model={$selectedModel} onsubmit={submit} onstop={stop} onaudio={handleAudio} oncanceltranscription={cancelTranscription} opensettings={() => openDrawer('settings')} opensystem={() => openDrawer('system')} onerror={setError} /></div>
    <RuntimeStatus runtime={$runtime} {context} />
  </main>
</div>

<style>
  .shell{height:100vh;height:100dvh;overflow:hidden}.backdrop{position:fixed;z-index:19;inset:0;border:0;background:rgb(0 0 0 / 35%)}aside{position:fixed;z-index:20;inset:0 auto 0 0;width:min(22rem,90vw);transform:translateX(-105%);transition:transform .2s ease;border-right:1px solid var(--line);background:var(--panel);padding:1.2rem;display:flex;flex-direction:column;gap:1.2rem;box-shadow:0 0 60px rgb(0 0 0 / 18%)}aside.open{transform:translateX(0)}.drawer-head,.brand,.actions{display:flex;align-items:center}.drawer-head{justify-content:space-between}.brand{gap:.65rem}.brand i{width:1rem;height:1rem;background:var(--accent);border-radius:35% 65% 55% 45%;transform:rotate(25deg)}.new{display:flex;align-items:center;justify-content:center;gap:.5rem;border:1px solid var(--line);background:var(--bg);color:var(--ink);border-radius:.75rem;padding:.7rem}.tabs{display:grid;grid-template-columns:1fr 1fr;border-bottom:1px solid var(--line)}.tabs button{border:0;border-bottom:2px solid transparent;background:transparent;color:var(--muted);padding:.7rem}.tabs button.active{border-color:var(--accent);color:var(--ink)}.system-label{display:grid;gap:.5rem;font-size:.78rem;color:var(--muted)}.system-label textarea{resize:vertical;border:1px solid var(--line);border-radius:.7rem;background:var(--bg);color:var(--ink);padding:.75rem}.local{margin-top:auto}.local span{font-size:.62rem;color:var(--muted);letter-spacing:.12em}.local p{font-size:.83rem}.privacy{font-size:.68rem;line-height:1.5;color:var(--muted)}main{height:100%;min-width:0;min-height:0;display:flex;flex-direction:column;position:relative;overflow:hidden}header{height:4.5rem;flex:0 0 4.5rem;border-bottom:1px solid var(--line);display:grid;grid-template-columns:1fr auto 1fr;align-items:center;padding:0 1.2rem;background:color-mix(in srgb,var(--bg) 86%,transparent)}header>.brand{justify-self:center}.actions{justify-self:end;gap:.45rem}.dock{position:absolute;z-index:4;left:0;right:0;bottom:2.35rem;padding:1rem clamp(1rem,4vw,4rem);background:linear-gradient(transparent,var(--bg) 32%)}.empty .dock{top:42%;bottom:auto;transform:translateY(-8%);background:transparent}.error{position:absolute;z-index:12;top:5.2rem;left:50%;transform:translateX(-50%);background:#812d2a;color:white;padding:.65rem 1rem;border-radius:.6rem;font-size:.8rem;max-width:80%}@media(max-width:680px){header{padding:0 .7rem}.empty .dock{top:36%}.dock{padding:.7rem;bottom:2rem}.actions{gap:.25rem}}
</style>
