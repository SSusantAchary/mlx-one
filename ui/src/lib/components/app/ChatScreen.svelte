<script lang="ts">
  import { onDestroy, onMount, tick } from 'svelte';
  import { base } from '$app/paths';
  import { Menu, Moon, Plus, Sun, X, Search, Box, Cable, Settings2, SquarePen, MessageSquareText } from '@lucide/svelte';
  import ChatForm from './ChatForm.svelte';
  import ChatMessages from './ChatMessages.svelte';
  import GenerationSettings from './GenerationSettings.svelte';
  import RuntimeStatus from './RuntimeStatus.svelte';
  import ModelBadge from './ModelBadge.svelte';
  import { transcribeAudio } from '$lib/services/audio';
  import { streamChat } from '$lib/services/chat';
  import { streamBuffer } from '$lib/services/streamBuffer';
  import { modelDefaults } from '$lib/services/modelSettings';
  import { fetchModels } from '$lib/services/models';
  import { fetchRuntime } from '$lib/services/runtime';
  import { error, messages, streaming } from '$lib/stores/chat';
  import { models, selectedModel } from '$lib/stores/models';
  import { runtime } from '$lib/stores/runtime';
  import { settings } from '$lib/stores/settings';
  import type { Message } from '$lib/types/api';

  const MAX_AUDIO_BYTES = 25 * 1024 * 1024;
  interface Conversation { id: string; title: string; messages: Message[]; system: string; draft: string; }
  let conversations = $state<Conversation[]>([{ id: 'initial', title: 'New chat', messages: [], system: '', draft: '' }]);
  let activeId = $state('initial'), query = $state(''), searching = $state(false);
  const filtered = $derived(conversations.filter(c => `${c.title} ${c.messages.map(m => m.content).join(' ')}`.toLowerCase().includes(query.toLowerCase())));
  let dark = $state(true);
  let drawerOpen = $state(false), sidebarOpen = $state(false), viewportWidth = $state(1440);
  let panelElement = $state<HTMLDivElement>();
  let previousFocus: HTMLElement | null = null;
  function closePanel() { drawerOpen = false;
    if (viewportWidth <= 650 && previousFocus?.closest('.navigation')) document.querySelector<HTMLElement>('[aria-label="Open sidebar"]')?.focus({ preventScroll: true });
    else previousFocus?.focus({ preventScroll: true });
  }
  function trapPanel(event: KeyboardEvent) {
    if (event.key !== 'Tab' || !panelElement) return;
    const nodes = [...panelElement.querySelectorAll<HTMLElement>('button:not(:disabled), input, select, textarea, [tabindex="0"]')];
    const first = nodes[0], last = nodes.at(-1);
    if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
    else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
  }
  let activePanel = $state<'settings' | 'system' | 'models' | 'mcp'>('settings');
  let draft = $state('');
  let systemMessage = $state('');
  let transcribing = $state(false);
  let composer: { focus: () => void } | undefined;
  let systemInput = $state<HTMLTextAreaElement>();
  let budgetCustomized = false;
  const configuredModels = new Set<string>();
  let requestVersion = 0;
  let controller: AbortController | null = null;
  let audioController: AbortController | null = null;
  const context = $derived($models.find((item) => item.id === $selectedModel)?.mlx.context_length ?? 0);
  const modelInfo = $derived($models.find(item => item.id === $selectedModel)?.mlx);
  const transcriptionEnabled = $derived(Boolean($runtime?.transcription?.enabled));

  onMount(async () => {
    dark = true;
    document.documentElement.classList.toggle('dark', dark);
    try { await connect(); } catch (reason) { setError(reason); }
  });
  onDestroy(() => { controller?.abort(); audioController?.abort(); });

  async function connect() {
    if ($streaming) return;
    $error = null;
    const version = requestVersion;
    try {
      $models = await fetchModels($settings.api_key);
      if (!$models.some(m => m.id === $selectedModel)) $selectedModel = $models[0]?.id ?? '';
      if (!configuredModels.has($selectedModel)) {
        $settings = modelDefaults($settings, $models.find(m => m.id === $selectedModel)?.mlx, budgetCustomized);
        configuredModels.add($selectedModel);
      }
      const snapshot = await fetchRuntime($settings.api_key);
      $runtime = version === requestVersion ? snapshot : { ...snapshot, ...($runtime ?? {}) };
    } catch (reason) { setError(reason); }
  }
  function setError(reason: unknown) { $error = reason instanceof Error ? reason.message : String(reason); }
  const id = () => crypto.randomUUID();
  function requestHistory(history: Message[]) {
    history = history.filter(m => m.role !== 'assistant' || m.content.trim());
    return systemMessage.trim()
      ? [{ id: 'system-message', role: 'system' as const, content: systemMessage.trim() }, ...history]
      : history;
  }
  async function generate(history: Message[]) {
    if (!$selectedModel || $streaming) return;
    requestVersion += 1;
    $streaming = true;
    $error = null;
    controller = new AbortController();
    const assistantId = id();
    $messages = [...history, { id: assistantId, role: 'assistant', content: '', streaming: true, progress: { generated_tokens: 0, phase: 'prefill' } }];
    const buffered = streamBuffer((text, thought) => {
      $messages = $messages.map(item => item.id === assistantId ? {
        ...item, content: item.content + text,
        reasoning_content: (item.reasoning_content ?? '') + thought
      } : item);
    });
    try {
      const result = await streamChat($selectedModel, requestHistory(history), { ...$settings, reasoning_effort: modelInfo?.architecture === 'gpt_oss' ? $settings.reasoning_effort : undefined }, controller.signal, buffered.push, progress => {
        $messages = $messages.map(item => item.id === assistantId ? { ...item, progress } : item);
      });
      buffered.flush();
      $messages = $messages.map(item => item.id === assistantId ? { ...item, metrics: { ...result.metrics }, model: $selectedModel, model_info: modelInfo, finish_reason: result.finishReason, progress: undefined, streaming: false } : item);
      $runtime = { ...($runtime ?? { backend: 'mlx', device: 'gpu' }), ...result.metrics };
    } catch (reason) {
      buffered.flush();
      if (!(reason instanceof DOMException && reason.name === 'AbortError')) setError(reason);
      $messages = $messages.map(item => item.id === assistantId ? { ...item, streaming: false, finish_reason: reason instanceof DOMException && reason.name === 'AbortError' ? 'cancelled' : 'error' } : item);
    } finally { buffered.flush(); $streaming = false; controller = null; saveCurrent(); }
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
  function saveCurrent() {
    const title = $messages.find(m => m.role === 'user')?.content.slice(0,48) || 'New chat';
    conversations = conversations.map(c => c.id === activeId ? { ...c, title, messages: [...$messages], system: systemMessage, draft } : c);
  }
  function switchChat(chatId: string) {
    if ($streaming || transcribing) return;
    saveCurrent(); const chat = conversations.find(c => c.id === chatId); if (!chat) return;
    activeId = chat.id; $messages = [...chat.messages]; systemMessage = chat.system; draft = chat.draft; $error = null;
  }
  function newChat(history: Message[] = [], system = '') {
    if ($streaming || transcribing) return;
    sidebarOpen = false; saveCurrent(); activeId = id();
    conversations = [...conversations, { id: activeId, title: history.find(m => m.role === 'user')?.content.slice(0,48) || 'New chat', messages: [...history], system, draft: '' }];
    $messages = [...history]; draft = ''; systemMessage = system; $error = null;
  }
  function closeChat(chatId: string) {
    if ($streaming || transcribing) return;
    saveCurrent(); conversations = conversations.filter(c => c.id !== chatId);
    if (activeId !== chatId) return;
    const next = conversations.at(-1);
    if (next) { activeId = next.id; $messages = [...next.messages]; systemMessage = next.system; draft = next.draft; }
    else { activeId = id(); $messages = []; draft = ''; systemMessage = ''; conversations = [{ id: activeId, title: 'New chat', messages: [], system: '', draft: '' }]; }
    $error = null;
  }
  function regenerate(messageId: string) {
    if ($streaming) return;
    const history = $messages.slice(0,$messages.findIndex(m => m.id === messageId));
    if (history.at(-1)?.role === 'user') void generate(history);
  }
  function retryMore(messageId: string) {
    const message = $messages.find(m => m.id === messageId);
    const available = context ? Math.max(0, context - (message?.metrics?.prompt_tokens ?? 0)) : 8192;
    const next = Math.min(available, Math.max(2048, $settings.max_tokens * 2));
    if (next <= $settings.max_tokens) { setError('No more room in this context. Start a new chat or shorten the input.'); return; }
    $settings = { ...$settings, max_tokens: next };
    regenerate(messageId);
  }
  function editMessage(messageId: string, text: string) {
    if ($streaming) return;
    const index = $messages.findIndex(m => m.id === messageId); if (index < 0) return;
    const changed: Message = { id: messageId, role: $messages[index].role, content: text };
    const history = [...$messages.slice(0,index), changed]; $messages = history; saveCurrent();
    if (changed.role === 'user') void generate(history);
  }
  function branch(messageId: string) {
    const index = $messages.findIndex(m => m.id === messageId);
    if (index >= 0) newChat($messages.slice(0,index+1), systemMessage);
  }
  function deleteFrom(messageId: string) {
    if ($streaming) return;
    const index = $messages.findIndex(m => m.id === messageId);
    if (index >= 0) { $messages = $messages.slice(0,index); saveCurrent(); }
  }
  function toggleTheme() { dark = !dark; document.documentElement.classList.toggle('dark', dark); }
  async function openDrawer(panel: 'settings' | 'system' | 'models' | 'mcp') {
    previousFocus = document.activeElement as HTMLElement; sidebarOpen = false; activePanel = panel; drawerOpen = true;
    await tick(); panelElement?.querySelector<HTMLElement>('button, input, textarea')?.focus();
    if (panel === 'system') { await tick(); systemInput?.focus(); }
  }
</script>

<svelte:window bind:innerWidth={viewportWidth} onkeydown={(event) => { if (event.key === 'Escape') { if (drawerOpen) closePanel(); sidebarOpen = false; } }} />
<div class="shell" class:empty={!$messages.length}>
  {#if drawerOpen || sidebarOpen}<button class="backdrop" onclick={() => { closePanel(); sidebarOpen = false; }} aria-label="Close panel"></button>{/if}
  <aside class="navigation" class:open={sidebarOpen} aria-label="Main navigation" inert={drawerOpen || (viewportWidth <= 650 && !sidebarOpen)}>
    <div class="brand"><img src={`${base}/mlx-one-mark.png`} alt=""/><strong>mlx-one</strong><small>LOCAL</small><button class="icon-button mobile-close" aria-label="Close navigation" onclick={() => sidebarOpen = false}><X size={16}/></button></div>
    <nav>
      <button onclick={() => newChat()} disabled={$streaming || transcribing}><SquarePen size={19}/> New chat</button>
      <button onclick={() => searching = !searching} aria-expanded={searching}><Search size={19}/> Search</button>
      <button onclick={() => openDrawer('models')}><Box size={19}/> Models</button>
      <button onclick={() => openDrawer('mcp')}><Cable size={19}/> MCP <small>Unavailable</small></button>
      <button onclick={() => openDrawer('settings')}><Settings2 size={19}/> Settings</button>
      <button onclick={() => openDrawer('system')}><MessageSquareText size={19}/> System message</button>
    </nav>
    {#if searching}<label class="search"><Search size={15}/><input bind:value={query} aria-label="Search conversations" placeholder="Search conversations…"/></label>{/if}
    <h2>Recent conversations</h2>
    <div class="recent">
      {#each filtered as chat (chat.id)}<button class:active={chat.id === activeId} disabled={$streaming || transcribing} onclick={() => switchChat(chat.id)}>{chat.id === activeId ? ($messages.find(m => m.role === 'user')?.content.slice(0,48) || chat.title) : chat.title}</button>{/each}
      {#if !filtered.length}<p>No conversations found.</p>{/if}
    </div>
    <div class="sidebar-bottom"><span class="local-dot"></span> On-device inference <button class="icon-button" onclick={toggleTheme} aria-label="Toggle theme">{#if dark}<Sun size={16}/>{:else}<Moon size={16}/>{/if}</button></div>
    <p class="privacy">Session-only history. Chats clear on reload.</p>
  </aside>
  {#if drawerOpen}
    <div class="drawer" bind:this={panelElement} onkeydown={trapPanel} role="dialog" aria-modal="true" aria-label={`${activePanel} panel`} tabindex="-1">
      <div class="drawer-head"><strong>{activePanel === 'system' ? 'System message' : activePanel === 'models' ? 'Connected models' : activePanel === 'mcp' ? 'MCP connections' : 'Generation settings'}</strong><button class="icon-button" onclick={closePanel} aria-label="Close panel"><X size={18}/></button></div>
      {#if activePanel === 'settings'}<GenerationSettings bind:settings={$settings} architecture={modelInfo?.architecture} onbudgetchange={() => budgetCustomized = true} onconnect={connect}/>
      {:else if activePanel === 'system'}<label class="system-label">Instructions for this conversation<textarea bind:this={systemInput} bind:value={systemMessage} rows="10" placeholder="Optional instructions for the model"></textarea></label>
      {:else if activePanel === 'models'}
        <p class="panel-copy">Models exposed by this local server. Loading a different checkpoint requires restarting the server.</p>
        {#each $models as item}<button class="model-card" disabled={$streaming} onclick={() => $selectedModel = item.id}><ModelBadge model={item.id} info={item.mlx}/><span>{item.mlx.architecture} · {$selectedModel === item.id ? 'Selected' : 'Select model'}</span></button>{/each}
        {#if !$models.length}<p class="panel-copy">No model connected. Enter your API key in Settings and connect.</p>{/if}
      {:else}<div class="capability"><Cable size={32}/><h3>No MCP connections available</h3><p>This server does not expose MCP connection management. Use a client such as Kilo for external tools. Chat and local inference work without MCP.</p></div>{/if}
    </div>
  {/if}
  <main inert={drawerOpen || (viewportWidth <= 650 && sidebarOpen)}>
    <header>
      <button class="icon-button mobile-menu" onclick={() => sidebarOpen = true} aria-label="Open sidebar"><Menu size={19}/></button>
      <div class="chat-tabs" aria-label="Conversation tabs">
        {#each conversations as chat (chat.id)}<div class="chat-tab" class:active={chat.id === activeId}><button disabled={$streaming || transcribing} onclick={() => switchChat(chat.id)}>{chat.id === activeId ? ($messages.find(m => m.role === 'user')?.content.slice(0,24) || 'New chat') : chat.title.slice(0,24)}</button><button disabled={$streaming || transcribing} onclick={() => closeChat(chat.id)} aria-label={`Close ${chat.title}`}><X size={14}/></button></div>{/each}
        <button class="icon-button new-tab" disabled={$streaming || transcribing} onclick={() => newChat()} aria-label="New conversation"><Plus size={19}/></button>
      </div>
      <button class="icon-button mobile-settings" onclick={() => openDrawer('settings')} aria-label="Open settings"><Settings2 size={18}/></button>
    </header>
    {#if $error}<div class="error" role="alert">{$error}<button onclick={() => $error = null} aria-label="Dismiss error"><X size={15}/></button></div>{/if}
    {#key activeId}<ChatMessages messages={$messages} busy={$streaming} onedit={editMessage} onbranch={branch} ondelete={deleteFrom} onregenerate={regenerate} onretry={retryMore}/>{/key}
    <div class="dock"><ChatForm bind:this={composer} bind:value={draft} busy={$streaming} {transcribing} {transcriptionEnabled} model={$selectedModel} {modelInfo} messages={$messages} {context} onsubmit={submit} onstop={stop} onaudio={handleAudio} oncanceltranscription={cancelTranscription} opensettings={() => openDrawer('settings')} opensystem={() => openDrawer('system')} onerror={setError}/></div>
    <RuntimeStatus runtime={$runtime} {context}/>
  </main>
</div>
<style>
.shell{height:100vh;height:100dvh;display:grid;grid-template-columns:17rem 1fr;overflow:hidden;gap:.5rem;padding:.5rem}.navigation{background:var(--panel);border-radius:1.5rem;display:flex;flex-direction:column;padding:1.2rem .65rem;min-height:0}.brand{display:flex;align-items:center;gap:.65rem;padding:0 .5rem 1.2rem}.brand strong{font-size:1.1rem;font-weight:600;letter-spacing:-.04em}.brand small{margin-left:auto;color:var(--muted);font-size:.55rem;letter-spacing:.12em}.brand img{width:2.4rem;height:2.4rem;object-fit:cover;border-radius:50%}nav{display:grid;gap:.25rem}nav button{display:flex;align-items:center;gap:.8rem;border:0;border-radius:.7rem;background:transparent;color:var(--ink);padding:.75rem .65rem;text-align:left;font-size:.88rem}nav button:hover{background:var(--hover)}nav small{margin-left:auto;font-size:.55rem;color:var(--muted)}h2{font-size:.75rem;font-weight:600;margin:2rem .65rem .7rem}.recent{display:flex;flex-direction:column;overflow:auto;min-height:0;gap:.2rem}.recent button{flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-align:left;border:0;border-radius:.75rem;padding:.7rem;background:transparent;color:var(--muted);font-size:.8rem}.recent button.active,.recent button:hover{background:var(--hover);color:var(--ink)}.recent p{font-size:.75rem;color:var(--muted);padding:.5rem}.sidebar-bottom{margin-top:auto;display:flex;align-items:center;gap:.4rem;padding:1rem .5rem .3rem;font-size:.65rem;color:var(--muted)}.sidebar-bottom button{margin-left:auto}.local-dot{height:6px;width:6px;border-radius:50%;background:#88bb9d}.privacy{font-size:.6rem;color:var(--muted);padding:0 .5rem;margin:.3rem 0}.search{display:flex;align-items:center;gap:.4rem;margin:1rem .5rem 0;border:1px solid var(--line);padding:.5rem;border-radius:.6rem;color:var(--muted)}.search input{min-width:0;width:100%;border:0;background:transparent;color:var(--ink);font-size:.75rem;outline:none}main{height:100%;min-width:0;min-height:0;display:flex;flex-direction:column;position:relative;overflow:hidden}header{height:3.7rem;flex:0 0 3.7rem;display:flex;align-items:center;gap:.5rem;padding:0 .5rem}.chat-tabs{display:flex;align-items:center;gap:.4rem;overflow:auto;min-width:0}.chat-tab{display:flex;border:1px solid transparent;border-radius:.7rem;flex-shrink:0}.chat-tab.active{background:var(--panel);border-color:var(--line)}.chat-tab>button{background:none;border:0;color:var(--muted);padding:.55rem .65rem;font-size:.8rem}.chat-tab.active>button{color:var(--ink)}.chat-tab>button+button{padding-left:.1rem;display:grid;place-items:center}.new-tab{background:transparent;border:0;flex-shrink:0}.dock{position:absolute;z-index:4;left:0;right:0;bottom:2.6rem;padding:1rem clamp(1rem,4vw,4rem);background:linear-gradient(transparent,var(--bg) 28%)}.empty .dock{bottom:7%}.backdrop{position:fixed;z-index:19;inset:0;border:0;background:#0007}.drawer{position:fixed;z-index:20;right:1rem;top:1rem;bottom:1rem;width:min(24rem,calc(100vw - 2rem));background:var(--panel);border:1px solid var(--line);border-radius:1.2rem;padding:1.2rem;overflow:auto;box-shadow:0 20px 80px #0006}.drawer-head{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem}.drawer-head strong{font-size:.95rem;font-weight:500}.system-label{display:grid;gap:.6rem;color:var(--muted);font-size:.8rem}.system-label textarea{border:1px solid var(--line);border-radius:.6rem;padding:.7rem;background:var(--bg);color:var(--ink);resize:vertical}.model-card{display:grid;gap:.7rem;text-align:left;width:100%;border:1px solid var(--line);background:var(--bg);border-radius:.8rem;padding:.8rem;margin-top:.8rem;color:var(--ink)}.model-card>span{font-size:.7rem;color:var(--muted)}.panel-copy,.capability p{color:var(--muted);font-size:.8rem;line-height:1.7}.capability{padding:2rem .5rem}.capability h3{font-size:1rem}.mobile-menu,.mobile-settings,.mobile-close{display:none}.error{position:absolute;z-index:12;top:4rem;left:50%;transform:translateX(-50%);background:#812d2a;color:white;padding:.65rem 1rem;border-radius:.6rem;font-size:.8rem;max-width:85%;display:flex;gap:.8rem}.error button{color:white;border:0;background:none}button:disabled{opacity:.4;cursor:not-allowed}@media(max-width:900px){.shell{grid-template-columns:13rem 1fr}.brand small,nav small{display:none}.navigation{padding-top:1rem}.dock{padding:.8rem 1rem}}@media(max-width:650px){.shell{grid-template-columns:1fr;padding:0}.navigation{position:fixed;z-index:20;inset:.5rem auto .5rem .5rem;width:min(18rem,85vw);transform:translateX(-110%);transition:transform .2s}.navigation.open{transform:translateX(0)}.mobile-menu,.mobile-settings,.mobile-close{display:grid;flex-shrink:0}.mobile-settings{margin-left:auto}header{padding:0 .6rem}.dock{padding:.7rem;bottom:2.4rem}.empty .dock{bottom:3.5rem}.chat-tabs{flex:1}.drawer{right:.5rem;left:.5rem;width:auto;top:.5rem;bottom:.5rem}}
</style>
