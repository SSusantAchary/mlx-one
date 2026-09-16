<script lang="ts">
  import { onDestroy } from 'svelte';
  import {
    ArrowUp, AudioLines, ChevronDown, FileAudio, MessageSquareText,
    Mic, Plus, Settings2, Square, X
  } from '@lucide/svelte';

  const MAX_RECORDING_SECONDS = 5 * 60;
  let {
    busy, transcribing, transcriptionEnabled, model, value = $bindable(''),
    onsubmit, onstop, onaudio, oncanceltranscription, opensettings, opensystem, onerror
  } = $props<{
    busy: boolean;
    transcribing: boolean;
    transcriptionEnabled: boolean;
    model: string;
    value?: string;
    onsubmit: (text: string) => void;
    onstop: () => void;
    onaudio: (file: File) => void;
    oncanceltranscription: () => void;
    opensettings: () => void;
    opensystem: () => void;
    onerror: (message: string) => void;
  }>();

  let menuOpen = $state(false);
  let recording = $state(false);
  let seconds = $state(0);
  let input: HTMLTextAreaElement;
  let fileInput: HTMLInputElement;
  let recorder: MediaRecorder | null = null;
  let mediaStream: MediaStream | null = null;
  let chunks: Blob[] = [];
  let timer: ReturnType<typeof setInterval> | null = null;
  let discardRecording = false;

  function send() {
    const text = value.trim();
    if (!text || busy || transcribing) return;
    value = '';
    onsubmit(text);
  }
  function keydown(event: KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(); }
    if (event.key === 'Escape') menuOpen = false;
  }
  function chooseFile() { menuOpen = false; fileInput.click(); }
  function selected(event: Event) {
    const target = event.currentTarget as HTMLInputElement;
    const file = target.files?.[0];
    if (file) onaudio(file);
    target.value = '';
  }
  function supportedMimeType() {
    return ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/mp4']
      .find((type) => MediaRecorder.isTypeSupported(type)) ?? '';
  }
  async function startRecording() {
    if (!transcriptionEnabled || busy || transcribing || recording) return;
    try {
      if (!navigator.mediaDevices?.getUserMedia || typeof MediaRecorder === 'undefined') {
        throw new Error('Microphone recording is not supported by this browser. Upload an audio file instead.');
      }
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mimeType = supportedMimeType();
      recorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);
      chunks = [];
      discardRecording = false;
      seconds = 0;
      recorder.ondataavailable = (event) => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = () => {
        const type = recorder?.mimeType || mimeType || 'audio/webm';
        const blob = new Blob(chunks, { type });
        cleanupMedia();
        if (!discardRecording && blob.size) {
          const extension = type.includes('ogg') ? 'ogg' : type.includes('mp4') ? 'm4a' : 'webm';
          onaudio(new File([blob], `recording.${extension}`, { type }));
        }
      };
      recorder.start(500);
      recording = true;
      timer = setInterval(() => {
        seconds += 1;
        if (seconds >= MAX_RECORDING_SECONDS) stopRecording();
      }, 1000);
    } catch (reason) {
      cleanupMedia();
      onerror(reason instanceof Error ? reason.message : String(reason));
    }
  }
  function stopRecording() {
    if (recorder?.state === 'recording') recorder.stop();
  }
  function cancelRecording() {
    discardRecording = true;
    if (recorder?.state === 'recording') recorder.stop(); else cleanupMedia();
  }
  function cleanupMedia() {
    if (timer) clearInterval(timer);
    timer = null;
    mediaStream?.getTracks().forEach((track) => track.stop());
    mediaStream = null;
    recorder = null;
    chunks = [];
    recording = false;
  }
  function formatTime(time: number) {
    return `${Math.floor(time / 60)}:${String(time % 60).padStart(2, '0')}`;
  }
  function focus() { input?.focus(); }
  export { focus };
  onDestroy(() => { discardRecording = true; cleanupMedia(); });
</script>

<div class="composer" class:recording>
  <textarea bind:this={input} bind:value onkeydown={keydown} rows="2" placeholder="Type a message…" aria-label="Message" disabled={busy || transcribing}></textarea>
  <div class="bar">
    <div class="leading">
      <button class="round" class:active={menuOpen} onclick={() => menuOpen = !menuOpen} aria-label="Add" aria-expanded={menuOpen}><Plus size={21} /></button>
      {#if menuOpen}
        <div class="menu" role="menu">
          <button role="menuitem" onclick={chooseFile} disabled={!transcriptionEnabled}><FileAudio size={18} /><span>Upload audio<small>{transcriptionEnabled ? 'MP3, WAV, M4A, OGG or WebM' : 'Start with --transcription-model'}</small></span></button>
          <button role="menuitem" onclick={() => { menuOpen = false; opensystem(); }}><MessageSquareText size={18} /><span>System message<small>Guide the model response</small></span></button>
          <button role="menuitem" onclick={() => { menuOpen = false; opensettings(); }}><Settings2 size={18} /><span>Generation settings<small>Sampling, reasoning and API key</small></span></button>
        </div>
      {/if}
      {#if recording}
        <span class="recording-label"><i></i>{formatTime(seconds)}</span>
        <button class="round danger" onclick={cancelRecording} aria-label="Cancel recording"><X size={18} /></button>
        <button class="record-stop" onclick={stopRecording}><Square size={14} fill="currentColor" /> Use recording</button>
      {:else if transcribing}
        <span class="transcribing"><AudioLines size={18} /> Transcribing locally…</span>
        <button class="cancel-text" onclick={oncanceltranscription}>Cancel</button>
      {/if}
    </div>
    <div class="trailing">
      {#if !recording && !transcribing}
        <button class="round mic" onclick={startRecording} disabled={!transcriptionEnabled || busy} aria-label="Record audio" title={transcriptionEnabled ? 'Record audio' : 'Start the server with --transcription-model to enable the microphone'}><Mic size={20} /></button>
      {/if}
      <div class="model" title={model}><span>{model || 'No model'}</span><ChevronDown size={15} /></div>
      {#if busy}<button class="send stop" onclick={onstop} aria-label="Stop generation"><Square size={16} fill="currentColor" /></button>{:else}<button class="send" onclick={send} aria-label="Send message" disabled={!value.trim() || transcribing || recording}><ArrowUp size={21} /></button>{/if}
    </div>
  </div>
  <input bind:this={fileInput} class="file" type="file" accept="audio/*" onchange={selected} aria-label="Upload audio file" />
</div>
<small class="hint">Enter to send · Shift+Enter for a new line</small>

<style>
  .composer{position:relative;max-width:64rem;margin:0 auto;min-height:11rem;padding:1.3rem 1.35rem 1rem;border:1px solid var(--line);border-radius:2rem;background:var(--panel);box-shadow:0 20px 60px rgb(0 0 0 / 9%)}
  textarea{width:100%;min-height:5.8rem;max-height:13rem;resize:none;border:0;background:transparent;color:var(--ink);outline:none;font-size:1.08rem;line-height:1.5;padding:0}
  .bar,.leading,.trailing{display:flex;align-items:center}.bar{justify-content:space-between;gap:.75rem}.leading,.trailing{gap:.55rem}.round,.send{display:grid;place-items:center;width:2.75rem;height:2.75rem;border-radius:50%;border:1px solid var(--line);background:var(--bg);color:var(--ink)}.round:hover,.round.active{border-color:var(--accent);color:var(--accent)}button:disabled{opacity:.35;cursor:not-allowed}.send{border:0;background:var(--accent);color:#062417}.send.stop{background:#df665e;color:white}.mic{background:transparent}.model{height:2.6rem;max-width:19rem;display:flex;align-items:center;gap:.4rem;padding:0 .8rem;border:1px solid var(--line);border-radius:1rem;background:var(--bg);font-size:.8rem}.model span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.menu{position:absolute;z-index:10;left:1rem;bottom:4.5rem;width:min(21rem,calc(100vw - 3rem));padding:.45rem;border:1px solid var(--line);border-radius:1rem;background:var(--panel);box-shadow:0 22px 60px rgb(0 0 0 / 18%)}.menu button{width:100%;display:flex;align-items:flex-start;gap:.75rem;border:0;border-radius:.7rem;background:transparent;color:var(--ink);padding:.7rem;text-align:left}.menu button:hover{background:var(--accent2)}.menu span{display:grid;gap:.15rem}.menu small{color:var(--muted);font-size:.68rem}.file{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}.recording-label,.transcribing{display:flex;align-items:center;gap:.45rem;color:var(--muted);font-size:.78rem}.recording-label i{width:.55rem;height:.55rem;border-radius:50%;background:#e6534b;animation:pulse 1s infinite}.danger{color:#d84e47}.record-stop,.cancel-text{border:0;border-radius:.7rem;padding:.6rem .75rem;background:var(--accent2);color:var(--ink);display:flex;align-items:center;gap:.4rem;font-size:.75rem}.cancel-text{background:transparent;color:var(--muted)}.hint{display:block;text-align:center;margin:.55rem;color:var(--muted);font-size:.7rem}@keyframes pulse{50%{opacity:.3}}@media(max-width:680px){.composer{min-height:9.5rem;border-radius:1.4rem}.model{max-width:8.5rem}.transcribing{font-size:0}.hint{display:none}}
</style>
