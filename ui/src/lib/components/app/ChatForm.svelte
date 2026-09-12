<script lang="ts">
  import { ArrowUp, Square } from '@lucide/svelte';
  let { busy, onsubmit, onstop } = $props<{ busy: boolean; onsubmit: (text: string) => void; onstop: () => void }>();
  let value = $state('');
  function send() { const text = value.trim(); if (!text || busy) return; value = ''; onsubmit(text); }
  function keydown(event: KeyboardEvent) { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(); } }
</script>

<div class="composer">
  <textarea bind:value onkeydown={keydown} rows="1" placeholder="Message the model…" aria-label="Message" disabled={busy}></textarea>
  {#if busy}<button class="send stop" onclick={onstop} aria-label="Stop generation"><Square size={16} fill="currentColor" /></button>{:else}<button class="send" onclick={send} aria-label="Send message" disabled={!value.trim()}><ArrowUp size={19} /></button>{/if}
  <small>Enter to send · Shift+Enter for a new line</small>
</div>

<style>
  .composer { position:relative; max-width:52rem; margin:0 auto; padding-bottom:1.6rem; }
  textarea { width:100%; min-height:4.3rem; max-height:12rem; resize:vertical; border:1px solid var(--line); border-radius:1.2rem; padding:1.2rem 4rem 1.2rem 1.2rem; background:var(--panel); color:var(--ink); outline:none; box-shadow:0 18px 55px rgb(0 0 0 / 10%); }
  textarea:focus { border-color:var(--accent); }
  .send { position:absolute; right:.75rem; top:.72rem; width:2.8rem; height:2.8rem; display:grid; place-items:center; border:0; border-radius:.85rem; background:var(--accent); color:#062417; }
  .send:disabled { opacity:.35; }
  .stop { background:#df665e; color:white; }
  small { display:block; text-align:center; margin-top:.55rem; color:var(--muted); font-size:.7rem; }
</style>
