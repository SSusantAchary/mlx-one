<script lang="ts">
  import { tick } from 'svelte';
  import ChatMessage from './ChatMessage.svelte';
  import type { Message } from '$lib/types/api';
  let { messages } = $props<{ messages: Message[] }>();
  let viewport: HTMLDivElement;
  let followOutput = true;

  function updateFollowState() {
    const distanceFromBottom = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
    followOutput = distanceFromBottom < 80;
  }

  $effect(() => {
    messages;
    if (followOutput) {
      void tick().then(() => {
        if (viewport) viewport.scrollTop = viewport.scrollHeight;
      });
    }
  });
</script>

<div class="messages" bind:this={viewport} onscroll={updateFollowState} aria-live="polite">
  {#if messages.length === 0}
    <section class="welcome"><h1>Hello there</h1><p>Type a message or upload audio to get started</p></section>
  {:else}
    {#each messages as message (message.id)}<ChatMessage {message} />{/each}
  {/if}
</div>

<style>
  .messages { flex:1 1 0; min-height:0; overflow-y:auto; overscroll-behavior:contain; padding:3rem clamp(1rem,4vw,4rem) 14rem; scrollbar-gutter:stable; }
  .welcome { text-align:center; margin:5vh auto 0; }
  h1 { font-size:clamp(2.2rem,5vw,3.5rem); letter-spacing:-.05em; margin:.5rem 0 1rem; }
  p { color:var(--muted); font-size:clamp(1rem,2vw,1.35rem); }
</style>
