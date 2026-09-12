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
    <section class="welcome"><span>MLX / LOCAL</span><h1>Your Mac.<br />Your model.</h1><p>Native generation through mlx-one—private, direct, and built for Apple Silicon.</p></section>
  {:else}
    {#each messages as message (message.id)}<ChatMessage {message} />{/each}
  {/if}
</div>

<style>
  .messages { flex:1 1 0; min-height:0; overflow-y:auto; overscroll-behavior:contain; padding:3rem clamp(1rem,4vw,4rem) 9rem; scrollbar-gutter:stable; }
  .welcome { max-width:50rem; margin:12vh auto; }
  .welcome span { color:var(--accent); font:700 .72rem ui-monospace,monospace; letter-spacing:.16em; }
  h1 { font-size:clamp(3rem,8vw,6.5rem); line-height:.88; letter-spacing:-.07em; margin:1rem 0 1.5rem; }
  p { color:var(--muted); max-width:31rem; line-height:1.7; }
</style>
