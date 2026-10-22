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
    <section class="welcome" aria-label="mlx-one local chat">
      <p class="eyebrow">MLX / LOCAL</p>
      <h1>Your Mac.<br />Your model.</h1>
      <p class="summary">Native generation through mlx-one—private, direct, and built for Apple Silicon.</p>
    </section>
  {:else}
    {#each messages as message (message.id)}<ChatMessage {message} />{/each}
  {/if}
</div>

<style>
  .messages { flex:1 1 0; min-height:0; overflow-y:auto; overscroll-behavior:contain; padding:3rem clamp(1rem,4vw,4rem) 14rem; scrollbar-gutter:stable; }
  .welcome { position:relative; z-index:0; max-width:78rem; margin:clamp(1.5rem,7vh,5rem) auto 0; padding:0 clamp(.5rem,2vw,1.5rem); }
  .eyebrow { margin:0 0 clamp(2.5rem,8vh,6rem); color:var(--accent); font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:clamp(.78rem,1.3vw,1.35rem); font-weight:800; letter-spacing:.22em; }
  h1 { max-width:14ch; margin:0; color:var(--ink); font-size:clamp(2.4rem,7.5vw,6.6rem); font-weight:500; letter-spacing:-.09em; line-height:.93; }
  .summary { max-width:58rem; margin:clamp(2rem,6vh,5rem) 0 0; color:var(--muted); font-size:clamp(1rem,2.3vw,2rem); letter-spacing:-.035em; line-height:1.45; }
  @media (max-width:680px) { .messages { padding-top:1.5rem; } .welcome { margin-top:1rem; } .eyebrow { margin-bottom:2.5rem; } h1 { font-size:clamp(2.04rem,10.8vw,3.9rem); } .summary { margin-top:2rem; font-size:1.05rem; } }
</style>
