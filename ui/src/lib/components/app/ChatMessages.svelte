<script lang="ts">
  import ChatMessage from './ChatMessage.svelte';
  import type { Message } from '$lib/types/api';
  let { messages } = $props<{ messages: Message[] }>();
</script>

<div class="messages" aria-live="polite">
  {#if messages.length === 0}
    <section class="welcome"><span>MLX / LOCAL</span><h1>Your Mac.<br />Your model.</h1><p>Native generation through mlx-one—private, direct, and built for Apple Silicon.</p></section>
  {:else}
    {#each messages as message (message.id)}<ChatMessage {message} />{/each}
  {/if}
</div>

<style>
  .messages { flex:1; overflow:auto; padding:3rem clamp(1rem,4vw,4rem) 9rem; }
  .welcome { max-width:50rem; margin:12vh auto; }
  .welcome span { color:var(--accent); font:700 .72rem ui-monospace,monospace; letter-spacing:.16em; }
  h1 { font-size:clamp(3rem,8vw,6.5rem); line-height:.88; letter-spacing:-.07em; margin:1rem 0 1.5rem; }
  p { color:var(--muted); max-width:31rem; line-height:1.7; }
</style>
