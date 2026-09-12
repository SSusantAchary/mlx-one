<script lang="ts">
  import type { ModelInfo } from '$lib/types/api';
  let { models, value } = $props<{ models: ModelInfo[]; value: string }>();
</script>
<label><span>Loaded model</span><select value={value} aria-label="Loaded model">{#each models as model}<option value={model.id}>{model.id}</option>{/each}</select></label>
<style>label{display:grid;gap:.18rem}span{font-size:.62rem;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}select{max-width:19rem;border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:.65rem;padding:.5rem .7rem}</style>
