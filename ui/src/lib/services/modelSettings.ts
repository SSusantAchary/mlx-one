import type { ModelInfo, Settings } from '$lib/types/api';

/** UI defaults for reasoning models; explicit output limits always win. */
export function modelDefaults(settings: Settings, model: ModelInfo['mlx'] | undefined, budgetCustomized = false): Settings {
  if (model?.architecture !== 'gpt_oss') return settings;
  const recommended = Math.max(1, Math.min(8192, model.context_length - Math.min(512, Math.floor(model.context_length / 2))));
  return {
    ...settings,
    max_tokens: !budgetCustomized && settings.max_tokens === 512 ? recommended : settings.max_tokens,
    reasoning_effort: settings.reasoning_effort ?? 'low'
  };
}
