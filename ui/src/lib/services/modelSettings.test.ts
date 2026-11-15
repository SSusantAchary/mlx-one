import { describe, expect, it } from 'vitest';
import { modelDefaults } from './modelSettings';
import type { ModelInfo, Settings } from '$lib/types/api';
const defaults: Settings = {temperature:0.7,top_p:0.9,max_tokens:512,api_key:'',reasoning:'auto',reasoning_budget:-1};
const model: ModelInfo['mlx'] = {architecture:'gpt_oss',parameters:null,quantization:null,context_length:16384,revision:null};
describe('reasoning model defaults', () => {
  it('gives GPT-OSS an answer budget and low effort', () => {
    expect(modelDefaults(defaults,model)).toMatchObject({max_tokens:8192,reasoning_effort:'low'});
  });
  it('preserves explicit budgets and effort', () => {
    expect(modelDefaults({...defaults,max_tokens:32},model,true).max_tokens).toBe(32);
    expect(modelDefaults(defaults,model,true).max_tokens).toBe(512);
    expect(modelDefaults({...defaults,reasoning_effort:'high'},model).reasoning_effort).toBe('high');
  });
  it('does not change other model defaults and bounds the recommendation', () => {
    expect(modelDefaults(defaults,{...model,architecture:'gpt2'})).toBe(defaults);
    expect(modelDefaults(defaults,{...model,context_length:2048}).max_tokens).toBe(1536);
    expect(modelDefaults(defaults,{...model,context_length:64}).max_tokens).toBe(32);
  });
});
