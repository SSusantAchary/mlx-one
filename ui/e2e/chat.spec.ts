import { expect, test } from '@playwright/test';

test('streams a response and stops a second generation', async ({ page }) => {
  await page.goto('/');
  await expect(page.locator('.model')).toContainText('test/native');

  const composer = page.getByLabel('Message', { exact: true });
  await composer.fill('hello');
  await page.getByLabel('Send message', { exact: true }).click();
  await expect(page.locator('article[aria-label="assistant message"]')).toContainText(
    'Hello from native MLX'
  );

  await composer.fill('slow response');
  await page.getByLabel('Send message', { exact: true }).click();
  const responses = page.locator('article[aria-label="assistant message"]');
  await expect(responses.nth(1)).toContainText('tick');
  await page.getByLabel('Stop generation', { exact: true }).click();
  await expect(page.getByLabel('Send message', { exact: true })).toBeVisible();
});

test('keeps long generated output in a scrolling message viewport', async ({ page }) => {
  await page.goto('/');
  const composer = page.getByLabel('Message', { exact: true });
  await composer.fill('long output');
  await page.getByLabel('Send message', { exact: true }).click();
  await expect(page.locator('article[aria-label="assistant message"]')).toContainText(
    'Response line 99'
  );

  const dimensions = await page.locator('.messages').evaluate((element) => ({
    clientHeight: element.clientHeight,
    scrollHeight: element.scrollHeight,
    scrollTop: element.scrollTop,
    overflowY: getComputedStyle(element).overflowY
  }));
  expect(dimensions.overflowY).toBe('auto');
  expect(dimensions.scrollHeight).toBeGreaterThan(dimensions.clientHeight);
  expect(dimensions.scrollTop).toBeGreaterThan(0);
});

test('uploads audio and inserts an editable local transcript', async ({ page }) => {
  await page.goto('/');
  await page.getByLabel('Upload audio file').setInputFiles({
    name: 'recording.webm',
    mimeType: 'audio/webm',
    buffer: Buffer.from('fake browser audio')
  });
  const composer = page.getByLabel('Message', { exact: true });
  await expect(composer).toHaveValue('editable local transcript');
  await expect(page.locator('article[aria-label="user message"]')).toHaveCount(0);
});

test('retains per-turn metrics, shows context, and branches conversations', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 1000 });
  let count = 0;
  await page.route('**/v1/chat/completions', async route => {
    count += 1;
    const output = count === 1 ? 7 : 11;
    const chunks = [
      { choices: [{ delta: { content: `Answer ${count}` } }] },
      { choices: [{ delta: {}, finish_reason: 'stop' }], mlx: { prompt_tokens: 20, actual_prefill_tokens: 15, reused_prompt_tokens: 5, generated_tokens: output, context_used: 20+output, prefill_ms: 100, decode_ms: 200, generation_latency_ms: 500, tokens_per_second: output*2, decode_tokens_per_second: 35, prompt_tokens_per_second: 150, time_to_first_token_ms: 100 } }
    ];
    await route.fulfill({ contentType: 'text/event-stream', body: chunks.map(c => `data: ${JSON.stringify(c)}\n\n`).join('') + 'data: [DONE]\n\n' });
  });
  await page.goto('/');
  const composer = page.getByLabel('Message', { exact: true });
  await composer.fill('first question'); await page.getByLabel('Send message', { exact: true }).click();
  const answers = page.locator('article.assistant');
  await expect(answers.first()).toContainText('7 tokens');
  await composer.fill('second question'); await page.getByLabel('Send message', { exact: true }).click();
  await expect(answers.nth(1)).toContainText('11 tokens');
  await expect(answers.first()).toContainText('7 tokens');
  await page.getByLabel('Context usage', { exact: true }).click();
  const details = page.getByLabel('Token usage details');
  await expect(details).toContainText('31 / 4,096');
  await expect(details).toContainText('Reused prefix');
  await page.screenshot({ path: '../local/ui-redesign-desktop.png', fullPage: true, animations: 'disabled' });
  await page.getByLabel('Close context details').click();
  await answers.first().getByLabel('Branch conversation').click();
  await expect(answers).toHaveCount(1);
  await expect(answers.first()).toContainText('7 tokens');
  await expect(page.locator('.chat-tab')).toHaveCount(2);
  await page.locator('.chat-tab').first().locator('button').first().click();
  await expect(answers).toHaveCount(2);
  await page.getByRole('button', { name: 'Search', exact: true }).click();
  await page.getByLabel('Search conversations').fill('no such question');
  await expect(page.getByText('No conversations found.')).toBeVisible();
});

test('edits and regenerates a user turn, deletes a branch, and exposes model details', async ({ page }) => {
  await page.goto('/');
  const composer = page.getByLabel('Message', { exact: true });
  await composer.fill('original prompt'); await page.getByLabel('Send message', { exact: true }).click();
  await expect(page.locator('article.assistant')).toContainText('Hello from native MLX');
  const user = page.locator('article[aria-label="user message"]');
  await user.getByLabel('Edit message', { exact: true }).click();
  await user.getByRole('textbox', { name: 'Edit message', exact: true }).fill('edited prompt');
  await user.getByRole('button', { name: 'Save & send' }).click();
  await expect(page.locator('article.assistant')).toContainText('Hello from native MLX');
  await expect(user).toContainText('edited prompt');
  await expect(page.locator('article.assistant')).toHaveCount(1);
  await page.locator('article.assistant').getByLabel('Regenerate response').click();
  await expect(page.locator('article.assistant')).toContainText('Hello from native MLX');
  await page.locator('article.assistant').getByLabel('Delete from here').click();
  await expect(page.locator('article.assistant')).toHaveCount(0);
  await page.getByRole('button', { name: 'Models', exact: true }).click();
  await expect(page.getByRole('dialog')).toContainText('test/native');
  await page.keyboard.press('Escape');
  await expect(page.getByRole('dialog')).toHaveCount(0);
});

test('mobile navigation remains usable without horizontal overflow', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/');
  await page.getByLabel('Open sidebar').click();
  await expect(page.getByRole('navigation').getByRole('button', { name: 'New chat', exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'Models', exact: true }).click();
  await expect(page.getByRole('dialog')).toContainText('Connected models');
  await page.getByLabel('Close panel', { exact: true }).last().click();
  await expect.poll(async () => (await page.locator('.navigation').boundingBox())?.x).toBeLessThan(-250);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  expect(await page.locator('.bar').evaluate(el => el.scrollWidth <= el.clientWidth + 1)).toBe(true);
  await page.screenshot({ path: '../local/ui-redesign-mobile.png', fullPage: true, animations: 'disabled' });
});

test('keeps server details when a response finishes before initial runtime lookup', async ({ page }) => {
  let release!: () => void;
  const ready = new Promise<void>(resolve => { release = resolve; });
  await page.route('**/v1/runtime', async route => {
    await ready;
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify({
      backend: 'mlx', device: 'gpu', chip: 'Apple M4', tokens_per_second: 0,
      scheduler: { active_slots: 0, parallel_slots: 1, queue_depth: 0, queue_capacity: 8 },
      server: { cache_type_k: 'bf16', cache_type_v: 'bf16' }
    }) });
  });
  await page.route('**/v1/chat/completions', async route => {
    await route.fulfill({ contentType: 'text/event-stream', body:
      'data: {"choices":[{"delta":{"content":"done"}}]}\n\n' +
      'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"mlx":{"generated_tokens":8,"tokens_per_second":32,"decode_tokens_per_second":35}}\n\n' +
      'data: [DONE]\n\n' });
  });
  await page.goto('/');
  await expect(page.locator('.model')).toContainText('test/native');
  await page.getByLabel('Message', { exact: true }).fill('quick response');
  await page.getByLabel('Send message', { exact: true }).click();
  await expect(page.locator('article.assistant')).toContainText('8 tokens');
  release();
  const footer = page.getByLabel('Runtime statistics');
  await expect(footer).toContainText('Apple M4');
  await expect(footer).toContainText('KV bf16/bf16');
  await expect(footer).toContainText('overall 32.0 tok/s');
});

test('shows hidden-token progress and can retry a blank length-limited response', async ({ page }) => {
  const requests: { messages: { role: string; content: string }[]; max_tokens: number }[] = [];
  page.on('request', request => {
    if (request.url().endsWith('/v1/chat/completions')) requests.push(request.postDataJSON());
  });
  await page.goto('/');
  await expect(page.locator('.model')).toContainText('test/native');
  await page.getByLabel('Message', { exact: true }).fill('reasoning-only request');
  await page.getByLabel('Send message', { exact: true }).click();
  await expect(page.locator('article.assistant')).toContainText('Thinking…');
  await expect(page.locator('article.assistant')).toContainText('64 tokens generated');
  await expect(page.locator('article.assistant')).toContainText('Output limit reached before an answer was produced');
  await page.getByRole('button', { name: 'Retry with more tokens' }).click();
  await expect(page.locator('article.assistant')).toContainText('Hello from native MLX');
  await expect(page.locator('article.assistant')).toHaveCount(1);
  expect(requests).toHaveLength(2);
  expect(requests[0].max_tokens).toBe(512);
  expect(requests[1].max_tokens).toBe(2048);
  expect(requests[1].messages.filter(m => m.role === 'assistant')).toHaveLength(0);
  await expect(page.getByLabel('Send message', { exact: true })).toBeVisible();
});

test('uses GPT-OSS coding defaults while preserving a manually selected budget', async ({ page }) => {
  const requests: { max_tokens: number; reasoning_effort?: string }[] = [];
  await page.route('**/v1/models', async route => {
    await route.fulfill({ contentType: 'application/json', body: JSON.stringify({data:[{
      id:'test/native',object:'model',owned_by:'local',mlx:{architecture:'gpt_oss',parameters:null,
      quantization:{mode:'mxfp4',bits:4},context_length:16384,revision:null}
    }]}) });
  });
  await page.route('**/v1/chat/completions', async route => {
    await route.fulfill({contentType:'text/event-stream',body:
      'data: {"choices":[{"delta":{"content":"Hello from native MLX"}}]}\n\n' +
      'data: {"choices":[{"delta":{},"finish_reason":"stop"}],"mlx":{"generated_tokens":1}}\n\n' +
      'data: [DONE]\n\n'});
  });
  page.on('request', r => { if(r.url().endsWith('/v1/chat/completions')) requests.push(r.postDataJSON()); });
  await page.goto('/');
  await expect(page.locator('.model')).toContainText('test/native');
  await page.getByRole('button', {name:'Settings',exact:true}).click();
  await expect(page.getByLabel('Max tokens', {exact:true})).toHaveValue('8192');
  await expect(page.getByLabel('Reasoning effort', {exact:true})).toHaveValue('low');
  await page.getByLabel('Close panel', {exact:true}).last().click();
  await page.getByLabel('Message', {exact:true}).fill('hello');
  await page.getByLabel('Send message', {exact:true}).click();
  await expect(page.locator('article.assistant')).toContainText('Hello from native MLX');
  expect(requests[0]).toMatchObject({max_tokens:8192,reasoning_effort:'low'});
  await page.getByRole('button', {name:'Settings',exact:true}).click();
  await page.getByLabel('Max tokens', {exact:true}).fill('512');
  await page.getByRole('button', {name:'Connect',exact:true}).click();
  await expect(page.getByLabel('Max tokens', {exact:true})).toHaveValue('512');
});
