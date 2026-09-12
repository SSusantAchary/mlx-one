import { expect, test } from '@playwright/test';

test('streams a response and stops a second generation', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByLabel('Loaded model', { exact: true })).toHaveValue('test/native');

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
