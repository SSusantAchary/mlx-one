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
