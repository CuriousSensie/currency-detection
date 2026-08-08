import { expect, test } from "@playwright/test";

test("shows an honest unavailable-model state", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: /One note/i })).toBeVisible();
  await expect(page.getByText(/No predictions are simulated/i)).toBeVisible();
  await expect(page.getByRole("button", { name: "Classify note" })).toBeDisabled();
});

test("methodology states the single-note boundary", async ({ page }) => {
  await page.goto("/methodology");
  await expect(page.getByText(/one denomination/i)).toBeVisible();
});
