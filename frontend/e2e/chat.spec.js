// Real website, API, retrieval and citation validation; scripted LLM (tests/e2e_server.py).
import { expect, test } from "@playwright/test";

const QUESTION = "How are graph neural networks used for intrusion detection?";

async function ask(page, question) {
  await page.getByLabel("Your question").fill(question);
  await page.getByRole("button", { name: "Ask" }).click();
}

test("asks a question and gets an answer with a verifiable citation", async ({ page }) => {
  const problems = [];
  page.on("console", (m) => m.type() === "error" && problems.push(m.text()));
  page.on("pageerror", (e) => problems.push(e.message));

  await page.goto("/");
  await expect(page.getByText(/\d+ papers · scripted/)).toBeVisible();
  await ask(page, QUESTION);
  await expect(page.getByRole("status")).toBeVisible();

  const answer = page.getByRole("article", { name: "Answer" });
  await expect(answer.getByText("Answered", { exact: true })).toBeVisible();
  await expect(answer.getByText("Scripted finding taken")).toBeVisible();
  await expect(answer.getByText("scripted-generator")).toBeVisible();

  const card = answer.getByTestId("reference");
  await expect(card.getByText("Verified quote")).toBeVisible();
  const quote = await card.locator("blockquote").innerText();
  expect(quote.split(/\s+/).length).toBeGreaterThanOrEqual(5);

  const paper = card.locator(".reference-title a");
  await expect(paper).toHaveAttribute("href", /^https?:\/\//);
  await expect(paper).toHaveAttribute("target", "_blank");

  await answer.getByRole("link", { name: "Reference 1" }).click();
  await expect(page).toHaveURL(/#m\d+-ref-1$/);
  expect(problems).toEqual([]);
});

test("follow-up keeps context, and clear starts a new conversation", async ({ page }) => {
  await page.goto("/");
  await ask(page, QUESTION);
  const answers = page.getByRole("article", { name: "Answer" });
  await expect(answers).toHaveCount(1);

  let followUp;
  page.on("response", async (r) => {
    if (r.url().endsWith("/api/chat")) followUp = await r.json();
  });
  await page.getByRole("button", { name: "Scripted follow-up?" }).click();
  await expect(page.getByLabel("Your question")).toHaveValue("Scripted follow-up?");
  await page.getByRole("button", { name: "Ask" }).click();
  await expect(answers).toHaveCount(2);
  await expect.poll(() => followUp?.meta.retrieval_query).toContain(QUESTION);
  await expect(page.getByText(QUESTION)).toBeVisible();

  const previous = followUp.conversation_id;
  await page.getByRole("button", { name: "Clear conversation" }).click();
  await expect(answers).toHaveCount(0);
  await expect(page.getByText("Ask a research question")).toBeVisible();

  await ask(page, QUESTION);
  await expect(answers).toHaveCount(1);
  await expect.poll(() => followUp.conversation_id).not.toBe(previous);
  expect(followUp.meta.retrieval_query).toBe(QUESTION);
});

test("off-topic question is declined without calling the model", async ({ page }) => {
  await page.goto("/");
  await ask(page, "What is a good recipe for chocolate cake?");
  const answer = page.getByRole("article", { name: "Answer" });
  await expect(answer.getByText("Insufficient evidence")).toBeVisible();
  await expect(answer.getByText("no model called")).toBeVisible();
  await expect(answer.getByTestId("reference")).toHaveCount(0);
});

test("a fabricated quote never appears as a citation", async ({ page }) => {
  await page.goto("/");
  await ask(page, `${QUESTION} simulate fabricated quote`);
  const answer = page.getByRole("article", { name: "Answer" });
  await expect(answer.getByText("Insufficient evidence")).toBeVisible();
  await expect(answer.getByTestId("reference")).toHaveCount(0);
  await expect(page.getByText("invented by the model")).toHaveCount(0);
  await answer.getByText(/Validation notes/).click();
  await expect(answer.getByText(/quote not found in the cited chunk/)).toBeVisible();
});

test("model failure shows a safe error and keeps the question", async ({ page }) => {
  await page.goto("/");
  const question = `${QUESTION} simulate llm failure`;
  await ask(page, question);
  const alert = page.getByRole("alert");
  await expect(alert).toHaveText("The answer could not be generated. Try again.");
  await expect(page.getByLabel("Your question")).toHaveValue(question);
  await expect(page.getByText("internal-address")).toHaveCount(0);
});

test("no secret reaches the browser", async ({ page, request }) => {
  const bodies = [];
  page.on("response", async (r) => bodies.push(await r.text().catch(() => "")));
  await page.goto("/");
  await expect(page.getByText(/\d+ papers/)).toBeVisible();
  bodies.push(await (await request.get("/api/health")).text());
  const everything = bodies.join("\n");
  expect(everything.length).toBeGreaterThan(1000);
  expect(everything).not.toMatch(/postgres(ql)?:\/\/|sk-ant|ANTHROPIC_API_KEY|POSTGRES_PASSWORD/);
});
