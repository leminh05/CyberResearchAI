import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App.jsx";
import { answer } from "./testData.js";

const HEALTH = { status: "ok", database: true, llm: true, provider: "ollama", papers: 377, chunks: 7691 };
const json = (body, status = 200) => new Response(body === null ? null : JSON.stringify(body), { status });

// Routes fetch by URL; chat replies are queued per test.
let chatReplies;
let chatBodies;
let deleted;

beforeEach(() => {
  chatReplies = [];
  chatBodies = [];
  deleted = [];
  vi.stubGlobal("fetch", vi.fn((url, options) => {
    if (url === "/api/health") return Promise.resolve(json(HEALTH));
    if (options?.method === "DELETE") {
      deleted.push(url);
      return Promise.resolve(json(null, 204));
    }
    chatBodies.push(JSON.parse(options.body));
    const reply = chatReplies.shift();
    return typeof reply === "function" ? reply(options) : Promise.resolve(reply);
  }));
});

afterEach(() => vi.unstubAllGlobals());

async function ask(user, text) {
  await user.type(screen.getByLabelText("Your question"), text);
  await user.click(screen.getByRole("button", { name: "Ask" }));
}

describe("chat interface", () => {
  it("sends a question and shows history with the cited answer", async () => {
    const user = userEvent.setup();
    chatReplies.push(json(answer()));
    render(<App />);
    await ask(user, "How are GNNs used?");

    expect(screen.getByText("How are GNNs used?")).toBeInTheDocument();
    expect(await screen.findByRole("link", { name: "GNN Intrusion Detection" })).toBeInTheDocument();
    expect(chatBodies[0]).toEqual({ question: "How are GNNs used?", conversation_id: null });
    expect(screen.getByLabelText("Your question")).toHaveValue("");
  });

  it("sends the conversation id with follow-ups and keeps earlier messages", async () => {
    const user = userEvent.setup();
    chatReplies.push(json(answer()), json(answer({ summary: "Second answer." })));
    render(<App />);
    await ask(user, "First?");
    await screen.findByText("GNNs model network flows.");
    await ask(user, "Second?");
    await screen.findByText("Second answer.");

    expect(chatBodies[1].conversation_id).toBe(answer().conversation_id);
    expect(screen.getByText("First?")).toBeInTheDocument();
    expect(screen.getByText("GNNs model network flows.")).toBeInTheDocument();
  });

  it("shows a loading state and blocks sending while waiting", async () => {
    const user = userEvent.setup();
    let finish;
    chatReplies.push(() => new Promise((resolve) => { finish = resolve; }));
    render(<App />);
    await ask(user, "Slow?");

    expect(screen.getByRole("status")).toHaveTextContent(/Searching papers/);
    await user.type(screen.getByLabelText("Your question"), "Another");
    expect(screen.getByRole("button", { name: "Ask" })).toBeDisabled();

    finish(json(answer()));
    await waitFor(() => expect(screen.queryByRole("status")).not.toBeInTheDocument());
    expect(chatBodies).toHaveLength(1);
  });

  it("submits with Enter but not Shift+Enter, and ignores blank input", async () => {
    const user = userEvent.setup();
    chatReplies.push(json(answer()));
    render(<App />);
    const box = screen.getByLabelText("Your question");
    await user.type(box, "   {Enter}");
    expect(chatBodies).toHaveLength(0);
    await user.clear(box);
    await user.type(box, "Line one{Shift>}{Enter}{/Shift}line two");
    expect(chatBodies).toHaveLength(0);
    await user.type(box, "{Enter}");
    await waitFor(() => expect(chatBodies).toHaveLength(1));
    expect(chatBodies[0].question).toBe("Line one\nline two");
  });

  it("shows the server's error message and keeps the question for a retry", async () => {
    const user = userEvent.setup();
    chatReplies.push(json({ error: { code: "rate_limited", message: "Too many questions. Try again in 5 s." } }, 429));
    render(<App />);
    await ask(user, "Limited?");

    expect(await screen.findByRole("alert")).toHaveTextContent("Too many questions");
    expect(screen.getByLabelText("Your question")).toHaveValue("Limited?");
  });

  it("explains a network failure", async () => {
    const user = userEvent.setup();
    chatReplies.push(() => Promise.reject(new TypeError("Failed to fetch")));
    render(<App />);
    await ask(user, "Offline?");
    expect(await screen.findByRole("alert")).toHaveTextContent("Cannot reach the server");
  });

  it("starts a new conversation after the old one expired", async () => {
    const user = userEvent.setup();
    chatReplies.push(
      json(answer()),
      json({ error: { code: "conversation_not_found", message: "This conversation has expired." } }, 404),
      json(answer()),
    );
    render(<App />);
    await ask(user, "One?");
    await screen.findByText("GNNs model network flows.");
    await ask(user, "Two?");
    await screen.findByRole("alert");
    await user.click(screen.getByRole("button", { name: "Ask" }));
    await waitFor(() => expect(chatBodies).toHaveLength(3));
    expect(chatBodies[2]).toEqual({ question: "Two?", conversation_id: null });
  });

  it("clear removes messages, forgets the conversation on the server and starts fresh", async () => {
    const user = userEvent.setup();
    chatReplies.push(json(answer()), json(answer()));
    render(<App />);
    await ask(user, "Remember me?");
    await screen.findByText("GNNs model network flows.");

    await user.click(screen.getByRole("button", { name: "Clear conversation" }));
    expect(screen.queryByText("Remember me?")).not.toBeInTheDocument();
    expect(deleted).toEqual([`/api/conversations/${answer().conversation_id}`]);

    await ask(user, "New?");
    await waitFor(() => expect(chatBodies).toHaveLength(2));
    expect(chatBodies[1].conversation_id).toBeNull();
  });

  it("clear during loading aborts the request and shows no late answer", async () => {
    const user = userEvent.setup();
    let signal;
    chatReplies.push((options) => new Promise((_, reject) => {
      signal = options.signal;
      signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
    }));
    render(<App />);
    await ask(user, "Cancel me?");
    await user.click(screen.getByRole("button", { name: "Clear conversation" }));

    expect(signal.aborted).toBe(true);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("a follow-up suggestion fills the input without sending", async () => {
    const user = userEvent.setup();
    chatReplies.push(json(answer()));
    render(<App />);
    await ask(user, "Q?");
    await user.click(await screen.findByRole("button", { name: "What about concept drift?" }));
    expect(screen.getByLabelText("Your question")).toHaveValue("What about concept drift?");
    expect(chatBodies).toHaveLength(1);
  });

  it("reports an unavailable backend from the health check", async () => {
    fetch.mockImplementationOnce(() => Promise.resolve(json({ status: "unavailable", database: false, llm: true }, 503)));
    render(<App />);
    expect(await screen.findByText("Database unavailable")).toBeInTheDocument();
  });
});
