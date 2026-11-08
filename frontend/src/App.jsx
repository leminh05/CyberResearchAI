import { Fragment, useEffect, useRef, useState } from "react";
import { askQuestion, deleteConversation, getHealth } from "./api.js";
import Answer from "./Answer.jsx";

const MAX_QUESTION_CHARS = 2000; // the backend enforces this too

const EXAMPLES = [
  "How are graph neural networks used for intrusion detection?",
  "What defenses exist against prompt injection?",
  "How does concept drift affect network intrusion detectors?",
];

const PROVIDER_NAMES = { ollama: "Ollama (local)", gemini: "Gemini", claude: "Claude" };

function HealthBadge({ health }) {
  if (!health) return null;
  if (health.status === "ok") {
    return <span className="health health-ok">{health.papers} papers</span>;
  }
  const problem = !health.database ? "Database unavailable" : "Answer model unavailable";
  return <span className="health health-bad">{problem}</span>;
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [health, setHealth] = useState(null);
  const [provider, setProvider] = useState(null);
  const conversationId = useRef(null);
  const nextId = useRef(1);
  const inFlight = useRef(null);
  const inputRef = useRef(null);
  const lastMessageRef = useRef(null);

  useEffect(() => {
    getHealth()
      .then((body) => {
        setHealth(body);
        setProvider(body.provider);
      })
      .catch(() => setHealth({ status: "unavailable", database: false }));
  }, []);

  useEffect(() => {
    // Start of the newest message: a long answer must be read from its beginning.
    lastMessageRef.current?.scrollIntoView?.({ behavior: "smooth", block: "start" });
  }, [messages]);

  useEffect(() => {
    if (!loading) return undefined;
    setElapsed(0);
    const started = Date.now();
    const timer = setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => clearInterval(timer);
  }, [loading]);

  const add = (message) => setMessages((all) => [...all, { id: nextId.current++, ...message }]);

  async function send(event) {
    event?.preventDefault();
    const question = input.trim();
    if (!question || loading) return;
    add({ role: "user", text: question });
    setInput("");
    setLoading(true);
    const controller = new AbortController();
    inFlight.current = controller;
    try {
      const answer = await askQuestion(question, conversationId.current, controller.signal, provider ?? undefined);
      conversationId.current = answer.conversation_id;
      add({ role: "assistant", answer });
    } catch (error) {
      if (error.name === "AbortError") return;
      if (error.code === "conversation_not_found") conversationId.current = null;
      add({ role: "error", text: error.message });
      setInput((current) => current || question); // keep the question so it can be retried
    } finally {
      if (inFlight.current === controller) {
        inFlight.current = null;
        setLoading(false);
        inputRef.current?.focus();
      }
    }
  }

  function clearConversation() {
    inFlight.current?.abort();
    inFlight.current = null;
    if (conversationId.current) deleteConversation(conversationId.current).catch(() => {});
    conversationId.current = null;
    setMessages([]);
    setInput("");
    setLoading(false);
    inputRef.current?.focus();
  }

  function onFollowUp(text) {
    setInput(text);
    inputRef.current?.focus();
  }

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>CyberResearchAI</h1>
          <p className="tagline">Answers from indexed cybersecurity papers, with citations checked against the source text</p>
        </div>
        <div className="topbar-actions">
          <HealthBadge health={health} />
          {health?.providers?.length > 1 ? (
            <select
              aria-label="Answer model"
              value={provider ?? ""}
              onChange={(event) => setProvider(event.target.value)}
              disabled={loading}
            >
              {health.providers.map((name) => (
                <option key={name} value={name}>
                  {PROVIDER_NAMES[name] ?? name}
                </option>
              ))}
            </select>
          ) : (
            provider && <span className="health health-ok">{PROVIDER_NAMES[provider] ?? provider}</span>
          )}
          <button type="button" className="secondary" onClick={clearConversation} disabled={!messages.length && !loading}>
            Clear conversation
          </button>
        </div>
      </header>

      <main className="messages" role="log" aria-live="polite" aria-label="Conversation">
        {messages.length === 0 && !loading && (
          <div className="empty">
            <h2>Ask a research question</h2>
            <p>Follow-up questions keep the context of this conversation. Vietnamese is welcome.</p>
            <div className="follow-ups">
              {EXAMPLES.map((text) => (
                <button type="button" key={text} className="chip" onClick={() => onFollowUp(text)}>
                  {text}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((message, index) => (
          <Fragment key={message.id}>
            {index === messages.length - 1 && <div ref={lastMessageRef} />}
            {message.role === "user" && <p className="bubble-user">{message.text}</p>}
            {message.role === "error" && (
              <p className="error" role="alert">
                {message.text}
              </p>
            )}
            {message.role === "assistant" && (
              <Answer answer={message.answer} idPrefix={`m${message.id}`} onFollowUp={onFollowUp} />
            )}
          </Fragment>
        ))}

        {loading && (
          <p className="loading" role="status">
            <span className="spinner" aria-hidden="true" />
            Searching papers and checking citations… {elapsed} s
            {provider === "ollama" && elapsed >= 5 && (
              <span className="loading-hint"> A local model can take a few minutes.</span>
            )}
          </p>
        )}
      </main>

      <form className="composer" onSubmit={send}>
        <textarea
          ref={inputRef}
          aria-label="Your question"
          placeholder="Ask about cybersecurity research…"
          rows={2}
          maxLength={MAX_QUESTION_CHARS}
          value={input}
          onChange={(event) => setInput(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) send(event);
          }}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Ask
        </button>
      </form>
    </div>
  );
}
