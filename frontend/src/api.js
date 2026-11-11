// HTTP client. All answers and citations are produced and validated by the backend.

export class ApiError extends Error {
  constructor(code, message) {
    super(message);
    this.code = code;
  }
}

async function request(path, options) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw new ApiError("network", "Cannot reach the server. Check that it is running, then try again.");
  }
  const body = await response.json().catch(() => null);
  if (!response.ok && !body?.status) {
    throw new ApiError(
      body?.error?.code ?? "http_error",
      body?.error?.message ?? `The server answered with HTTP ${response.status}.`,
    );
  }
  return body;
}

export function askQuestion(question, conversationId, signal, provider) {
  return request("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, conversation_id: conversationId, provider }),
    signal,
  });
}

export function deleteConversation(conversationId) {
  return request(`/api/conversations/${conversationId}`, { method: "DELETE" });
}

export function getHealth() {
  return request("/api/health");
}
