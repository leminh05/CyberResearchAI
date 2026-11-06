// API responses for tests, in the shape of web/schemas.py.

export const reference = (over = {}) => ({
  number: 1,
  title: "GNN Intrusion Detection",
  authors: ["Ann Lee", "Bo Tran"],
  year: 2024,
  url: "https://doi.org/10.1/gnn",
  doi: "10.1/gnn",
  arxiv_id: null,
  section: "2 Method",
  location: "p. 3",
  content_source: "pdf",
  paper_id: 1,
  chunk_id: 10,
  similarity: 0.71,
  ...over,
});

export const answer = (over = {}) => ({
  conversation_id: "11111111-1111-4111-8111-111111111111",
  question: "How are GNNs used?",
  status: "answered",
  summary: "GNNs model network flows.",
  claims: [
    {
      statement: "GNNs detect intrusions from flow graphs.",
      kind: "evidence",
      citations: [{ reference: 1, quote: "graph neural networks detect intrusion" }],
    },
  ],
  references: [reference()],
  ambiguity_note: "",
  limitations: "",
  follow_ups: ["What about concept drift?"],
  warnings: [],
  meta: { gated: false, evidence_count: 8, best_similarity: 0.71, model: "fake-model", retrieval_query: "q" },
  ...over,
});
