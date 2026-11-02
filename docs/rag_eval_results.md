# RAG evaluation results

Command: `python main.py rag-eval --provider ollama` (2026-11-02)

- Answer generator: local `qwen2.5:7b` via Ollama (Q4_K_M, GTX 1650 Ti 4 GB + CPU),
  about 2-3 minutes per question. **The Claude API path was not run live**
  (no API key); it is covered by unit tests with a fake client.
- Corpus: 377 papers, 148 with PDF text, ~7,600 chunks, `all-MiniLM-L6-v2` embeddings.
- Questions: `config/rag_test_questions.yaml`; pass rules: `src/cyberresearch/rag/evaluation.py`.

| id | category | best sim. | status | evidence claims | verified citations | rejected citations | pass |
|---|---|---:|---|---:|---:|---:|:---:|
| ids-gnn | sufficient | 0.59 | answered | 4 | 4 | 0 | yes |
| ids-drift | sufficient | 0.57 | answered | 2 | 3 | 0 | yes |
| ai-threat-hunting | sufficient | 0.76 | partial | 2 | 2 | 0 | yes |
| ai-rl-vulns | sufficient | 0.79 | answered | 4 | 4 | 1 | yes |
| ns-datasets | sufficient | 0.62 | answered | 1 | 1 | 0 | yes |
| vi-llm-vulns (Vietnamese) | sufficient | 0.58 | partial | 2 | 2 | 0 | yes |
| weak-bgp | weak | 0.41 | insufficient_evidence | 0 | 0 | 0 | yes |
| weak-5g | weak | 0.55 | insufficient_evidence | 0 | 0 | 0 | yes |
| weak-pqc | weak | 0.69 | answered | 3 | 3 | 0 | yes |
| out-cake | outside | 0.19 | insufficient_evidence (gated, no LLM call) | 0 | 0 | 0 | yes |
| out-football | outside | 0.17 | insufficient_evidence (gated, no LLM call) | 0 | 0 | 0 | yes |
| out-riemann | outside | 0.18 | insufficient_evidence (gated, no LLM call) | 0 | 0 | 0 | yes |
| amb-poisoning | ambiguous | 0.43 | partial | 2 | 3 | 0 | yes |
| amb-drift | ambiguous | 0.42 | partial | 2 | 3 | 0 | yes |
| amb-agents | ambiguous | 0.56 | insufficient_evidence | 0 | 1 | 1 | yes |

**15/15 passed.**

## Observations

- **Similarity gate (0.30):** out-of-corpus questions scored 0.17-0.19; the
  lowest in-domain English question scored 0.41.
- **Citation validation caught model errors:** 2 citations whose quote did not
  occur in the cited chunk were rejected (ai-rl-vulns, amb-agents); no
  unverified citation reached the user.
- **Vietnamese questions:** the embedding model is English-only. Before query
  translation, the Vietnamese question retrieved an unrelated Indonesian paper
  (similarity 0.34); with translation it reaches 0.58 and the right papers.
- **Issues found and fixed during the runs:** `qwen2.5:7b` once answered a
  Vietnamese question in Chinese (answer language is now stated explicitly), and
  labelled a BGP question `out_of_scope` instead of `insufficient_evidence`
  (status definitions clarified in the prompt). An earlier run of the
  Vietnamese question ended with an unexplained error that did not reproduce;
  generation errors are now logged.
- **Pass rules test safety, not answer quality:** they check that answers are
  cited, that unsupported questions are not answered, and that ambiguity is
  flagged. They do not measure completeness or correctness; that is the
  evaluation planned for Phase 4.
