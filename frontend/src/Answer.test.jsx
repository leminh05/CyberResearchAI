import { render, screen, within } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import Answer from "./Answer.jsx";
import { answer, reference } from "./testData.js";

const show = (data) => render(<Answer answer={data} idPrefix="m1" onFollowUp={vi.fn()} />);

describe("citation rendering", () => {
  it("links markers to references and titles to the paper, with verified quotes", () => {
    show(answer());
    const mark = screen.getByRole("link", { name: "Reference 1" });
    expect(mark).toHaveAttribute("href", "#m1-ref-1");
    expect(document.getElementById("m1-ref-1")).toBeInTheDocument();

    const paper = screen.getByRole("link", { name: "GNN Intrusion Detection" });
    expect(paper).toHaveAttribute("href", "https://doi.org/10.1/gnn");
    expect(paper).toHaveAttribute("target", "_blank");
    expect(paper).toHaveAttribute("rel", "noopener noreferrer");

    const card = screen.getByTestId("reference");
    expect(within(card).getByText(/graph neural networks detect intrusion/)).toBeInTheDocument();
    expect(within(card).getByText("Verified quote")).toBeInTheDocument();
    expect(card).toHaveTextContent("Ann Lee, Bo Tran · 2024 · 2 Method, p. 3");
  });

  it("never renders a non-http link", () => {
    show(answer({ references: [reference({ url: "javascript:alert(1)" })] }));
    expect(screen.queryByRole("link", { name: /GNN/ })).not.toBeInTheDocument();
    expect(screen.getByText(/no link available/)).toBeInTheDocument();
  });

  it("shows a reference without url as plain text", () => {
    show(answer({ references: [reference({ url: null, title: null })] }));
    expect(screen.getByText(/Untitled/)).toBeInTheDocument();
    expect(screen.getByText(/no link available/)).toBeInTheDocument();
  });

  it("does not repeat a marker or quote cited twice", () => {
    const citation = { reference: 1, quote: "same quote of several words here" };
    show(answer({ claims: [{ statement: "Claim.", kind: "evidence", citations: [citation, citation] }] }));
    expect(screen.getAllByRole("link", { name: "Reference 1" })).toHaveLength(1);
    expect(screen.getAllByText(/same quote of several words/)).toHaveLength(1);
  });

  it("abbreviates long author lists", () => {
    show(answer({ references: [reference({ authors: ["A", "B", "C", "D"] })] }));
    expect(screen.getByTestId("reference")).toHaveTextContent("A, B, C et al.");
  });

  it("renders model text as text, not HTML", () => {
    show(answer({ claims: [{ statement: "<img src=x onerror=alert(1)>", kind: "evidence", citations: [] }] }));
    expect(document.querySelector("img")).toBeNull();
    expect(screen.getByText(/<img src=x/)).toBeInTheDocument();
  });

  it("separates interpretation from findings and labels the summary as unchecked", () => {
    show(answer({
      claims: [
        { statement: "A finding.", kind: "evidence", citations: [{ reference: 1, quote: "a quote of five words" }] },
        { statement: "My reasoning.", kind: "interpretation", citations: [] },
      ],
    }));
    expect(screen.getByText("Findings from the papers")).toBeInTheDocument();
    expect(screen.getByText(/^Interpretation/)).toBeInTheDocument();
    expect(screen.getByText("Model summary (not citation-checked)")).toBeInTheDocument();
  });

  it("shows an abstention without findings or references", () => {
    show(answer({
      status: "insufficient_evidence",
      summary: "No passage matches this question.",
      claims: [],
      references: [],
      meta: { gated: true, evidence_count: 0, best_similarity: 0.1, model: null, retrieval_query: "q" },
    }));
    expect(screen.getByText("Insufficient evidence")).toBeInTheDocument();
    expect(screen.getByText("No passage matches this question.")).toBeInTheDocument();
    expect(screen.queryByText("References")).not.toBeInTheDocument();
    expect(screen.getByText(/no model called/)).toBeInTheDocument();
  });

  it("shows validation warnings", () => {
    show(answer({ status: "partial", warnings: ["rejected citation E2: quote not found in the cited chunk"] }));
    expect(screen.getByText("Partial")).toBeInTheDocument();
    expect(screen.getByText(/rejected citation E2/)).toBeInTheDocument();
  });
});
