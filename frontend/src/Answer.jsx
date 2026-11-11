// Renders one validated answer from the API. Decides nothing about citations:
// every claim, quote and link shown here was checked by the backend.

const STATUS = {
  answered: ["Answered", "Supported by retrieved evidence", "good"],
  partial: ["Partial", "Only part of the question is supported", "warn"],
  insufficient_evidence: ["Insufficient evidence", "No supported answer", "warn"],
  out_of_scope: ["Out of scope", "Not a cybersecurity research question", "muted"],
  refused: ["Declined", "The model declined to answer", "muted"],
};

const isHttpUrl = (url) => /^https?:\/\//i.test(url ?? "");

function formatAuthors(authors) {
  if (!authors.length) return "Unknown authors";
  return authors.slice(0, 3).join(", ") + (authors.length > 3 ? " et al." : "");
}

function CitationMarks({ citations, idPrefix }) {
  const numbers = [...new Set(citations.map((c) => c.reference))];
  return numbers.map((n) => (
    <a key={n} className="mark" href={`#${idPrefix}-ref-${n}`} aria-label={`Reference ${n}`}>
      [{n}]
    </a>
  ));
}

function Reference({ reference, quotes, idPrefix }) {
  const { number, title, authors, year, url, section, location } = reference;
  const name = title || "Untitled";
  return (
    <li id={`${idPrefix}-ref-${number}`} className="reference" data-testid="reference">
      <span className="reference-number">{number}</span>
      <div>
        <p className="reference-title">
          {isHttpUrl(url) ? (
            <a href={url} target="_blank" rel="noopener noreferrer">
              {name}
            </a>
          ) : (
            <>
              {name} <span className="no-link">(no link available)</span>
            </>
          )}
        </p>
        <p className="reference-meta">
          {formatAuthors(authors)} · {year ?? "n.d."} · {section ? `${section}, ` : ""}
          {location}
        </p>
        {quotes.map((quote) => (
          <blockquote key={quote}>
            <span className="verified">Verified quote</span> {quote}
          </blockquote>
        ))}
      </div>
    </li>
  );
}

function ClaimList({ title, claims, idPrefix }) {
  if (!claims.length) return null;
  return (
    <section>
      <h3>{title}</h3>
      <ul className="claims">
        {claims.map((claim, i) => (
          <li key={i}>
            {claim.statement} <CitationMarks citations={claim.citations} idPrefix={idPrefix} />
          </li>
        ))}
      </ul>
    </section>
  );
}

export default function Answer({ answer, idPrefix, onFollowUp }) {
  const [label, hint, tone] = STATUS[answer.status] ?? [answer.status, "", "muted"];
  const findings = answer.claims.filter((c) => c.kind === "evidence");
  const interpretations = answer.claims.filter((c) => c.kind !== "evidence");
  const quotesOf = (number) => [
    ...new Set(
      answer.claims.flatMap((c) => c.citations.filter((q) => q.reference === number).map((q) => q.quote)),
    ),
  ];
  const { meta } = answer;

  return (
    <article className="answer" aria-label="Answer">
      <p className="status-row">
        <span className={`status status-${tone}`}>{label}</span>
        <span className="status-hint">{hint}</span>
      </p>

      {answer.summary && (
        <section>
          {findings.length > 0 && <h3>Model summary (not citation-checked)</h3>}
          <p>{answer.summary}</p>
        </section>
      )}

      <ClaimList title="Findings from the papers" claims={findings} idPrefix={idPrefix} />
      <ClaimList
        title="Interpretation (the assistant's reasoning, not a finding of the papers)"
        claims={interpretations}
        idPrefix={idPrefix}
      />

      {answer.ambiguity_note && <p className="note">Ambiguous terms: {answer.ambiguity_note}</p>}
      {answer.limitations && <p className="note">Limitations: {answer.limitations}</p>}

      {answer.references.length > 0 && (
        <section>
          <h3>References</h3>
          <ol className="references">
            {answer.references.map((reference) => (
              <Reference
                key={reference.number}
                reference={reference}
                quotes={quotesOf(reference.number)}
                idPrefix={idPrefix}
              />
            ))}
          </ol>
        </section>
      )}

      {answer.warnings.length > 0 && (
        <details className="warnings">
          <summary>Validation notes ({answer.warnings.length})</summary>
          <ul>
            {answer.warnings.map((warning, i) => (
              <li key={i}>{warning}</li>
            ))}
          </ul>
        </details>
      )}

      {answer.follow_ups.length > 0 && (
        <div className="follow-ups">
          {answer.follow_ups.map((text) => (
            <button type="button" key={text} className="chip" onClick={() => onFollowUp(text)}>
              {text}
            </button>
          ))}
        </div>
      )}

      <p className="meta">
        {meta.evidence_count} passages retrieved · best similarity {meta.best_similarity.toFixed(2)} ·{" "}
        {meta.gated ? "no model called" : meta.model}
      </p>
    </article>
  );
}
