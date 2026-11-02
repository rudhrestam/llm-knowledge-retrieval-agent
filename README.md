# LLM Knowledge Retrieval Agent (Demo)

A small, runnable demo of a knowledge-retrieval agent for an operations team: ask a question in plain language, get back the best-matching passages from a set of runbooks and FAQs — with sources cited, and nothing invented.

> This is a representative demo of patterns used in production systems, built with sample documents and original code. It contains no employer code, data, or internal details.

## Why this pattern

Operations teams lose a lot of time hunting through runbooks, tickets, and documentation for answers that already exist somewhere. A retrieval agent fixes the finding problem — but only if it can be trusted. So this demo is built around one guardrail:

**The agent may only answer from retrieved passages. If nothing in the index matches well enough, it says so instead of guessing.**

That grounding rule is what separates a useful internal tool from a confident hallucination machine.

## Architecture

```
sample_docs/ ──► ingest ──► chunk (paragraphs) ──► TF-IDF index
                                                        │
 user question ──► vectorize ──► cosine similarity ──► top passages
                                                        │
                              score below threshold? ──► "not grounded, won't guess"
                                                        │
                              answer = passages + source citations
```

- **Ingest & chunk:** documents in `sample_docs/` are split into paragraph-sized passages.
- **Index:** passages are vectorized with scikit-learn TF-IDF.
- **Retrieve:** the question is vectorized the same way and ranked by cosine similarity.
- **Answer with citations:** the top passages are returned with their source file and match score. A minimum-score threshold enforces the grounding guardrail.

In a production version, the TF-IDF index can be swapped for embeddings and a vector store, and the passage list can feed an LLM that phrases the answer — the retrieval, citation, and threshold layers stay the same.

## Run it

```bash
pip install -r requirements.txt
python agent.py "how do I restart the reporting database?"
python agent.py "what happens if a batch job fails?"
python agent.py            # interactive mode
```

Example questions to try: restarting a service, batch failures, access requests, incident first response — and something off-topic (try "what's the weather?") to see the guardrail refuse to guess.

## Tech stack

Python · scikit-learn (TF-IDF, cosine similarity) · sample text documents

## What it demonstrates

- Retrieval-based agent design with grounded, cited answers
- A simple but real hallucination guardrail (score threshold + refusal)
- Clean separation of ingest / index / retrieve / answer stages
- A design that scales from TF-IDF to embeddings without restructuring
