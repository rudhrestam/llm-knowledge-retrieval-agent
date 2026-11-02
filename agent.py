#!/usr/bin/env python3
"""
Knowledge retrieval agent (demo).

Indexes a small folder of sample runbook / FAQ documents with TF-IDF and
answers a question by returning the best-matching passages, with sources.
The agent only answers from retrieved passages: if nothing matches well
enough, it says so instead of guessing. That grounding rule is the point
of the demo.

Usage:
    python agent.py "how do I restart the reporting database?"
    python agent.py            # interactive mode
"""

import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS_DIR = Path(__file__).parent / "sample_docs"
MIN_SCORE = 0.08  # below this, the match is treated as "not grounded"
TOP_K = 3


def load_chunks(docs_dir: Path):
    """Read each document and split it into paragraph-sized chunks."""
    chunks = []  # list of (source_name, chunk_text)
    for path in sorted(docs_dir.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for para in text.split("\n\n"):
            para = " ".join(para.split())
            if len(para) > 60:  # skip headings / fragments
                chunks.append((path.name, para))
    return chunks


class KnowledgeAgent:
    def __init__(self, chunks):
        self.sources = [c[0] for c in chunks]
        self.texts = [c[1] for c in chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.matrix = self.vectorizer.fit_transform(self.texts)

    def ask(self, question: str, top_k: int = TOP_K):
        q_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(q_vec, self.matrix)[0]
        ranked = scores.argsort()[::-1][:top_k]
        results = [
            (self.sources[i], float(scores[i]), self.texts[i])
            for i in ranked
            if scores[i] >= MIN_SCORE
        ]
        return results


def format_answer(question: str, results) -> str:
    if not results:
        return (
            "I don't have grounded information on that in the indexed "
            "documents, so I won't guess. Try rephrasing, or check a "
            "runbook that covers the topic."
        )
    lines = [f"Q: {question}\n", "Best-matching passages from the knowledge base:\n"]
    for n, (source, score, text) in enumerate(results, 1):
        lines.append(f"{n}. [{source} · match {score:.2f}] {text}\n")
    lines.append(
        "Answer grounded in the passages above — no information outside "
        "the indexed documents was used."
    )
    return "\n".join(lines)


def main():
    chunks = load_chunks(DOCS_DIR)
    if not chunks:
        sys.exit(f"No sample documents found in {DOCS_DIR}")
    agent = KnowledgeAgent(chunks)
    print(f"Indexed {len(chunks)} passages from {len(set(agent.sources))} documents.\n")

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(format_answer(question, agent.ask(question)))
        return

    print("Ask a question (Ctrl+C to exit).")
    try:
        while True:
            question = input("\n> ").strip()
            if question:
                print(format_answer(question, agent.ask(question)))
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


if __name__ == "__main__":
    main()
