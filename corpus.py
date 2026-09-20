"""The information environment — a fixed document store + retriever the agent
searches. This is the Branch-B (offline) analogue of finqa_agent's TableWorld:
a REAL, deterministic, ungameable source of evidence.

Why a real retriever (not a judge-simulated one)? Same reason as finqa's real
table store: the grounded signal at the bottom of the reward has to be
ungameable. `search` returns the actual top-k passages BM25 ranks for the query;
`read` returns the actual bytes of a passage. An agent that "sounds right" but
never retrieved the gold evidence gets a low retrieval-hit-rate and a
groundedness penalty the reward (reward.py, TODO #1) can see.

Two corpus backends (config.corpus_backend), cheapest first:
  * "bundled"    — the passages that ship WITH each multi-hop question (~10 paragraphs:
                   2 gold supporting + 8 distractors). A tiny, fully-offline, per-question
                   corpus. No download, deterministic. Used for sanity + local. This is the
                   direct analogue of finqa's "gold table + distractor tables".
  * "wiki_index" — (cloud, TODO wiring) a full-Wikipedia retriever (Search-R1's E5/BM25
                   index) shared across all questions — the setup whose number is comparable
                   to the paper. Swapped in on the pod; the agent/tools/reward don't change.

The BM25 here is a small, dependency-free pure-Python implementation (Okapi BM25).
It's HARNESS, not learning logic — deterministic retrieval so tests are reproducible
and the loop runs with no network and no vector DB. For the real cloud run you can
point `corpus_backend="wiki_index"` at a stronger dense retriever; the DocStore
interface (search / get) stays identical so nothing downstream changes.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens. Deliberately simple + deterministic — the
    retriever is scaffolding, not the thing being learned."""
    return _WORD_RE.findall(text.lower())


@dataclass
class Doc:
    """One retrievable passage. `title` is the corpus key (HotpotQA/2Wiki index
    paragraphs by article title); `text` is the paragraph body."""
    title: str
    text: str

    def snippet(self, max_chars: int = 240) -> str:
        t = " ".join(self.text.split())
        return t if len(t) <= max_chars else t[:max_chars].rstrip() + "…"


@dataclass
class DocStore:
    """A fixed set of passages + a BM25 index over them. Read-only: the agent never
    mutates it, so (like TableWorld) there's no snapshot/restore to worry about.

    For the bundled backend this holds ONE question's ~10 paragraphs; for the
    wiki_index backend it would front a shared full-corpus index (same interface)."""
    docs: dict[str, Doc] = field(default_factory=dict)          # title -> Doc
    k1: float = 1.5
    b: float = 0.75

    # --- lazily built inverted index (populated in __post_init__) ---
    _df: dict[str, int] = field(default_factory=dict)          # token -> doc frequency
    _tf: dict[str, Counter] = field(default_factory=dict)      # title -> token counts
    _len: dict[str, int] = field(default_factory=dict)         # title -> doc length
    _avglen: float = 0.0
    _n: int = 0

    def __post_init__(self):
        self._build_index()

    def _build_index(self) -> None:
        self._df, self._tf, self._len = {}, {}, {}
        for title, doc in self.docs.items():
            toks = _tokenize(f"{title} {doc.text}")            # index the title too
            self._tf[title] = Counter(toks)
            self._len[title] = len(toks)
            for tok in set(toks):
                self._df[tok] = self._df.get(tok, 0) + 1
        self._n = len(self.docs)
        self._avglen = (sum(self._len.values()) / self._n) if self._n else 0.0

    def _idf(self, term: str) -> float:
        # Okapi BM25 idf with the usual +0.5 smoothing; floored at 0 so a term in
        # every doc doesn't go negative.
        n_qi = self._df.get(term, 0)
        return max(0.0, math.log((self._n - n_qi + 0.5) / (n_qi + 0.5) + 1.0))

    def _score(self, query_toks: list[str], title: str) -> float:
        tf, dl = self._tf[title], self._len[title]
        s = 0.0
        for term in query_toks:
            f = tf.get(term, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self._avglen or 1.0))
            s += self._idf(term) * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query: str, k: int = 3) -> list[tuple[Doc, float]]:
        """Top-k passages by BM25. Deterministic tie-break by title so runs are
        reproducible. Returns (Doc, score) pairs."""
        q = _tokenize(query)
        scored = [(self._score(q, title), title) for title in self.docs]
        scored.sort(key=lambda x: (-x[0], x[1]))               # score desc, title asc
        return [(self.docs[title], score) for score, title in scored[:k] if score > 0.0]

    def get(self, title: str) -> Doc | None:
        """Fetch one passage by exact title, else a unique case-insensitive match."""
        if title in self.docs:
            return self.docs[title]
        low = title.strip().lower()
        hits = [d for t, d in self.docs.items() if t.lower() == low]
        return hits[0] if len(hits) == 1 else None

    @property
    def titles(self) -> list[str]:
        return list(self.docs.keys())

    @staticmethod
    def from_passages(passages: Iterable[tuple[str, str]], **kw) -> "DocStore":
        """Build from (title, text) pairs — the shape data.py hands over per question."""
        docs = {title: Doc(title, text) for title, text in passages}
        return DocStore(docs=docs, **kw)
