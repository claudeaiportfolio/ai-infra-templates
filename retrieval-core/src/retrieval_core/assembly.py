"""Context assembly — turning retrieved candidates into the exact text placed in
the model's window, under a fixed token budget.

This is where "context engineering" actually lives. More context is not
monotonically better, the budget is finite, and the right policy is
workload-dependent — so the policy is made **explicit and swappable** and meant
to be measured (policy x accuracy x tokens x latency) rather than inherited as a
framework default. Three policies:

- ``top_k_by_fused``      — pack ranked chunks until the budget (no rerank).
- ``rerank_then_top_k``   — pack reranked chunks until the budget.
- ``rerank_then_compress``— rerank, then extractively compress each chunk (keep
  the sentences most overlapping the query) so more distinct chunks fit.

The compressor is deterministic and model-free (query-term sentence overlap), so
it adds no model call and no nondeterminism. Token counting and citation
formatting are injected, so the package bakes in neither a tokenizer choice nor
any domain-specific metadata field names. The default token counter uses
``tiktoken`` (optional extra ``retrieval-core[tiktoken]``); pass your own
``token_counter`` to count for a different model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache

from retrieval_core.config import AssemblyConfig, AssemblyPolicy
from retrieval_core.types import Candidate, TokenCounter

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\w+")


@dataclass
class AssemblyResult:
    context: str
    chunks_used: int
    tokens: int
    policy: AssemblyPolicy


@lru_cache(maxsize=1)
def _default_encoder() -> object:
    try:
        import tiktoken
    except ImportError as exc:  # pragma: no cover - exercised via message only
        raise ImportError(
            "Default token counting needs tiktoken. Install retrieval-core[tiktoken] "
            "or pass an explicit token_counter to assemble()."
        ) from exc
    return tiktoken.get_encoding("cl100k_base")


def default_token_counter(text: str) -> int:
    """``cl100k_base`` token count (GPT-4 / text-embedding-3 family). Used when
    ``assemble`` is called without an explicit ``token_counter``."""
    enc = _default_encoder()
    return len(enc.encode(text))  # type: ignore[attr-defined]


def default_citation(chunk: Candidate) -> str:
    """Neutral citation: the candidate's id. Override via ``citation_fn`` to
    render domain metadata (e.g. ``source :: heading :: p.12``)."""
    return str(chunk.id)


def _format(chunk: Candidate, citation_fn: Callable[[Candidate], str], text: str | None = None) -> str:
    return f"[{citation_fn(chunk)}]\n{text if text is not None else chunk.text}"


def _truncate(text: str, max_tokens: int, count: TokenCounter) -> str:
    """Token-budgeted truncation. Uses a binary search over a character prefix so
    it works with any ``token_counter`` (no decode step assumed)."""
    if count(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text[:mid]) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def _compress(text: str, query: str, max_tokens: int, count: TokenCounter) -> str:
    """Extractive: keep whole sentences, ranked by query-term overlap, until the
    per-chunk token cap — preserving original order so the prose still reads."""
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s]
    if len(sentences) <= 1:
        return _truncate(text, max_tokens, count)
    terms = {w.lower() for w in _WORD.findall(query)}
    scored = [
        (i, sum(1 for w in _WORD.findall(s) if w.lower() in terms))
        for i, s in enumerate(sentences)
    ]
    keep: set[int] = set()
    used = 0
    for idx, _ in sorted(scored, key=lambda pair: pair[1], reverse=True):
        cost = count(sentences[idx])
        if used + cost > max_tokens:
            continue
        keep.add(idx)
        used += cost
    if not keep:  # even the best single sentence overflows — hard truncate it
        return _truncate(sentences[max(scored, key=lambda p: p[1])[0]], max_tokens, count)
    return " ".join(sentences[i] for i in sorted(keep))


def assemble(
    candidates: Sequence[Candidate],
    *,
    config: AssemblyConfig,
    query: str,
    token_counter: TokenCounter | None = None,
    citation_fn: Callable[[Candidate], str] | None = None,
) -> AssemblyResult:
    """Pack candidates into a context string under ``config.token_budget``.

    Order is whatever ``retrieve`` returned (RRF or reranked — the caller maps
    the policy to the rerank flag). ``rerank_then_compress`` compresses each
    chunk to fit more distinct evidence; the others pack whole chunks greedily.
    """
    count = token_counter or default_token_counter
    cite = citation_fn or default_citation

    parts: list[str] = []
    used = 0
    for chunk in candidates:
        if config.policy == "rerank_then_compress":
            budget_left = config.token_budget - used
            if budget_left <= 0:
                break
            text = _compress(
                chunk.text, query, min(config.compress_per_chunk_tokens, budget_left), count
            )
            formatted = _format(chunk, cite, text)
        else:
            formatted = _format(chunk, cite)
        cost = count(formatted)
        if used + cost > config.token_budget:
            break
        parts.append(formatted)
        used += cost
    return AssemblyResult(
        context="\n\n".join(parts),
        chunks_used=len(parts),
        tokens=used,
        policy=config.policy,
    )
