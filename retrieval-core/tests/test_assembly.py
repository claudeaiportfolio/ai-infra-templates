"""Tests for context assembly. A deterministic word-count token counter is
injected so budgets are exact and readable; one test exercises the real
tiktoken default counter."""
from __future__ import annotations

import pytest

from retrieval_core import (
    AssemblyConfig,
    Candidate,
    assemble,
    default_citation,
    default_token_counter,
)


def words(text: str) -> int:
    return len(text.split())


def cands(*texts: str) -> list[Candidate]:
    return [Candidate(id=i + 1, text=t) for i, t in enumerate(texts)]


class TestPacking:
    def test_packs_until_budget(self) -> None:
        # Each formatted chunk = "[<id>]\n<text>" -> word count is 1 + len(words).
        config = AssemblyConfig(policy="top_k_by_fused", token_budget=8)
        out = assemble(
            cands("one two three", "four five six", "seven eight nine"),
            config=config, query="q", token_counter=words,
        )
        # chunk1 = 4 words, chunk2 = 4 words -> 8 == budget; chunk3 would overflow.
        assert out.chunks_used == 2
        assert out.tokens == 8

    def test_stops_at_first_overflow(self) -> None:
        config = AssemblyConfig(policy="top_k_by_fused", token_budget=3)
        out = assemble(cands("a b c d e"), config=config, query="q", token_counter=words)
        assert out.chunks_used == 0
        assert out.context == ""

    def test_preserves_input_order(self) -> None:
        config = AssemblyConfig(policy="rerank_then_top_k", token_budget=100)
        out = assemble(cands("first", "second", "third"), config=config, query="q", token_counter=words)
        assert out.context.index("first") < out.context.index("second") < out.context.index("third")
        assert out.policy == "rerank_then_top_k"


class TestCitation:
    def test_default_citation_is_id(self) -> None:
        config = AssemblyConfig(policy="top_k_by_fused", token_budget=100)
        out = assemble([Candidate(id=42, text="body")], config=config, query="q", token_counter=words)
        assert out.context.startswith("[42]\n")

    def test_custom_citation_fn(self) -> None:
        config = AssemblyConfig(policy="top_k_by_fused", token_budget=100)
        c = Candidate(id=1, text="body", metadata={"source": "report.pdf", "page": 7})
        out = assemble(
            [c], config=config, query="q", token_counter=words,
            citation_fn=lambda x: f"{x.metadata['source']} p.{x.metadata['page']}",
        )
        assert out.context.startswith("[report.pdf p.7]\n")

    def test_default_citation_helper(self) -> None:
        assert default_citation(Candidate(id="abc", text="t")) == "abc"


class TestCompression:
    def test_keeps_query_relevant_sentences(self) -> None:
        text = "The cat sat. The dog ran fast. A bird flew high above."
        config = AssemblyConfig(
            policy="rerank_then_compress", token_budget=100, compress_per_chunk_tokens=4
        )
        out = assemble(
            [Candidate(id=1, text=text)], config=config, query="dog ran", token_counter=words,
        )
        assert "dog ran fast" in out.context
        # the cap (4 words) excludes the other sentences
        assert "bird flew" not in out.context

    def test_single_sentence_truncates(self) -> None:
        config = AssemblyConfig(
            policy="rerank_then_compress", token_budget=100, compress_per_chunk_tokens=3
        )
        out = assemble(
            [Candidate(id=1, text="alpha beta gamma delta epsilon")],
            config=config, query="alpha", token_counter=words,
        )
        # truncation is by tokens of the counter; word counter -> first 3 words kept
        body = out.context.split("\n", 1)[1]
        assert body.split() == ["alpha", "beta", "gamma"]

    def test_non_compress_policy_keeps_full_text(self) -> None:
        config = AssemblyConfig(policy="rerank_then_top_k", token_budget=100)
        out = assemble(
            [Candidate(id=1, text="one two three four five")],
            config=config, query="one", token_counter=words,
        )
        assert "one two three four five" in out.context


class TestDefaultTokenCounter:
    def test_counts_with_tiktoken(self) -> None:
        # Smoke test the real default counter (tiktoken is a dev dependency).
        n = default_token_counter("hello world")
        assert isinstance(n, int)
        assert n >= 2

    def test_default_counter_used_when_omitted(self) -> None:
        config = AssemblyConfig(policy="top_k_by_fused", token_budget=1000)
        out = assemble([Candidate(id=1, text="hello world")], config=config, query="q")
        assert out.chunks_used == 1
        assert out.tokens > 0


class TestAssemblyConfigValidation:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"policy": "top_k_by_fused", "token_budget": 0}, "token_budget"),
            (
                {"policy": "rerank_then_compress", "token_budget": 10, "compress_per_chunk_tokens": 0},
                "compress_per_chunk_tokens",
            ),
        ],
    )
    def test_rejects_bad_values(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            AssemblyConfig(**kwargs)
