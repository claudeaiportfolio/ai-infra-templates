"""platform-core — shared infra plumbing for portfolio services.

Currently ships the ``queue`` module (RQ-on-Redis). Future modules (db, llm,
embeddings, otel, chunking) are added lazily as consumers need them, each behind
its own optional extra.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
