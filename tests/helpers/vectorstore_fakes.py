from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Iterable

from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from langchain_gaussdb.vectorstore import GaussDBVectorStore


def mark_vectorstore_initialized(
    store: GaussDBVectorStore,
) -> None:
    """Skip automatic initialization in focused query/write unit tests."""
    store._initialized = True


class DeterministicEmbeddings(Embeddings):
    def __init__(
        self,
        vectors: Iterable[list[float]] | None = None,
        dimension: int = 3,
        query_vector: list[float] | None = None,
    ) -> None:
        self.vectors = None if vectors is None else list(vectors)
        self.dimension = dimension
        self.query_vector = None if query_vector is None else list(query_vector)
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []
        self.async_document_calls: list[list[str]] = []
        self.async_query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        if self.vectors is not None:
            return [list(vector) for vector in self.vectors]
        return [
            [float(index + offset) for offset in range(self.dimension)]
            for index, _ in enumerate(texts)
        ]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        if self.query_vector is not None:
            return list(self.query_vector)
        return [0.0 for _ in range(self.dimension)]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.async_document_calls.append(list(texts))
        if self.vectors is not None:
            return [list(vector) for vector in self.vectors]
        return [
            [float(index + offset) for offset in range(self.dimension)]
            for index, _ in enumerate(texts)
        ]

    async def aembed_query(self, text: str) -> list[float]:
        self.async_query_calls.append(text)
        if self.query_vector is not None:
            return list(self.query_vector)
        return [0.0 for _ in range(self.dimension)]


class RecordingEngine:
    def __init__(self, fetch_results=None) -> None:
        self.executed = []
        self.fetched = []
        self.calls = []
        self.fetch_results = deque([] if fetch_results is None else fetch_results)
        self.closed = False
        self.fail_execute_operations = set()
        self.fail_fetch_operations = set()

    def execute(self, compiled, *, operation: str = "execute") -> None:
        self.calls.append(("execute", operation))
        if operation in self.fail_execute_operations:
            raise RuntimeError(f"{operation} failed")
        self.executed.append((compiled, operation))

    def fetch_all(self, compiled, *, operation: str = "fetch_all"):
        self.calls.append(("fetch", operation))
        if operation in self.fail_fetch_operations:
            raise RuntimeError(f"{operation} failed")
        self.fetched.append((compiled, operation))
        return list(self.fetch_results.popleft()) if self.fetch_results else []

    def close(self) -> None:
        self.closed = True
