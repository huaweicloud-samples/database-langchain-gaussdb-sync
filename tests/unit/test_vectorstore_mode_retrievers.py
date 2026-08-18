from __future__ import annotations

import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_gaussdb import GaussDBVectorStore
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings, RecordingEngine


def _store(mode: str = "dense") -> GaussDBVectorStore:
    return GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        embedding_dimension=3,
        engine=RecordingEngine(),
        retrieval_mode=mode,
    )


@pytest.mark.parametrize("mode", ["dense", "bm25", "hybrid"])
def test_standard_retriever_uses_configured_store_mode(monkeypatch, mode: str) -> None:
    store = _store(mode)
    calls = []

    def search(selected_mode, query, k=4, **kwargs):
        calls.append((selected_mode, query, k, kwargs))
        return [(Document(id="doc-1", page_content="matched"), 1.0)]

    monkeypatch.setattr(store, "_similarity_search_with_score_for_mode", search)
    retriever = store.as_retriever(search_kwargs={"k": 2, "filter": {"tenant": "a"}})

    documents = retriever.invoke("needle")

    assert isinstance(retriever, BaseRetriever)
    assert [document.id for document in documents] == ["doc-1"]
    assert calls == [(mode, "needle", 2, {"filter": {"tenant": "a"}})]


def test_standard_retriever_keeps_langchain_search_type_contract() -> None:
    with pytest.raises(ValueError, match="search_type"):
        _store().as_retriever(search_type="hybrid")


def test_removed_mode_specific_retriever_factories_are_not_public() -> None:
    store = _store()

    assert not hasattr(store, "as_bm25_retriever")
    assert not hasattr(store, "as_hybrid_retriever")
