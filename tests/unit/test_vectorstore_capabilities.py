from __future__ import annotations

import inspect

from langchain_gaussdb import GaussDBVectorStore
from tests.helpers.vectorstore_fakes import DeterministicEmbeddings, RecordingEngine


def test_constructor_does_not_offer_a_separate_capability_probe_switch() -> None:
    parameters = inspect.signature(GaussDBVectorStore).parameters

    assert "validate_connection" not in parameters


def test_capability_probe_is_not_a_vectorstore_public_api() -> None:
    store = GaussDBVectorStore(
        embedding=DeterministicEmbeddings(),
        table_name="documents",
        embedding_dimension=3,
        engine=RecordingEngine(),
    )

    assert not hasattr(store, "validate_capabilities")
