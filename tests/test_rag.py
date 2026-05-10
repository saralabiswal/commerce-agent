"""
Tests for the sentence-transformers RAG pipeline.

These tests use a tiny fake embedding model so CI verifies Chroma behavior without
downloading model weights or depending on Hugging Face availability.
"""
from __future__ import annotations


class FakeEmbeddingModel:
    """Deterministic embedding model used to keep vector-search tests hermetic."""

    model_name = "test/fake-embeddings"
    dimension = 3

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        if isinstance(texts, str):
            texts = [texts]
        return [self.embed_one(text) for text in texts]

    def embed_one(self, text: str) -> list[float]:
        text = text.lower()
        if "amazon" in text or "title" in text:
            return [1.0, 0.0, 0.0]
        if "walmart" in text or "bullet" in text:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_chunk_text_terminates_on_final_overlap():
    """Regression test for the overlap bug that previously caused setup to be killed."""
    from rag.ingestion import chunk_text

    text = "A" * 900
    chunks = chunk_text(text, size=500, overlap=100)

    assert len(chunks) == 2
    assert chunks[-1] == "A" * 500


def test_ingest_stores_sentence_transformer_metadata(monkeypatch, temp_chroma):
    """Ingestion should persist vector metadata so stale dummy indexes can be detected."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    import rag.ingestion as ingestion

    monkeypatch.setattr(ingestion, "get_embedding_model", lambda: FakeEmbeddingModel())

    count = ingestion.ingest(persist_dir=temp_chroma, verbose=False)

    client = chromadb.PersistentClient(
        path=temp_chroma,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    collection = client.get_collection("commerce_docs", embedding_function=None)

    assert count == 30
    assert collection.count() == 30
    assert collection.metadata["embedding_model"] == FakeEmbeddingModel.model_name
    assert collection.metadata["embedding_dimension"] == FakeEmbeddingModel.dimension


def test_retriever_uses_vector_query_and_returns_metadata(monkeypatch, temp_chroma):
    """Retriever should query Chroma with embeddings and preserve source metadata."""
    import rag.ingestion as ingestion
    import rag.retrieval as retrieval

    monkeypatch.setattr(ingestion, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: FakeEmbeddingModel())

    ingestion.ingest(persist_dir=temp_chroma, verbose=False)
    retriever = retrieval.RAGRetriever(persist_dir=temp_chroma)

    hits = retriever.retrieve("amazon title character limits", top_k=3)

    assert retriever.document_count == 30
    assert len(hits) == 3
    assert all(hit["text"] for hit in hits)
    assert all(hit["source"] for hit in hits)
    assert all("relevance_score" in hit for hit in hits)


def test_retriever_rebuilds_stale_dummy_index(monkeypatch, temp_chroma):
    """Old collections without embedding metadata should be rebuilt before retrieval."""
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    import rag.ingestion as ingestion
    import rag.retrieval as retrieval

    monkeypatch.setattr(ingestion, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: FakeEmbeddingModel())

    client = chromadb.PersistentClient(
        path=temp_chroma,
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    stale = client.create_collection(
        name="commerce_docs",
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    stale.add(
        documents=["old dummy chunk"],
        metadatas=[{"source": "old.txt", "category": "general", "filename": "old.txt"}],
        ids=["old-1"],
        embeddings=[[0.0, 0.0, 0.0]],
    )

    retriever = retrieval.RAGRetriever(persist_dir=temp_chroma)

    assert retriever.document_count == 30
    assert retriever._collection.metadata["embedding_model"] == FakeEmbeddingModel.model_name


def test_retriever_refreshes_collection_handle_after_reingest(monkeypatch, temp_chroma):
    """A retriever created before re-ingestion should recover from stale Chroma IDs."""
    import rag.ingestion as ingestion
    import rag.retrieval as retrieval

    monkeypatch.setattr(ingestion, "get_embedding_model", lambda: FakeEmbeddingModel())
    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: FakeEmbeddingModel())

    ingestion.ingest(persist_dir=temp_chroma, verbose=False)
    retriever = retrieval.RAGRetriever(persist_dir=temp_chroma)
    assert retriever.document_count == 30

    ingestion.ingest(persist_dir=temp_chroma, verbose=False)
    hits = retriever.retrieve("walmart bullet requirements", top_k=2)

    assert len(hits) == 2
    assert all(hit["source"] for hit in hits)
