"""
RAG ingestion pipeline.
Stores document chunks in ChromaDB with local sentence-transformers embeddings.

Run once after setup: python rag/ingestion.py

Owner: Sarala Biswal
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from rag.embeddings import get_embedding_model

DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 32


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks while preferring sentence boundaries."""
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap < 0:
        raise ValueError("chunk overlap must be non-negative")
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for i in range(end - 1, max(start + size - 100, start), -1):
                if text[i] in (".", "\n"):
                    end = i + 1
                    break
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if len(c) > 50]


def load_documents() -> list[dict]:
    """Load RAG source documents from disk with category metadata."""
    docs = []
    for path in sorted(DOCUMENTS_DIR.rglob("*.txt")):
        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(DOCUMENTS_DIR)
        category = relative.parts[0] if len(relative.parts) > 1 else "general"
        docs.append({
            "source": str(relative),
            "category": category,
            "filename": path.name,
            "text": text,
        })
    return docs


def ingest(persist_dir: str | None = None, verbose: bool = True) -> int:
    """
    Ingest documents into ChromaDB with sentence-transformers embeddings.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    persist_dir = persist_dir or settings.chroma_persist_dir
    os.makedirs(persist_dir, exist_ok=True)
    embedding_model = get_embedding_model()

    client = chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )

    try:
        client.delete_collection("commerce_docs")
    except Exception:
        pass

    collection = client.create_collection(
        name="commerce_docs",
        metadata={
            "hnsw:space": "cosine",
            "embedding_model": embedding_model.model_name,
            "embedding_dimension": embedding_model.dimension,
        },
        embedding_function=None,
    )

    documents = load_documents()
    if verbose:
        print(f"📚 Found {len(documents)} documents to ingest")

    total_chunks = 0
    for doc in documents:
        chunks = chunk_text(doc["text"])
        if verbose:
            print(f"   {doc['source']}: {len(chunks)} chunks")

        ids = [f"{doc['filename']}_{i:04d}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": doc["source"],
                "category": doc["category"],
                "filename": doc["filename"],
                "chunk_index": i,
            }
            for i in range(len(chunks))
        ]

        for start in range(0, len(chunks), BATCH_SIZE):
            end = start + BATCH_SIZE
            batch_chunks = chunks[start:end]
            collection.add(
                documents=batch_chunks,
                metadatas=metadatas[start:end],
                ids=ids[start:end],
                embeddings=embedding_model.embed(batch_chunks),
            )
            total_chunks += len(batch_chunks)

    if verbose:
        print(
            f"\n✅ Ingested {total_chunks} chunks into ChromaDB at {persist_dir} "
            f"using {embedding_model.model_name}"
        )

    return total_chunks


if __name__ == "__main__":
    ingest(verbose=True)
