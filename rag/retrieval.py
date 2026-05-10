"""
RAG retrieval — semantic search over ChromaDB stored chunks.
Uses local sentence-transformers embeddings by default.
"""
import os
import sys
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from rag.embeddings import get_embedding_model


class RAGRetriever:
    """
    Retrieves relevant document chunks using sentence-transformers vector search.
    """

    def __init__(self, persist_dir: str | None = None):
        self._persist_dir = persist_dir or settings.chroma_persist_dir
        self._embedding_model = get_embedding_model()
        self._collection = None

    def _ensure_collection(self):
        """Open the Chroma collection, rebuilding old indexes when needed."""
        if self._collection is not None:
            return

        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collection = client.get_collection("commerce_docs", embedding_function=None)
            metadata = collection.metadata or {}
            if metadata.get("embedding_model") != self._embedding_model.model_name:
                raise ValueError("RAG index uses a different embedding model")
            self._collection = collection
        except Exception:
            # ChromaDB not yet ingested, or an old dummy index exists.
            from rag.ingestion import ingest

            ingest(persist_dir=self._persist_dir, verbose=False)
            self._collection = None
            self._ensure_collection()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        category_filter: str | None = None,
    ) -> list[dict]:
        """Retrieve top-K chunks by semantic similarity."""
        self._ensure_collection()
        top_k = top_k or settings.rag_top_k

        where = {"category": category_filter} if category_filter else None
        query_embedding = self._embedding_model.embed_one(query)
        for attempt in range(2):
            try:
                results = self._collection.query(
                    query_embeddings=[query_embedding],
                    n_results=top_k,
                    where=where,
                    include=["documents", "metadatas", "distances"],
                )
                break
            except Exception:
                if attempt == 1:
                    raise
                self._collection = None
                self._ensure_collection()

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        chunks = []
        for text, meta, distance in zip(documents, metadatas, distances):
            chunks.append({
                "text": text,
                "source": meta.get("source", "unknown"),
                "category": meta.get("category", "general"),
                "filename": meta.get("filename", ""),
                "relevance_score": round(max(0.0, 1.0 - float(distance)), 3),
            })
        return chunks

    def retrieve_for_generation(
        self,
        product_category: str,
        retailer: str,
        product_type: str = "",
    ) -> str:
        """Retrieve and format context for content generation prompts."""
        queries = [
            f"{retailer} {product_category} title requirements character limit",
            f"{retailer} bullet points requirements best practices",
            f"{product_category} specifications requirements {product_type}",
        ]

        all_chunks = []
        seen_texts = set()

        for query in queries:
            for chunk in self.retrieve(query, top_k=3):
                if chunk["text"] not in seen_texts:
                    all_chunks.append(chunk)
                    seen_texts.add(chunk["text"])

        if not all_chunks:
            return "No specific retailer guidelines retrieved. Apply general best practices."

        lines = ["=== RETAILER & CATEGORY REQUIREMENTS (Retrieved Context) ===\n"]
        for chunk in all_chunks[:6]:
            lines.append(f"[Source: {chunk['source']}]")
            lines.append(chunk["text"])
            lines.append("")

        return "\n".join(lines)

    @property
    def document_count(self) -> int:
        self._ensure_collection()
        return self._collection.count()


@lru_cache(maxsize=1)
def get_retriever() -> RAGRetriever:
    return RAGRetriever()


def reset_retriever() -> None:
    """Clear the process-local retriever after Chroma collections are rebuilt."""
    get_retriever.cache_clear()
