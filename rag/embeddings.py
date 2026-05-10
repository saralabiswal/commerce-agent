"""
Embedding model wrapper.
Uses sentence-transformers locally — no API key, no cost, runs anywhere.
"""
from functools import lru_cache

from config import settings


class EmbeddingModel:
    """
    Thin wrapper around sentence-transformers for consistent embed() interface.
    Lazy-loads the model on first call to avoid startup overhead.
    """

    def __init__(self, model_name: str | None = None):
        self._model_name = model_name or settings.embedding_model
        self._model = None  # lazy load

    def _load(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self._model_name)

    def embed(self, texts: str | list[str]) -> list[list[float]]:
        """
        Embed one or more texts.

        Args:
            texts: A string or list of strings to embed.

        Returns:
            List of embedding vectors (each is a list of floats).
        """
        if self._model is None:
            self._load()

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_one(self, text: str) -> list[float]:
        """Embed a single string and return its vector."""
        return self.embed([text])[0]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return embedding dimension (384 for all-MiniLM-L6-v2)."""
        if self._model is None:
            self._load()
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Singleton embedding model — load once, reuse everywhere."""
    return EmbeddingModel()
