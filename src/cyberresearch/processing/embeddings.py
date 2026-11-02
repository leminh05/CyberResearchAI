"""Sentence Transformers embeddings."""

from __future__ import annotations

from typing import Protocol

from cyberresearch.config import EmbeddingSettings


class Embedder(Protocol):
    """Anything that turns texts into fixed-size vectors."""

    model_name: str
    dimension: int

    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class SentenceTransformerEmbedder:
    """Wraps a Sentence Transformers model running on the CPU.

    Vectors are L2-normalized, so cosine similarity equals the dot product
    and pgvector's cosine distance (``<=>``) ranks results correctly.
    The model is downloaded from Hugging Face on first use and then cached.
    """

    def __init__(self, settings: EmbeddingSettings) -> None:
        from sentence_transformers import SentenceTransformer  # heavy import, load lazily

        self.model_name = settings.model_name
        self.batch_size = settings.batch_size
        self.model = SentenceTransformer(settings.model_name, device="cpu")
        get_dim = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        self.dimension = int(get_dim())
        if self.dimension != settings.dimension:
            raise ValueError(
                f"{settings.model_name} produces {self.dimension}-d vectors but "
                f"settings.yaml expects {settings.dimension}; update embedding.dimension"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()
