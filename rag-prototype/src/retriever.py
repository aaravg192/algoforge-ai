from pathlib import Path
from typing import Any

from embedding_service import EmbeddingService
from vector_store import VectorStore


class ProblemRetriever:
    """
    Retrieves relevant AlgoForge problems using
    semantic vector similarity.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        documents: list[dict[str, Any]],
    ):
        self.embedding_service = (
            embedding_service
        )

        self.vector_store = vector_store

        self.documents = documents

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most semantically relevant problems.
        """

        if not query.strip():
            return []

        # ----------------------------------------------------
        # Convert query → embedding
        # ----------------------------------------------------

        query_embedding = (
            self.embedding_service.encode(
                [query]
            )
        )

        # ----------------------------------------------------
        # Search FAISS
        # ----------------------------------------------------

        scores, indices = (
            self.vector_store.search(
                query_embedding,
                top_k=top_k,
            )
        )

        # ----------------------------------------------------
        # Convert vector results → documents
        # ----------------------------------------------------

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):

            if index < 0:
                continue

            document = self.documents[
                int(index)
            ]

            results.append(
                {
                    "score": float(score),
                    "document": document,
                }
            )

        return results