from typing import Any

from embedding_service import EmbeddingService
from vector_store import VectorStore


class ProblemRetriever:
    """
    Retrieves relevant AlgoForge problems using
    semantic similarity + metadata filtering.
    """

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        documents: list[dict[str, Any]],
    ):
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.documents = documents

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        difficulty: str | None = None,
        topics: list[str] | None = None,
        paid_only: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        Retrieve relevant problems using semantic
        similarity and optional metadata filters.
        """

        if not query.strip():
            return []

        # ----------------------------------------------------
        # 1. Convert query → embedding
        # ----------------------------------------------------

        query_embedding = (
            self.embedding_service.encode(
                [query]
            )
        )

        # ----------------------------------------------------
        # 2. Retrieve a larger candidate pool from FAISS
        # ----------------------------------------------------

        candidate_k = max(top_k * 5, 20)

        scores, indices = (
            self.vector_store.search(
                query_embedding,
                top_k=candidate_k,
            )
        )

        # ----------------------------------------------------
        # 3. Apply metadata filters
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

            metadata = document["metadata"]

            # ------------------------------------------------
            # Difficulty filter
            # ------------------------------------------------

            if difficulty is not None:
                if metadata["difficulty"] != difficulty:
                    continue

            # ------------------------------------------------
            # Paid/free filter
            # ------------------------------------------------

            if paid_only is not None:
                if metadata["paid_only"] != paid_only:
                    continue

            # ------------------------------------------------
            # Topic filter
            # ------------------------------------------------

            if topics:
                problem_topics = set(
                    topic.lower()
                    for topic in metadata["topics"]
                )

                requested_topics = set(
                    topic.lower()
                    for topic in topics
                )

                if not problem_topics.intersection(
                    requested_topics
                ):
                    continue

            # ------------------------------------------------
            # Problem passed all filters
            # ------------------------------------------------

            results.append(
                {
                    "score": float(score),
                    "document": document,
                }
            )

        # ----------------------------------------------------
        # 4. Return final Top-K
        # ----------------------------------------------------

        return results[:top_k]