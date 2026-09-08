from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        """
        Create a FAISS vector store using inner product similarity.

        Since our embeddings are normalized, inner product
        is equivalent to cosine similarity.
        """

        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings: np.ndarray) -> None:
        """
        Add embeddings to the FAISS index.
        """

        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype("float32")

        self.index.add(embeddings)

        print(f"Added {len(embeddings)} vectors to FAISS.")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for the most similar vectors.

        Returns:
            scores
            indices
        """

        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype("float32")

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        return scores, indices

    def save(self, path: str | Path) -> None:
        """
        Save the FAISS index to disk.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(path),
        )

        print(f"Saved FAISS index to: {path}")

    def load(self, path: str | Path) -> None:
        """
        Load an existing FAISS index from disk.
        """

        path = Path(path)

        self.index = faiss.read_index(
            str(path),
        )

        print(f"Loaded FAISS index from: {path}")

    @property
    def size(self) -> int:
        """
        Number of vectors currently stored.
        """

        return self.index.ntotal