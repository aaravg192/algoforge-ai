from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """
    FAISS vector store for AlgoForge problem embeddings.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension

        # Inner Product + normalized embeddings
        # = cosine similarity
        self.index = faiss.IndexFlatIP(
            dimension
        )

    def add(
        self,
        embeddings: np.ndarray,
    ) -> None:
        """
        Add embeddings to the FAISS index.
        """

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D numpy array."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Expected embedding dimension "
                f"{self.dimension}, "
                f"got {embeddings.shape[1]}"
            )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self.index.add(
            embeddings
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ):
        """
        Search the vector store for the most similar
        embeddings.
        """

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(
                1, -1
            )

        if query_embedding.shape[1] != self.dimension:
            raise ValueError(
                f"Expected query dimension "
                f"{self.dimension}, "
                f"got {query_embedding.shape[1]}"
            )

        top_k = min(
            top_k,
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        return scores, indices

    @property
    def size(self) -> int:
        """
        Number of vectors stored.
        """

        return self.index.ntotal

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Save FAISS index to disk.
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

        print(
            f"Saved FAISS index to: {path}"
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "VectorStore":
        """
        Load an existing FAISS index.
        """

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {path}"
            )

        index = faiss.read_index(
            str(path)
        )

        store = cls(
            index.d
        )

        store.index = index

        return store