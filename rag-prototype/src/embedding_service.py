from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Convert text documents into numerical embeddings.
        """

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.astype("float32")

    def save_embeddings(
        self,
        embeddings: np.ndarray,
        path: str | Path,
    ) -> None:
        """
        Save embeddings to a NumPy .npy file.
        """

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(path, embeddings)

        print(f"Saved embeddings to: {path}")