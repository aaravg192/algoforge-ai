from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    Generates semantic embeddings using a pretrained
    Sentence Transformer model.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        print(
            f"Loading embedding model: {self.MODEL_NAME}"
        )

        self.model = SentenceTransformer(
            self.MODEL_NAME
        )

        print("Embedding model loaded.")

    def encode(
        self,
        texts: list[str],
    ):
        """
        Convert text into dense vector embeddings.
        """

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        return embeddings