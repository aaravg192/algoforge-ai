from pathlib import Path

import numpy as np

from embedding_service import EmbeddingService
from embedding_builder import build_embedding_text
from filter_pipeline import load_cleaned_documents


BASE_PATH = (
    Path(__file__).resolve().parent.parent
)

INPUT_PATH = (
    BASE_PATH
    / "data"
    / "filtered_problems.json"
)

OUTPUT_PATH = (
    BASE_PATH
    / "data"
    / "problem_embeddings.npy"
)


def main():

    print(
        "=== AlgoForge Embedding Pipeline ==="
    )

    # --------------------------------------------------------
    # Load filtered problems
    # --------------------------------------------------------

    documents = load_cleaned_documents(
        INPUT_PATH
    )

    print(
        f"Loaded documents: {len(documents)}"
    )

    # --------------------------------------------------------
    # Build semantic text
    # --------------------------------------------------------

    embedding_texts = [
        build_embedding_text(document)
        for document in documents
    ]

    print(
        f"Built embedding texts: "
        f"{len(embedding_texts)}"
    )

    # --------------------------------------------------------
    # Load pretrained model
    # --------------------------------------------------------

    embedding_service = EmbeddingService()

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    embeddings = embedding_service.encode(
        embedding_texts
    )

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    print(
        f"Embedding dtype: {embeddings.dtype}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        OUTPUT_PATH,
        embeddings,
    )

    print(
        f"Saved embeddings to: {OUTPUT_PATH}"
    )

    print(
        "\n=== Embedding Pipeline Complete ==="
    )


if __name__ == "__main__":
    main()